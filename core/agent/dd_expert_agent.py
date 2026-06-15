"""
DD+Expert scientific Q&A agent, now built on the deepagents harness.

This replaces the previous hand-coded LangGraph StateGraph (dd_plan → retrieve
→ expert_answer → dd_evaluate ⇄ retrieve → formula_plan → … → synthesize)
with a single autonomous deep agent. The LLM uses write_todos to plan
subquestions, calls retrieve_evidence + record_answer per subquestion, and
writes the final synthesized answer as its last message.

The public class name (DDExpertAgent), constructor, and run() contract are
preserved so callers (api/main.py, scripts/ask.py, scripts/bench.py,
evals/generate_predictions.py) need no changes.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain.tools import ToolRuntime, tool
from langgraph.errors import GraphRecursionError
from loguru import logger

from core.agent.dd_prompts import SYSTEM_PROMPT

# deepagents/langgraph serialize the AgentContext when snapshotting graph state,
# which emits noisy (harmless) "Pydantic serializer warnings" on every step.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


# Bound on the number of tool-calling cycles per query (LangGraph
# recursion_limit). A healthy 3–4 subquestion run is ~20-30 graph steps
# (each model call + each tool call counts as a step). 40 gives headroom for
# a couple of refinement retries while ensuring a runaway retrieve loop is
# cut off — and recovered via fallback synthesis — in seconds, not minutes.
# (gemini-3.5-flash tends to over-retrieve; a higher limit just wastes
# wall-clock before the recovery path kicks in.)
_RECURSION_LIMIT = 40

# T2 — abstention threshold. A subquestion's evidence is flagged "low relevance"
# when the top reranked match's confidence (sigmoid of the cross-encoder score)
# falls below this. Soft signal: the model is told to abstain rather than guess,
# but evidence is not dropped. Tune per reranker via PRESCISE_RERANK_MIN_SCORE.
_RERANK_MIN_SCORE = float(os.getenv("PRESCISE_RERANK_MIN_SCORE", "0.30"))


# ---------------------------------------------------------------------------
# Per-invocation state
# ---------------------------------------------------------------------------

@dataclass
class _RunState:
    """Mutable per-query bookkeeping that tools write into.

    A fresh instance is created inside DDExpertAgent.run() for every call, so
    concurrent queries running in different worker threads never share state.
    """
    expert_qa: List[Dict[str, str]] = field(default_factory=list)
    all_retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    best_rerank_score: float = 0.0  # max reranker confidence seen this run (T2)
    progress: Optional[Callable[[str], None]] = None  # streaming progress hook (P2)


@dataclass
class AgentContext:
    """Runtime context passed to every tool invocation.

    Carries the per-user retrieval scope and the mutable run state. deepagents
    delivers this to tools via `runtime: ToolRuntime[AgentContext]`.
    """
    allowed_owners: Optional[frozenset] = None
    run_state: _RunState = field(default_factory=_RunState)


# ---------------------------------------------------------------------------
# Helpers (reused from the previous implementation)
# ---------------------------------------------------------------------------

def _emit(ctx, msg: str) -> None:
    """Best-effort progress callback for streaming (P2). Never raises."""
    try:
        if ctx is not None and ctx.run_state is not None and ctx.run_state.progress:
            ctx.run_state.progress(msg)
    except Exception:  # noqa: BLE001
        pass


def _relevance_note(items: List[Dict[str, Any]], min_score: float):
    """Independent weak-evidence signal from the reranker (T2).

    Returns (note, confidence). ``confidence`` = sigmoid(top rerank score) in
    (0,1), or None when items carry no rerank_score (reranking off / failed).
    ``note`` is a non-empty 'low relevance' prefix only when confidence < min_score.
    """
    if not items:
        return "", None
    top = items[0].get("rerank_score")
    if top is None:
        return "", None
    conf = 1.0 / (1.0 + math.exp(-float(top)))
    if conf < min_score:
        note = (
            f"(LOW RELEVANCE — best match scored {conf:.2f}/1.00. The corpus may not "
            f"cover this subquestion. If the evidence below does not clearly answer "
            f"it, say the documents don't cover it rather than guessing, and do not "
            f"re-retrieve the same subquestion.)\n"
        )
        return note, conf
    return "", conf


def _selection_reason(item: Dict[str, Any]) -> str:
    """Classify a retrieved chunk by which signal drove its selection."""
    b = item.get("bm25_score", 0.0)
    f = item.get("faiss_score", 0.0)
    chunk = item.get("chunk", {})
    is_formula = chunk.get("chunk_type") == "formula"

    if is_formula:
        if b > f * 1.5:
            return "formula — keyword match"
        if f > b * 1.5:
            return "formula — semantic match"
        return "formula — combined match"
    if b > f * 1.5:
        return "keyword match"
    if f > b * 1.5:
        return "semantic match"
    return "combined match"


def _format_evidence(items: List[Dict[str, Any]]) -> str:
    """Format retrieved chunk items as an evidence block string."""
    if not items:
        return "(no evidence retrieved)"
    lines = []
    for rank, item in enumerate(items, start=1):
        chunk = item["chunk"]
        doc_id = chunk.get("doc_id", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        page_num = pages[0] if pages and pages[0] > 0 else chunk.get("page_number", "?")
        reason = _selection_reason(item)
        citation = f"[{doc_id}, page {page_num}] ({reason})"

        if chunk.get("chunk_type") == "formula":
            latex = chunk.get("latex_formula", "")
            if latex:
                context = chunk.get("context_text", "")
                text = f"[FORMULA]${latex}$[/FORMULA]"
                if context:
                    text += f"\nContext: {context}"
            else:
                text = chunk.get("text", "")
        else:
            text = chunk.get("text", "")

        lines.append(f"  [{rank}] {citation}\n  {text}")
    # Wrap in delimiters so the model treats this as untrusted DATA, not
    # instructions (prompt-injection isolation — see SYSTEM_PROMPT "SECURITY").
    return "<untrusted_evidence>\n" + "\n\n".join(lines) + "\n</untrusted_evidence>"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DDExpertAgent:
    """Autonomous deep-agent driver for scientific Q&A.

    Args mirror the previous implementation:
        retriever: HybridRetriever (must expose retrieve_with_formulas(...))
        embedder:  Embedder (exposes embed_text(text) -> list[float])
        llm:       GeminiClient (exposes langchain_model() -> ChatGoogleGenerativeAI | None)
        context_description: corpus summary embedded into the system prompt
        top_k_per_subq:  chunks to retrieve per subquestion (default 8)
        max_iterations:  retained for API compatibility; ignored by the
                         autonomous loop (the LLM judges when to stop, bounded
                         by _RECURSION_LIMIT)
    """

    # Max retrieve_evidence calls per query. Bounds the retrieve→refine loop so
    # an open-ended question can't thrash until it hits the recursion limit.
    # Covers 3–4 subquestions plus a refinement or two; past this the tool
    # refuses and tells the model to synthesize from what it has.
    _RETRIEVAL_CAP = int(os.getenv("PRESCISE_MAX_RETRIEVALS", "6"))

    def __init__(
        self,
        retriever,
        embedder,
        llm,
        context_description: str,
        top_k_per_subq: int = 8,
        max_iterations: int = 5,
    ) -> None:
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm
        self.context_description = context_description
        self.top_k_per_subq = top_k_per_subq
        self.max_iterations = max_iterations  # kept for caller compatibility

        self._agent = self._build_agent()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_agent(self):
        """Construct the deepagents agent with retriever-bound tools.

        Tools are defined as closures so they can reach the retriever/embedder
        without threading those through the runtime context. Per-query state
        (allowed_owners, run_state) is still carried via AgentContext.
        """
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from core.nlp.tokenizer import tokenize

        # Trim deepagents' default tool surface. Gemini chokes on the full
        # built-in roster (write_todos + 6 filesystem tools + execute + task
        # + our 2) — we've observed finish_reason=MALFORMED_FUNCTION_CALL on
        # the first model call, which empties the AIMessage and causes the
        # agent to exit with no synthesis. For a RAG agent we only need
        # write_todos (planning) plus our own retrieve_evidence /
        # record_answer. Profile registration is idempotent under the same
        # key so re-instantiating DDExpertAgent is safe.
        register_harness_profile(
            "google_genai",
            HarnessProfile(
                excluded_tools=frozenset(
                    {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task"}
                ),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )

        retriever = self.retriever
        embedder = self.embedder
        top_k = self.top_k_per_subq
        retrieval_cap = self._RETRIEVAL_CAP

        @tool
        def retrieve_evidence(subquestion: str, runtime: ToolRuntime) -> str:
            """Retrieve evidence chunks from the indexed scientific corpus for a single subquestion.

            Returns a formatted evidence block with up to top_k chunks, each
            annotated with [doc, page] and the selection reason
            (keyword/semantic/formula). Use one call per subquestion.
            """
            ctx: AgentContext = runtime.context
            allowed_owners = ctx.allowed_owners if ctx else None

            # Hard cap on retrievals per query. Some open-ended questions send
            # the model into a retrieve→refine→retrieve loop that never
            # converges and blows the LangGraph recursion limit. Past the cap,
            # refuse further retrieval and tell the model to synthesize from
            # what it already has.
            if ctx is not None:
                ctx.run_state.retrieval_count += 1
                if ctx.run_state.retrieval_count > retrieval_cap:
                    logger.warning(
                        f"[AGENT TOOL] retrieval cap ({retrieval_cap}) hit; "
                        f"instructing model to synthesize"
                    )
                    return (
                        f"(Retrieval limit reached — you have already gathered "
                        f"evidence {retrieval_cap} times. Do NOT call retrieve_evidence "
                        f"again. Use record_answer for any remaining subquestions, then "
                        f"write your final synthesized answer now from what you have.)"
                    )

            _emit(ctx, f"Searching the documents for: {subquestion[:70]}")
            try:
                import time as _t
                tokens = tokenize(subquestion)
                _e0 = _t.time()
                embedding = embedder.embed_text(subquestion)
                _embed_s = _t.time() - _e0
                _r0 = _t.time()
                items = retriever.retrieve_with_formulas(
                    tokens, embedding, top_k=top_k, allowed_owners=allowed_owners,
                    query_str=subquestion,
                )
                logger.info(
                    f"[AGENT TOOL] retrieve_evidence timing: embed={_embed_s:.2f}s "
                    f"retrieve+rerank={_t.time() - _r0:.2f}s ({len(items)} items)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[AGENT TOOL] retrieve_evidence failed: {type(exc).__name__}: {exc}"
                )
                return "(retrieval temporarily unavailable for this subquestion)"

            # Push every chunk into the per-run accumulator so the API can
            # surface them as sources alongside the final answer.
            if ctx is not None:
                ctx.run_state.all_retrieved_chunks.extend(items)
            _emit(ctx, f"Reviewed {len(items)} passages")

            # T2: independent weak-evidence signal from the reranker score.
            note, conf = _relevance_note(items, _RERANK_MIN_SCORE)
            if conf is not None and ctx is not None:
                ctx.run_state.best_rerank_score = max(ctx.run_state.best_rerank_score, conf)
            return note + _format_evidence(items)

        @tool
        def record_answer(subquestion: str, answer: str, runtime: ToolRuntime) -> str:
            """Record the answer to one subquestion so it appears in the final response payload.

            Call this once per subquestion after writing its evidence-grounded
            answer. Returns a short acknowledgement.
            """
            ctx: AgentContext = runtime.context
            if ctx is not None:
                ctx.run_state.expert_qa.append({"question": subquestion, "answer": answer})
            _emit(ctx, f"Drafted an answer for: {subquestion[:70]}")
            return "recorded"

        system_prompt = SYSTEM_PROMPT.format(context_description=self.context_description)

        model = self.llm.langchain_model() if self.llm is not None else None
        if model is None:
            # Mock/no-key mode: don't build a real agent; run() short-circuits.
            return None

        agent = create_deep_agent(
            model=model,
            tools=[retrieve_evidence, record_answer],
            system_prompt=system_prompt,
            context_schema=AgentContext,
        )
        return agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        allowed_owners: Optional[set] = None,
        history: Optional[List[Dict[str, str]]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the agent for one query.

        Args:
            query: user question.
            allowed_owners: per-user retrieval scope; e.g. {user_id, "__shared__"}.

        Returns:
            {"answer": str, "expert_qa": list, "all_retrieved_chunks": list}
        """
        # Mock mode: no LLM available — return the GeminiClient mock string so
        # local dev without an API key still gets a clean response.
        if self._agent is None:
            try:
                mock = self.llm.generate(query) if self.llm is not None else ""
            except Exception:  # noqa: BLE001
                mock = ""
            return {
                "answer": mock or "(LLM unavailable.)",
                "expert_qa": [],
                "all_retrieved_chunks": [],
            }

        owners = frozenset(allowed_owners) if allowed_owners is not None else None
        if on_progress:
            try:
                on_progress("Planning your question…")
            except Exception:  # noqa: BLE001
                pass

        # Per-user, freshly-computed corpus list, injected into the prompt at run
        # time (not baked into the system prompt at build). Fixes two issues: the
        # build-time description was GLOBAL (could leak other users' filenames
        # into the prompt) and went STALE after new uploads. getattr default
        # keeps test stubs without a .chunks attribute working.
        from core.agent.context_builder import build_context_description
        corpus_desc = build_context_description(
            getattr(self.retriever, "chunks", []), allowed_owners=owners
        )
        user_content = (
            "Documents indexed and available to you (the current user's corpus):\n"
            f"{corpus_desc}\n\nQuestion: {query}"
        )

        # P1: prepend prior conversation turns so follow-up questions ("what about
        # at higher temperature?") resolve against the session's context. Capped
        # by the caller (api passes the last few turns).
        messages: List[Dict[str, str]] = []
        for turn in (history or []):
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "").strip()
            if q:
                messages.append({"role": "user", "content": q})
            if a:
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_content})

        # Gemini sporadically returns MALFORMED_FUNCTION_CALL on the very
        # first turn, which yields an empty AIMessage and an agent run with
        # nothing retrieved and nothing recorded. That's unrecoverable by the
        # fallback synthesis (no Q&A to work from), so retry the whole loop
        # once before giving up. A fresh _RunState per attempt keeps the
        # accumulators clean.
        answer = ""
        run_state = _RunState()
        _ATTEMPTS = 2
        for attempt in range(1, _ATTEMPTS + 1):
            last = attempt == _ATTEMPTS
            run_state = _RunState(progress=on_progress)
            context = AgentContext(allowed_owners=owners, run_state=run_state)

            try:
                result = self._agent.invoke(
                    {"messages": messages},
                    context=context,
                    config={"recursion_limit": _RECURSION_LIMIT},
                )
            except GraphRecursionError:
                # The loop ran long without converging (e.g. a runaway
                # retrieve→refine cycle). Whatever Q&A was recorded is still in
                # run_state — synthesize from it rather than discarding the work.
                logger.warning(
                    f"[DD AGENT] recursion limit ({_RECURSION_LIMIT}) hit after "
                    f"{run_state.retrieval_count} retrievals; synthesizing from "
                    f"{len(run_state.expert_qa)} recorded Q&A pair(s)"
                )
                return {
                    "answer": self._fallback_synthesis(query, run_state),
                    "expert_qa": run_state.expert_qa,
                    "all_retrieved_chunks": run_state.all_retrieved_chunks,
                }
            except Exception as exc:  # noqa: BLE001
                # Transient model/retrieval errors (ReadTimeout, server
                # disconnects, 5xx) — retry the whole run before giving up,
                # since a fresh attempt often succeeds. Only surface the error
                # message on the final attempt.
                logger.error(
                    f"[DD AGENT] attempt {attempt}/{_ATTEMPTS} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if not last:
                    continue
                return {
                    "answer": (
                        "I couldn't complete this request due to a temporary issue "
                        "(the language model or retrieval was briefly unavailable). "
                        "Please try again in a moment."
                    ),
                    "expert_qa": run_state.expert_qa,
                    "all_retrieved_chunks": run_state.all_retrieved_chunks,
                }

            # Diagnostic dump of the entire message stream — gated by env var
            # so it's easy to flip on when debugging the "no synthesis
            # produced" / "query echoed" class of failures and off in steady
            # state. Enable with: PRESCISE_AGENT_DEBUG=1
            if os.getenv("PRESCISE_AGENT_DEBUG", "0") == "1":
                _dump_messages(result)

            answer = _extract_final_answer(result)
            if answer or run_state.expert_qa:
                break  # got a synthesis, or enough material for the fallback
            logger.warning(
                f"[DD AGENT] attempt {attempt}: model produced no synthesis and no "
                f"recorded Q&A (likely a malformed first tool call); "
                f"{'retrying once' if not last else 'giving up'}"
            )
        if not answer:
            # Gemini sometimes ends the loop without writing the final
            # synthesis (e.g. a sporadic MALFORMED_FUNCTION_CALL empties the
            # last AIMessage). If the research loop did record Q&A pairs, we
            # have everything needed — recover with one direct synthesis call
            # instead of surfacing an apology.
            answer = self._fallback_synthesis(query, run_state)
        answer = self._maybe_verify(answer, run_state)
        return {
            "answer": answer,
            "expert_qa": run_state.expert_qa,
            "all_retrieved_chunks": run_state.all_retrieved_chunks,
        }

    def _fallback_synthesis(self, query: str, run_state: _RunState) -> str:
        """Recover the final answer when the agent loop ended without one.

        Uses the recorded expert Q&A pairs and a single direct llm.generate()
        call (rate-limited + retried by GeminiClient). If even that fails, or
        nothing was recorded, fall back to the recorded answers verbatim or a
        retry message.
        """
        from core.agent.dd_prompts import FALLBACK_SYNTHESIS_PROMPT

        if not run_state.expert_qa:
            logger.warning("[DD AGENT] no synthesis and no recorded Q&A — asking user to retry")
            return (
                "I wasn't able to produce a full answer this turn — the agent finished "
                "without writing a final synthesis. Please retry the question."
            )

        qa_text = "\n\n".join(
            f"Q{i}: {p['question']}\nA{i}: {p['answer']}"
            for i, p in enumerate(run_state.expert_qa, start=1)
        )
        logger.warning(
            f"[DD AGENT] loop ended without synthesis; recovering from "
            f"{len(run_state.expert_qa)} recorded Q&A pair(s) via direct synthesis call"
        )
        try:
            return self.llm.generate(
                FALLBACK_SYNTHESIS_PROMPT.format(query=query, qa_pairs=qa_text)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[DD AGENT] fallback synthesis failed ({exc}); returning raw findings")
            parts = [p["answer"] for p in run_state.expert_qa if p.get("answer")]
            return (
                "Here is what I found on each aspect of your question:\n\n"
                + "\n\n".join(parts)
            )

    def _maybe_verify(self, answer: str, run_state: "_RunState") -> str:
        """T1 — optional answer-time faithfulness check.

        Off unless PRESCISE_ENABLE_FAITHFULNESS_CHECK=1 (it adds one LLM call).
        Grades whether the answer's claims are supported by the retrieved
        evidence; if not, appends a soft caveat. Never raises — verification
        must never break the answer.
        """
        if os.getenv("PRESCISE_ENABLE_FAITHFULNESS_CHECK", "0") != "1":
            return answer
        if not answer or not run_state.all_retrieved_chunks:
            return answer

        seen, parts, total = set(), [], 0
        for it in run_state.all_retrieved_chunks:
            ch = it.get("chunk", {})
            cid = ch.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            if ch.get("chunk_type") == "formula":
                t = ch.get("latex_formula") or ch.get("normalized_formula") or ch.get("text", "")
            else:
                t = ch.get("text", "")
            t = (t or "").strip()
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total > 6000:  # bound the verifier's context/cost
                break
        if not parts:
            return answer

        from core.agent.dd_prompts import FAITHFULNESS_CHECK_PROMPT

        try:
            raw = self.llm.generate(
                FAITHFULNESS_CHECK_PROMPT.format(answer=answer, evidence="\n\n".join(parts)),
                max_output_tokens=120,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[FAITHFULNESS] check failed ({type(exc).__name__}); skipping")
            return answer

        if raw.upper().startswith("UNSUPPORTED"):
            reason = raw.split(":", 1)[1].strip() if ":" in raw else ""
            logger.info(f"[FAITHFULNESS] flagged unsupported: {reason or '(unspecified)'}")
            return (
                answer
                + "\n\n_Note: an automated check flagged that some statements above may "
                "not be fully supported by the provided documents — please verify against "
                "the cited sources._"
            )
        return answer


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _dump_messages(result: Any) -> None:
    """Log every message in the agent result so we can see exactly what the
    model produced. Useful when llm_calls=1 but no synthesis appeared — tells
    us whether the model emitted empty content, returned a finish_reason
    indicating a safety block, called a tool we didn't expect, etc.
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    logger.info(f"[AGENT DEBUG] {len(messages)} messages in result:")
    for i, msg in enumerate(messages):
        mtype = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "?")
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        tool_calls = getattr(msg, "tool_calls", None) or []
        # response_metadata sometimes carries finish_reason / safety info
        meta = getattr(msg, "response_metadata", None) or {}
        finish = meta.get("finish_reason") if isinstance(meta, dict) else None
        snippet = (str(content)[:240] + "…") if content and len(str(content)) > 240 else content
        tc_summary = [f"{tc.get('name', '?')}({list(tc.get('args', {}).keys())})" for tc in tool_calls]
        logger.info(
            f"  [{i}] type={mtype} finish={finish} tool_calls={tc_summary} "
            f"content={snippet!r}"
        )


def _extract_final_answer(result: Any) -> str:
    """Pull the agent's final text reply out of a deepagents invoke() result.

    Only AIMessages count as the agent's answer. Tool messages are skipped,
    and we must NEVER fall back to the HumanMessage — doing so would echo
    the user's own query back as the "answer" when the model exits the loop
    without producing a final synthesis (a real failure mode we've seen).

    Returns the most recent AIMessage with non-empty text content, or ""
    if the model produced no synthesized answer (the caller recovers via
    fallback synthesis from the recorded Q&A pairs).
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
        # Accept only the assistant's own messages. Skip human / tool / system.
        if msg_type != "ai":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, list):
            # Gemini may return content as a list of parts
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(p.get("text", ""))
            content = " ".join(s for s in parts if s)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""
