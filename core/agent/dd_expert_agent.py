"""
DD+Expert iterative multi-agent answer system for PresciSE.

Replaces the single-pass ScientificAnswerAgent with a two-agent LangGraph loop:

  dd_plan → retrieve → expert_answer → dd_evaluate
                          ↑                  │ (not satisfied & iter < max)
                          └──────────────────┘
                                             │ (satisfied or iter >= max)
                                             ▼
                                       formula_plan → retrieve → expert_answer → dd_evaluate
                                                                                       │
                                                                                       ▼
                                                                                  synthesize → END

LLM call budget:
  Normal loop: 1 (plan) + N×2 (expert+evaluate) + 1 (formula_plan) + 2 (formula expert+eval) + 1 (synthesize)
  = max 14 LLM calls (N=5).  min 7 LLM calls (N=1).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from core.agent.dd_prompts import (
    DD_EVALUATE_PROMPT,
    DD_FORMULA_PLAN_PROMPT,
    DD_PLAN_PROMPT,
    DD_SYNTHESIZE_PROMPT,
    EXPERT_ANSWER_PROMPT,
)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class DDExpertState(TypedDict):
    query: str
    context_description: str
    subquestions: List[str]
    retrieved_per_subq: List[List[Dict[str, Any]]]
    expert_qa: List[Dict[str, str]]   # [{"question": str, "answer": str}, ...]
    iteration: int
    satisfied: bool
    final_answer: str
    subq_attempt_counts: Dict[str, int]  # normalised_q → attempt count
    formula_round_done: bool             # True once the forced formula round has executed
    parameter_priorities: Dict[str, int] # question_key → weight 1–3 (from DD_PLAN)
    parameter_scores: Dict[str, int]     # question_key → latest score 0–2 (from DD_EVALUATE)
    all_retrieved_chunks: List[Dict[str, Any]]  # every chunk seen across all retrieve() calls
    allowed_owners: Optional[set]        # owner-id scope for retrieval (None = no scoping)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DDExpertAgent:
    """
    Iterative DD+Expert LangGraph agent for scientific Q&A.

    Args:
        retriever: Any object with a ``retrieve_with_formulas(tokens, embedding, top_k)``
                   method returning a list of chunk-result dicts.
        embedder:  Object with an ``embed_text(text: str) -> List[float]`` method.
        llm:       GeminiClient (or any object with ``generate(prompt) -> str``).
        context_description: Corpus summary from ``build_context_description()``.
        top_k_per_subq: Chunks to retrieve per subquestion.
        max_iterations: Maximum retrieve/expert/evaluate cycles (default 5).
    """

    # Output-token cap for internal control-flow calls (plan / evaluate /
    # formula_plan). These emit short JSON, so this bounds worst-case
    # generation time without truncating valid output or touching the
    # expert/synthesis calls that produce the actual answer.
    _INTERNAL_MAX_TOKENS = 1024

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
        self.max_iterations = max_iterations

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(DDExpertState)

        workflow.add_node("dd_plan", self._dd_plan_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("expert_answer", self._expert_answer_node)
        workflow.add_node("dd_evaluate", self._dd_evaluate_node)
        workflow.add_node("formula_plan", self._formula_plan_node)
        workflow.add_node("synthesize", self._synthesize_node)

        workflow.set_entry_point("dd_plan")
        workflow.add_edge("dd_plan", "retrieve")
        workflow.add_edge("retrieve", "expert_answer")
        workflow.add_edge("expert_answer", "dd_evaluate")
        workflow.add_conditional_edges(
            "dd_evaluate",
            self._route_after_evaluate,
            {"synthesize": "synthesize", "formula_plan": "formula_plan", "retrieve": "retrieve"},
        )
        workflow.add_edge("formula_plan", "retrieve")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _route_after_evaluate(self, state: DDExpertState) -> str:
        """Three-way routing: continue normal loop → formula round → synthesize."""
        if state.get("formula_round_done", False):
            return "synthesize"
        if state["satisfied"] or state["iteration"] >= self.max_iterations:
            return "formula_plan"
        return "retrieve"

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _dd_plan_node(self, state: DDExpertState) -> DDExpertState:
        prompt = DD_PLAN_PROMPT.format(
            context_description=state["context_description"],
            query=state["query"],
        )
        try:
            raw = self.llm.generate(prompt, max_output_tokens=self._INTERNAL_MAX_TOKENS)
            parsed = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001
            from loguru import logger as _log
            _log.warning(f"[DD PLAN] planning failed ({exc}); falling back to single-subquestion plan")
            parsed = {"in_scope": True, "subquestions": [state["query"]], "priorities": {}}

        if not parsed.get("in_scope", True):
            reason = parsed.get("reason", "Query is outside the scope of the indexed documents.")
            state["final_answer"] = (
                f"I cannot answer this question from the indexed scientific documents. {reason}"
            )
            # Signal synthesize directly by marking satisfied with no subquestions
            state["subquestions"] = []
            state["satisfied"] = True
            state["subq_attempt_counts"] = {}
            state["parameter_priorities"] = {}
            state["parameter_scores"] = {}
        else:
            subquestions = parsed.get("subquestions", [state["query"]])
            state["subquestions"] = subquestions
            state["satisfied"] = False
            # Initialise attempt counts at 1 (first retrieve/expert pass is imminent)
            state["subq_attempt_counts"] = {q.strip().lower(): 1 for q in subquestions}
            # Extract priority weights (default 1 for any missing)
            raw_priorities = parsed.get("priorities", {})
            state["parameter_priorities"] = {
                q.strip().lower(): int(raw_priorities.get(q, 1))
                for q in subquestions
            }
            state["parameter_scores"] = {}

        state["retrieved_per_subq"] = []
        state["expert_qa"] = []
        state["iteration"] = 0
        state["formula_round_done"] = False
        state["all_retrieved_chunks"] = []
        return state

    def _retrieve_node(self, state: DDExpertState) -> DDExpertState:
        # Early exit: out-of-scope queries have no subquestions
        if not state["subquestions"]:
            return state

        from core.nlp.tokenizer import tokenize

        subquestions = state["subquestions"]
        # Batch-embed all subquestions in a single call instead of one
        # embed_text() per subquestion — identical vectors, far less per-call
        # overhead (and one batched GPU/CPU pass instead of N sequential ones).
        embeddings = self.embedder.embed_texts(subquestions)

        allowed_owners = state.get("allowed_owners")
        retrieved_per_subq: List[List[Dict[str, Any]]] = []
        for subq, embedding in zip(subquestions, embeddings):
            tokens = tokenize(subq)
            items = self.retriever.retrieve_with_formulas(
                tokens, embedding, top_k=self.top_k_per_subq, allowed_owners=allowed_owners
            )
            retrieved_per_subq.append(items)

        state["retrieved_per_subq"] = retrieved_per_subq

        # Accumulate every chunk seen — used by the agent API to build source lists.
        flat = [item for subq_items in retrieved_per_subq for item in subq_items]
        state["all_retrieved_chunks"] = state.get("all_retrieved_chunks", []) + flat

        return state

    def _expert_answer_node(self, state: DDExpertState) -> DDExpertState:
        subquestions = state["subquestions"]
        if not subquestions:
            return state

        # Prior Q&A: everything accumulated BEFORE this iteration's new answers
        prior_qa = state.get("expert_qa", [])
        prior_qa_text = _format_qa_pairs(prior_qa) if prior_qa else ""

        # Build combined prompt: all subquestions + their evidence blocks
        subq_evidence_parts = []
        for i, (subq, items) in enumerate(
            zip(subquestions, state["retrieved_per_subq"]), start=1
        ):
            evidence_lines = _format_evidence(items)
            subq_evidence_parts.append(
                f"Subquestion {i}: {subq}\n\nEvidence:\n{evidence_lines}"
            )

        subquestions_and_evidence = "\n\n---\n\n".join(subq_evidence_parts)
        prompt = EXPERT_ANSWER_PROMPT.format(
            established_facts=prior_qa_text or "(none — this is the first iteration)",
            subquestions_and_evidence=subquestions_and_evidence,
        )
        raw = self.llm.generate(prompt)
        answers = _parse_json_array(raw, expected_len=len(subquestions))

        # Append new Q&A pairs (accumulate across iterations)
        new_qa = [
            {"question": q, "answer": a}
            for q, a in zip(subquestions, answers)
        ]
        state["expert_qa"] = state.get("expert_qa", []) + new_qa
        return state

    def _dd_evaluate_node(self, state: DDExpertState) -> DDExpertState:
        state["iteration"] = state.get("iteration", 0) + 1

        if state["iteration"] >= self.max_iterations:
            state["satisfied"] = True
            return state

        qa_text = _format_qa_pairs(state["expert_qa"])
        prompt = DD_EVALUATE_PROMPT.format(
            query=state["query"],
            qa_pairs=qa_text,
        )
        raw = self.llm.generate(prompt, max_output_tokens=self._INTERNAL_MAX_TOKENS)
        parsed = _parse_json(raw)

        # Extract per-subquestion scores and accumulate into parameter_scores
        answer_scores = parsed.get("answer_scores", {})
        current_scores: Dict[str, int] = state.get("parameter_scores", {})
        for q_text, score in answer_scores.items():
            key = q_text.strip().lower()
            current_scores[key] = int(score)
        state["parameter_scores"] = current_scores

        if parsed.get("satisfied", True):
            state["satisfied"] = True
        else:
            state["satisfied"] = False
            refined = parsed.get("refined_subquestions", [])
            if refined:
                # --- Per-subquestion attempt cap ---
                counts: Dict[str, int] = state.get("subq_attempt_counts", {})
                _MAX_ATTEMPTS = 3

                surviving: List[str] = []
                for q in refined:
                    key = q.strip().lower()
                    attempts = counts.get(key, 0)
                    if attempts >= _MAX_ATTEMPTS:
                        from loguru import logger as _log
                        _log.debug(
                            f"DD: retiring subquestion after {attempts} attempts: {q!r}"
                        )
                    else:
                        counts[key] = attempts + 1
                        surviving.append(q)

                state["subq_attempt_counts"] = counts

                if surviving:
                    state["subquestions"] = surviving
                else:
                    # All refined topics exhausted — go to formula round
                    state["satisfied"] = True
            else:
                state["satisfied"] = True  # nothing new to ask → stop

        return state

    def _formula_plan_node(self, state: DDExpertState) -> DDExpertState:
        """
        Forced formula retrieval round — always runs once after the main loop ends.
        Generates 4 targeted formula subquestions and feeds them back into retrieve.
        """
        from loguru import logger as _log
        _log.info(f"[FORMULA ROUND] Generating formula subquestions for: {state['query']!r}")

        prompt = DD_FORMULA_PLAN_PROMPT.format(query=state["query"])
        try:
            raw = self.llm.generate(prompt, max_output_tokens=self._INTERNAL_MAX_TOKENS)
            parsed = _parse_json(raw)
            subquestions = parsed.get("subquestions", [])
            if not isinstance(subquestions, list) or len(subquestions) == 0:
                raise ValueError("No subquestions in response")
            subquestions = subquestions[:4]
        except Exception as exc:
            _log.warning(f"[FORMULA ROUND] LLM failed ({exc}), using fallback questions")
            q = state["query"].rstrip("?")
            subquestions = [
                f"What is the mathematical formula or equation for {q}?",
                f"What are the variables, constants, and units in the {q} equation?",
                f"What physical laws govern {q}?",
                f"What related mathematical relationships or derivations apply to {q}?",
            ]

        _log.info(f"[FORMULA ROUND] Subquestions: {subquestions}")
        state["subquestions"] = subquestions
        state["formula_round_done"] = True
        state["satisfied"] = False  # force another expert+evaluate cycle
        return state

    def _synthesize_node(self, state: DDExpertState) -> DDExpertState:
        # Out-of-scope: final_answer was already set in dd_plan
        if not state["expert_qa"]:
            return state

        # Compute completeness metrics for the synthesis prompt
        scores = state.get("parameter_scores", {})
        priorities = state.get("parameter_priorities", {})

        weighted_sum = sum(scores.get(k, 0) * priorities.get(k, 1) for k in priorities)
        max_possible = sum(2 * p for p in priorities.values())
        overall_pct = (weighted_sum / max_possible * 100) if max_possible > 0 else 100.0

        # Formula score: find the subquestion with priority == 2
        formula_score = next(
            (scores.get(k, 0) for k, p in priorities.items() if p == 2), 0
        )
        # Only caveat when the answer is GENUINELY incomplete — not merely
        # because no formula was found. Conceptual questions legitimately have
        # no formula, and a formula-centric disclaimer there is noise (and reads
        # as evasive). Driven by overall coverage alone now.
        show_caveat = overall_pct < 35.0

        qa_text = _format_qa_pairs(state["expert_qa"])
        prompt = DD_SYNTHESIZE_PROMPT.format(
            query=state["query"],
            qa_pairs=qa_text,
            show_caveat=show_caveat,
            formula_score=formula_score,
            overall_pct=round(overall_pct, 1),
        )
        try:
            state["final_answer"] = self.llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            from loguru import logger as _log
            _log.warning(f"[SYNTHESIZE] final synthesis failed ({exc}); assembling gathered findings")
            parts = [qa["answer"] for qa in state.get("expert_qa", []) if qa.get("answer")]
            if parts:
                state["final_answer"] = (
                    "Note: final synthesis was temporarily unavailable, so here are the "
                    "gathered findings:\n\n" + "\n\n".join(parts)
                )
            else:
                state["final_answer"] = (
                    "I couldn't complete the analysis due to a temporary issue. Please try again."
                )
        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, allowed_owners: Optional[set] = None) -> Dict[str, Any]:
        """
        Run the DD+Expert pipeline for a query.

        Args:
            query: the user question.
            allowed_owners: optional set of owner ids to scope retrieval to
                (e.g. {user_id, "__shared__"}). None disables scoping.

        Returns:
            Dict with at least ``{"answer": str}``.
        """
        initial: DDExpertState = {
            "query": query,
            "context_description": self.context_description,
            "subquestions": [],
            "retrieved_per_subq": [],
            "expert_qa": [],
            "iteration": 0,
            "satisfied": False,
            "final_answer": "",
            "subq_attempt_counts": {},
            "formula_round_done": False,
            "parameter_priorities": {},
            "parameter_scores": {},
            "all_retrieved_chunks": [],
            "allowed_owners": allowed_owners,
        }
        try:
            result = self._graph.invoke(initial)
        except Exception as exc:  # noqa: BLE001
            # Last-resort guard: a node failure (e.g. LLM/embedder unavailable
            # after retries) returns a clean message instead of a 500 + stack.
            from loguru import logger as _log
            _log.error(f"[DD AGENT] pipeline failed: {exc}")
            return {
                "answer": (
                    "I couldn't complete this request due to a temporary issue "
                    "(the language model or retrieval was briefly unavailable). "
                    "Please try again in a moment."
                ),
                "expert_qa": [],
                "all_retrieved_chunks": [],
            }
        return {
            "answer": result["final_answer"],
            "expert_qa": result["expert_qa"],
            "all_retrieved_chunks": result.get("all_retrieved_chunks", []),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _parse_json_array(text: str, expected_len: int) -> List[str]:
    """Parse JSON array from LLM output; pad/truncate to expected_len."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            answers = [str(x) for x in result]
            # Pad if shorter, truncate if longer
            while len(answers) < expected_len:
                answers.append("The available documents don't directly address this aspect.")
            return answers[:expected_len]
    except json.JSONDecodeError:
        pass
    # Fallback: return the raw text as a single answer for the first subq
    fallback = [text] + ["The available documents don't directly address this aspect."] * (expected_len - 1)
    return fallback[:expected_len]


def _selection_reason(item: Dict[str, Any]) -> str:
    """Classify a retrieved chunk by which signal drove its selection."""
    b = item.get("bm25_score", 0.0)
    f = item.get("faiss_score", 0.0)
    chunk = item.get("chunk", {})
    src = chunk.get("extraction_source", "") or chunk.get("chunk_type", "")
    is_formula = chunk.get("chunk_type") == "formula"

    if is_formula:
        if b > f * 1.5:
            return "formula — keyword match"
        if f > b * 1.5:
            return "formula — semantic match"
        return "formula — combined match"
    else:
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
    return "\n\n".join(lines)


def _format_qa_pairs(qa: List[Dict[str, str]]) -> str:
    """Format accumulated Q&A pairs for evaluation/synthesis prompts."""
    parts = []
    for i, pair in enumerate(qa, start=1):
        parts.append(f"Q{i}: {pair['question']}\nA{i}: {pair['answer']}")
    return "\n\n".join(parts)
