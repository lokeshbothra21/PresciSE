"""
Offline tests for the deepagents-based DDExpertAgent.

These tests build the agent against a hand-rolled BaseChatModel that scripts
tool calls deterministically (the langchain_core fakes don't implement
bind_tools, which the agent factory requires). They prove the public contract:

  * run() returns {"answer", "expert_qa", "all_retrieved_chunks"}.
  * The retrieve_evidence tool receives allowed_owners via runtime context
    and forwards it to the retriever.
  * record_answer side-effects into expert_qa.
  * A model exception is caught and returned as a friendly fallback dict
    instead of bubbling out as a 500.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ---------------------------------------------------------------------------
# Scripted fake chat model (supports bind_tools, unlike the langchain_core fakes)
# ---------------------------------------------------------------------------

class ScriptedChatModel(BaseChatModel):
    """Returns a scripted sequence of AIMessages. Supports tool binding."""

    responses: List[AIMessage]
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._idx >= len(self.responses):
            # Exhaust → emit a terminating, content-bearing AIMessage so the
            # agent loop can finish gracefully if it asks for more turns.
            msg = AIMessage(content="(end of script)")
        else:
            msg = self.responses[self._idx]
            self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


class RunawayRetrieveModel(BaseChatModel):
    """Records one answer, then calls retrieve_evidence forever — drives the
    agent into the LangGraph recursion limit to exercise the recovery path."""

    @property
    def _llm_type(self) -> str:
        return "runaway-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        # First assistant turn records an answer (so the fallback has material),
        # every later turn keeps retrieving and never synthesizes.
        n_ai = sum(1 for m in messages if getattr(m, "type", None) == "ai")
        if n_ai == 0:
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "record_answer",
                    "args": {"subquestion": "Q?", "answer": "Recorded answer."},
                    "id": f"rec",
                }],
            )
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_evidence", "args": {"subquestion": "again?"}, "id": f"r{n_ai}"}],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


class ExplodingChatModel(BaseChatModel):
    """Raises on first call. Verifies the agent's graceful-degradation guard."""

    @property
    def _llm_type(self) -> str:
        return "exploding-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        raise RuntimeError("simulated outage")

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve_with_formulas(self, tokens, embedding, top_k=8, allowed_owners=None):
        self.calls.append({"tokens": tokens, "allowed_owners": allowed_owners})
        return [
            {
                "score": 0.9,
                "bm25_score": 0.5,
                "faiss_score": 0.4,
                "chunk": {
                    "doc_id": "doc1",
                    "metadata": {"pages": [3]},
                    "text": "Ohm's law states V = IR.",
                    "chunk_type": "text",
                },
            }
        ]


class StubEmbedder:
    def embed_text(self, text):
        return [0.0] * 8


class FakeLLM:
    """Stand-in for GeminiClient. Only needs langchain_model() + generate()."""

    def __init__(self, chat_model):
        self._model = chat_model

    def langchain_model(self):
        return self._model

    def generate(self, *a, **k):  # used by run()'s mock fallback path
        return "MOCK"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _build_agent(chat_model):
    from core.agent.dd_expert_agent import DDExpertAgent
    retriever = StubRetriever()
    embedder = StubEmbedder()
    llm = FakeLLM(chat_model)
    agent = DDExpertAgent(retriever, embedder, llm, context_description="A small test corpus.")
    return agent, retriever


def test_run_returns_expected_contract_and_records_tools():
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "retrieve_evidence", "args": {"subquestion": "What is Ohm's law?"}, "id": "t1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "record_answer",
                    "args": {"subquestion": "What is Ohm's law?", "answer": "Ohm's law: V = IR."},
                    "id": "t2",
                }
            ],
        ),
        AIMessage(content="Ohm's law states V = IR."),
    ]
    model = ScriptedChatModel(responses=script)
    agent, retriever = _build_agent(model)

    out = agent.run("What is Ohm's law?", allowed_owners={"alice", "__shared__"})

    assert set(out.keys()) == {"answer", "expert_qa", "all_retrieved_chunks"}
    assert "V = IR" in out["answer"]
    assert out["expert_qa"] == [{"question": "What is Ohm's law?", "answer": "Ohm's law: V = IR."}]
    assert len(out["all_retrieved_chunks"]) == 1
    # allowed_owners was passed through to the retriever as a frozenset
    assert retriever.calls and retriever.calls[0]["allowed_owners"] == frozenset({"alice", "__shared__"})


def test_run_handles_model_exception_with_friendly_fallback():
    model = ExplodingChatModel()
    agent, _ = _build_agent(model)

    out = agent.run("anything", allowed_owners={"alice"})

    assert set(out.keys()) == {"answer", "expert_qa", "all_retrieved_chunks"}
    assert "temporary issue" in out["answer"]
    assert out["expert_qa"] == []
    assert out["all_retrieved_chunks"] == []


def test_extract_final_answer_never_echoes_user_query():
    """Regression guard: if the agent exits without a final AIMessage, the
    extractor must NOT walk back to the HumanMessage and return the user's
    own query as the answer. It returns "" so run() can recover."""
    from langchain_core.messages import HumanMessage, ToolMessage
    from core.agent.dd_expert_agent import _extract_final_answer
    fake_result = {
        "messages": [
            HumanMessage(content="Give me the summary of the uploaded document"),
            ToolMessage(content="some tool output", tool_call_id="t1"),
        ]
    }
    assert _extract_final_answer(fake_result) == ""


def test_missing_synthesis_recovers_from_recorded_qa():
    """If the loop ends without a final AIMessage but Q&A pairs were
    recorded, run() must recover via one direct synthesis call on the
    legacy generate() path instead of returning an apology."""
    # Script: retrieve → record → loop ends with an EMPTY final message
    # (simulates Gemini's sporadic MALFORMED_FUNCTION_CALL on the last turn).
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "retrieve_evidence", "args": {"subquestion": "What is Ohm's law?"}, "id": "t1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "record_answer",
                    "args": {"subquestion": "What is Ohm's law?", "answer": "Ohm's law: V = IR."},
                    "id": "t2",
                }
            ],
        ),
        AIMessage(content=""),  # empty final turn — no synthesis
    ]
    model = ScriptedChatModel(responses=script)
    agent, _ = _build_agent(model)
    # Track that the fallback used generate()
    agent.llm.generate = lambda prompt, **k: f"SYNTHESIZED FROM QA: {('V = IR' in prompt)}"

    out = agent.run("What is Ohm's law?")

    assert out["answer"] == "SYNTHESIZED FROM QA: True"
    assert out["expert_qa"] == [{"question": "What is Ohm's law?", "answer": "Ohm's law: V = IR."}]


def test_missing_synthesis_and_no_qa_retries_once_then_returns_retry_message():
    """An empty first turn (nothing produced at all) triggers ONE full-loop
    retry; if that also produces nothing, the user gets a polite retry
    message (not an echo, not a crash)."""
    # Two empty turns: one for the initial attempt, one for the retry.
    script = [AIMessage(content=""), AIMessage(content="")]
    model = ScriptedChatModel(responses=script)
    agent, _ = _build_agent(model)

    out = agent.run("anything")

    assert "without writing a final synthesis" in out["answer"]
    assert out["expert_qa"] == []
    # Both attempts consumed the script — proves the retry actually ran.
    assert model._idx == 2


def test_empty_first_attempt_recovers_on_retry():
    """First attempt dies on turn 1; the retry runs the full loop and
    produces a real answer."""
    script = [
        AIMessage(content=""),  # attempt 1: dead first turn
        # attempt 2: full successful loop
        AIMessage(
            content="",
            tool_calls=[
                {"name": "retrieve_evidence", "args": {"subquestion": "What is Ohm's law?"}, "id": "t1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "record_answer",
                    "args": {"subquestion": "What is Ohm's law?", "answer": "Ohm's law: V = IR."},
                    "id": "t2",
                }
            ],
        ),
        AIMessage(content="Ohm's law states V = IR."),
    ]
    model = ScriptedChatModel(responses=script)
    agent, retriever = _build_agent(model)

    out = agent.run("What is Ohm's law?")

    assert out["answer"] == "Ohm's law states V = IR."
    assert len(out["expert_qa"]) == 1
    # Retry used a fresh _RunState: chunks from the failed attempt don't leak.
    assert len(out["all_retrieved_chunks"]) == 1


def test_recursion_limit_recovers_via_fallback_synthesis(monkeypatch):
    """A runaway retrieve loop that hits the recursion limit must NOT surface
    the generic 'temporary issue' error — it should synthesize from whatever
    Q&A was recorded before the limit."""
    import core.agent.dd_expert_agent as mod
    # Shrink the recursion limit so the test hits it fast.
    monkeypatch.setattr(mod, "_RECURSION_LIMIT", 8)

    agent, _ = _build_agent(RunawayRetrieveModel())
    agent.llm.generate = lambda prompt, **k: "SYNTHESIZED DESPITE RECURSION LIMIT"

    out = agent.run("some open-ended question")

    assert out["answer"] == "SYNTHESIZED DESPITE RECURSION LIMIT"
    assert "temporary issue" not in out["answer"]
    assert len(out["expert_qa"]) >= 1  # the recorded answer survived


def test_transient_exception_on_first_attempt_recovers_on_retry():
    """A transient error (e.g. ReadTimeout) on attempt 1 must trigger a full
    retry rather than immediately returning the 'temporary issue' message."""
    # A model that raises once, then succeeds on the second agent invocation.
    counter = {"n": 0}

    class FlakyOnceModel(BaseChatModel):
        @property
        def _llm_type(self): return "flaky-fake"
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            # The very first model call (attempt 1, turn 1) raises; everything
            # after succeeds with a direct final answer.
            counter["n"] += 1
            if counter["n"] == 1:
                raise TimeoutError("simulated read timeout")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Recovered answer."))])
        def bind_tools(self, tools, **kwargs): return self  # type: ignore[override]

    agent, _ = _build_agent(FlakyOnceModel())
    out = agent.run("anything")
    assert out["answer"] == "Recovered answer."
    assert "temporary issue" not in out["answer"]


def test_run_mock_mode_when_llm_has_no_model():
    """When langchain_model() returns None (no API key), run() falls back to
    the GeminiClient.generate() mock string rather than building an agent."""
    class MockLessLLM:
        def langchain_model(self):
            return None
        def generate(self, *a, **k):
            return "MOCK MODE: hello"
    from core.agent.dd_expert_agent import DDExpertAgent
    agent = DDExpertAgent(StubRetriever(), StubEmbedder(), MockLessLLM(), context_description="x")
    out = agent.run("hi")
    assert out["answer"] == "MOCK MODE: hello"
    assert out["expert_qa"] == []
    assert out["all_retrieved_chunks"] == []
