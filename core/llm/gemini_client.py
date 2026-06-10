"""
LangChain-based Gemini LLM wrapper for PresciSE.

Replaces custom GeminiClient with LangChain's ChatGoogleGenerativeAI.
"""

import os
import threading
import time
from typing import List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class LLMError(RuntimeError):
    """Raised when a Gemini generation call fails (after retries)."""


class _StatsCallback(BaseCallbackHandler):
    """Bridges LangChain LLM events into GeminiClient's per-request stats.

    The deepagents agent calls the chat model directly through LangChain,
    bypassing GeminiClient.generate(). This callback updates the same
    thread-local stats so [REQUEST STATS] keeps reporting accurate
    llm_calls / llm_time across both code paths.

    A thread-local "inside_generate" flag prevents double-counting when
    generate() is the caller — generate() already records the call itself.
    """

    def __init__(self, client: "GeminiClient") -> None:
        self._client = client

    def _stats(self):
        return self._client._stats_obj()

    # The unused params (serialized/messages/prompts/response/error) are part
    # of LangChain's BaseCallbackHandler interface — required by signature
    # even though these hooks only track timing/counts. Prefixed with _ to
    # quiet linters.
    def on_chat_model_start(self, _serialized, _messages, **_kwargs):  # noqa: ANN001
        s = self._stats()
        if getattr(s, "inside_generate", False):
            return
        s._cb_t0 = time.time()

    def on_llm_start(self, _serialized, _prompts, **_kwargs):  # noqa: ANN001
        s = self._stats()
        if getattr(s, "inside_generate", False):
            return
        s._cb_t0 = time.time()

    def on_llm_end(self, _response, **_kwargs):  # noqa: ANN001
        s = self._stats()
        if getattr(s, "inside_generate", False):
            return
        s.calls += 1
        t0 = getattr(s, "_cb_t0", None)
        if t0 is not None:
            s.llm_time += time.time() - t0

    def on_llm_error(self, _error, **_kwargs):  # noqa: ANN001
        s = self._stats()
        if getattr(s, "inside_generate", False):
            return
        s.calls += 1
        t0 = getattr(s, "_cb_t0", None)
        if t0 is not None:
            s.llm_time += time.time() - t0


# Transient errors worth retrying. We retry broadly on Exception because the
# google-genai stack raises many provider-specific types for rate limits and
# 5xx; non-retryable errors (e.g. auth) simply exhaust the 3 attempts quickly.
_RETRYABLE = (Exception,)


class GeminiClient:
    """
    LangChain-based Gemini wrapper with rate limiting.
    
    Uses ChatGoogleGenerativeAI from langchain-google-genai for:
    - Standardized LLM interface
    - Better error handling
    - Streaming support
    - Token counting
    - Easy model swapping
    
    Rate Limit: 10 requests per minute to conserve API quota.
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.2):
        """
        Initialize Gemini LLM with LangChain and rate limiting.

        Args:
            model_name: Gemini model identifier. If None, reads
                PRESCISE_GEMINI_MODEL (default: gemini-3.5-flash). Notes:
                - gemini-3.5-flash is the default: it handles the agentic
                  tool-calling loop reliably (no empty-response bug) and is a
                  GA model (not -preview).
                - gemini-2.5-flash also works but its DYNAMIC thinking budget
                  intermittently returns empty responses; we pin
                  thinking_budget to work around that (see below).
                - gemini-2.5-flash-lite is too weak for the loop (skips tools,
                  echoes the query) — only for single-shot completion use.
            temperature: Sampling temperature (0.0-1.0)
        """
        if model_name is None:
            model_name = os.getenv("PRESCISE_GEMINI_MODEL", "gemini-3.5-flash")
        # Check both GEMINI_API_KEY and GOOGLE_API_KEY (fallback)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.is_mock = False

        # Thinking budget (gemini-2.5 family). The DEFAULT dynamic budget
        # (thinking_budget unset / -1) intermittently returns an EMPTY
        # response — finish_reason=STOP with 0 output tokens — on tool-bound
        # requests for some queries (~50% repro on certain questions). The
        # model allocates its whole output budget to internal thinking and
        # emits nothing. Pinning a small FIXED budget eliminated this in
        # testing (0/6 empty at 128 and 512, vs 3/6 with the dynamic
        # default). 256 is a comfortable middle. Override with
        # PRESCISE_THINKING_BUDGET (0 disables thinking entirely; -1 restores
        # the buggy dynamic default).
        try:
            self.thinking_budget = int(os.getenv("PRESCISE_THINKING_BUDGET", "256"))
        except ValueError:
            self.thinking_budget = 256
        
        # Rate limiting: requests per minute, configurable via PRESCISE_RATE_LIMIT.
        # Default 60 (suits a paid tier and lets a single ~12-call query run
        # without waiting); set higher for more headroom, or lower for free tier
        # (~15). The lock guards request_times so concurrent queries (each runs
        # in a thread-pool worker) share one global budget instead of racing.
        try:
            self.rate_limit = max(1, int(os.getenv("PRESCISE_RATE_LIMIT", "60")))
        except ValueError:
            self.rate_limit = 60
        self.request_times: List[float] = []
        self._rate_lock = threading.Lock()

        # Per-request call stats. Thread-local so each query (which runs in its
        # own thread-pool worker) gets an isolated count even under concurrency.
        self._stats = threading.local()

        # Stats callback: bumps thread-local counters whenever LangChain runs a
        # chat-model call. This is the bridge that lets [REQUEST STATS] keep
        # working when the agent calls the model via deepagents (no longer
        # routed through this class's generate() method).
        self._stats_callback = _StatsCallback(self)

        if not api_key:
            print("WARNING: GEMINI_API_KEY not found. Using Mock Mode.")
            self.is_mock = True
            self.llm = None
        else:
            try:
                # LangChain's rate limiter is bucket-based: requests_per_second
                # = rate_limit / 60. The bucket size of 1 means no burst; raise
                # max_bucket_size for headroom on paid tiers if needed.
                rate_limiter = InMemoryRateLimiter(
                    requests_per_second=self.rate_limit / 60.0,
                    check_every_n_seconds=0.1,
                    max_bucket_size=max(1, self.rate_limit // 4),
                )
                # Fail fast on hung calls. Without a timeout the HTTP client
                # will wait until the upstream load-balancer drops the
                # connection (~90s), which has bitten us as 84s "Server
                # disconnected" failures with no retry. 60s is a comfortable
                # ceiling for the heavier gemini-3.x models (a 45s cap caused
                # spurious ReadTimeouts on slow calls); retries=2 keeps the
                # overall worst-case bounded even with backoff.
                # convert_system_message_to_human was needed for legacy Gemini
                # models that lacked a system role. Modern Gemini handles
                # system messages natively, and demoting the prompt to a
                # human turn weakens tool-calling directives — the agent
                # would reply with a preamble like "I will summarize..."
                # instead of actually calling retrieve_evidence.
                model_kwargs = dict(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    request_timeout=60,
                    retries=2,
                    rate_limiter=rate_limiter,
                    callbacks=[self._stats_callback],
                )
                # Pin the thinking budget unless explicitly set to the dynamic
                # default (-1), which is the buggy empty-response path.
                if self.thinking_budget >= 0:
                    model_kwargs["thinking_budget"] = self.thinking_budget
                self.llm = ChatGoogleGenerativeAI(**model_kwargs)
                self.model_name = model_name
                # Surface the active model at startup so we don't have to
                # guess whether env overrides took effect.
                print(
                    f"[GeminiClient] active model: {model_name} "
                    f"(rate {self.rate_limit}/min, thinking_budget={self.thinking_budget})"
                )
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini Client ({e}). Using Mock Mode.")
                self.is_mock = True
                self.llm = None

    def _wait_for_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.
        
        Rate: 10 requests per minute (1 request every 6 seconds).

        The lock is held across the sleep on purpose: it serialises concurrent
        callers so they queue for the shared budget rather than all sleeping
        against a stale snapshot of request_times.
        """
        with self._rate_lock:
            current_time = time.time()

            # Remove timestamps older than 60 seconds
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            # If we've made `rate_limit` requests in the last minute, wait
            if len(self.request_times) >= self.rate_limit:
                oldest_request = self.request_times[0]
                wait_time = 60 - (current_time - oldest_request) + 0.1  # Small buffer

                if wait_time > 0:
                    print(f"[RATE LIMIT] Waiting {wait_time:.1f}s to stay within {self.rate_limit} req/min...")
                    time.sleep(wait_time)

                    # Re-clean request times after waiting
                    current_time = time.time()
                    self.request_times = [t for t in self.request_times if current_time - t < 60]

            # Record this request
            self.request_times.append(current_time)

    # ------------------------------------------------------------------
    # Per-request call statistics (thread-local)
    # ------------------------------------------------------------------
    def _stats_obj(self):
        if not hasattr(self._stats, "calls"):
            self._stats.calls = 0
            self._stats.llm_time = 0.0
        return self._stats

    def begin_request(self) -> None:
        """Reset per-request LLM stats. Call once at the start of a request."""
        s = self._stats_obj()
        s.calls = 0
        s.llm_time = 0.0

    def request_stats(self) -> dict:
        """Return {llm_calls, llm_time_s} accumulated since begin_request()."""
        s = self._stats_obj()
        return {"llm_calls": s.calls, "llm_time_s": round(s.llm_time, 2)}

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _invoke(self, prompt: str, max_output_tokens: "int | None" = None) -> str:
        """Single rate-limited LLM call, retried with exponential backoff."""
        self._wait_for_rate_limit()
        llm = self.llm if max_output_tokens is None else self.llm.bind(
            max_output_tokens=max_output_tokens
        )
        response = llm.invoke(prompt)
        content = response.content
        # Newer Gemini models may return content as a list of parts
        if isinstance(content, list):
            content = " ".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )
        return content

    def generate(self, prompt: str, max_output_tokens: "int | None" = None) -> str:
        """
        Generate a response from the LLM, rate-limited and retried.

        Args:
            prompt: Input prompt string
            max_output_tokens: optional cap on generated tokens. Use for internal
                control-flow calls (planning/evaluation) that emit short JSON, to
                bound generation time without affecting the final answer. Leave
                None for the expert/synthesis calls that produce the answer.

        Returns:
            Generated text response.

        Raises:
            LLMError: if the call fails after all retries. Callers must handle
            this (previously failures were silently returned as a string and
            corrupted downstream JSON parsing).
        """
        stats = self._stats_obj()
        t0 = time.time()
        # Suppress the LangChain callback's stats accounting while we're inside
        # this method, so generate()'s own finally-block is the single source
        # of truth for legacy callers.
        stats.inside_generate = True
        try:
            if self.is_mock:
                return (
                    "MOCK MODE (MOCK RESPONSE):\n"
                    "Based on the retrieved evidence, here is a summary:\n"
                    "- The evidence discusses Michaelis-Menten kinetics and its enzymatic reaction rates.\n"
                    "- It also mentions Allosteric regulation and conformational shifts.\n"
                    "(Note: This is a simulated response because the API key was missing or invalid.)"
                )
            try:
                return self._invoke(prompt, max_output_tokens=max_output_tokens)
            except Exception as e:
                raise LLMError(f"Gemini generation failed after retries: {e}") from e
        finally:
            # Count the logical call (including failures and any rate-limit
            # waits / retries folded into the elapsed time).
            stats.calls += 1
            stats.llm_time += time.time() - t0
            stats.inside_generate = False

    def langchain_model(self) -> Optional[ChatGoogleGenerativeAI]:
        """
        Return the underlying ChatGoogleGenerativeAI instance for direct use
        by LangChain harnesses (e.g. deepagents.create_deep_agent). Stats and
        rate limiting are wired in via callbacks/rate_limiter attached to the
        model at init, so callers don't lose telemetry. Returns None in mock
        mode (no API key).
        """
        return self.llm