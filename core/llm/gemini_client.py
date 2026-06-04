"""
LangChain-based Gemini LLM wrapper for PresciSE.

Replaces custom GeminiClient with LangChain's ChatGoogleGenerativeAI.
"""

import os
import threading
import time
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class LLMError(RuntimeError):
    """Raised when a Gemini generation call fails (after retries)."""


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

    def __init__(self, model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.2):
        """
        Initialize Gemini LLM with LangChain and rate limiting.
        
        Args:
            model_name: Gemini model identifier
            temperature: Sampling temperature (0.0-1.0)
        """
        # Check both GEMINI_API_KEY and GOOGLE_API_KEY (fallback)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.is_mock = False
        
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

        if not api_key:
            print("WARNING: GEMINI_API_KEY not found. Using Mock Mode.")
            self.is_mock = True
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    convert_system_message_to_human=True  # Gemini compatibility
                )
                self.model_name = model_name
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