"""Cross-encoder reranker for PresciSE (self-hosted, no external API).

Reranks (query, passage) pairs with BAAI/bge-reranker-v2-m3 — a strong open
reranker that runs on your own GPU/CPU, so document text never leaves the box.
A reranker is the single highest-ROI retrieval-quality lever: it re-scores a
wide candidate pool jointly against the query (unlike the bi-encoder + min-max
blend), surfacing the genuinely relevant chunks before they reach the LLM.

Lazy-loaded and thread-safe, mirroring Embedder. Construct it only when
reranking is enabled (see HybridRetriever / api wiring) so tests and mock mode
never download the ~2GB model.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger


def _passage_text(item: Dict[str, Any]) -> str:
    """Extract the text the reranker should score for a retrieved item.

    Formula chunks are scored on their LaTeX + surrounding context; text chunks
    on their body. Falls back to the raw ``text`` field.
    """
    chunk = item.get("chunk", {}) if isinstance(item, dict) else {}
    if chunk.get("chunk_type") == "formula":
        latex = (
            chunk.get("latex_formula")
            or chunk.get("normalized_formula")
            or chunk.get("formula_text")
            or ""
        )
        ctx = chunk.get("context_text", "")
        combined = f"{latex} {ctx}".strip()
        return combined or chunk.get("text", "")
    return chunk.get("text", "")


class Reranker:
    """Lazy, thread-safe cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = os.getenv("PRESCISE_RERANKER_MODEL", model_name)
        self._model = None
        # CrossEncoder.predict() is not safe to call concurrently on one model;
        # serialise load + inference behind a single lock (like Embedder).
        self._lock = threading.Lock()
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — torch always present in practice
            self.device = "cpu"

    def _load(self) -> None:
        """Lazy-load the cross-encoder (downloads on first use)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info(f"Loading reranker {self.model_name} on {self.device}...")
                    # Cap sequence length to bound per-pair cost (esp. on CPU).
                    self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
                    logger.info(f"Reranker loaded on {self.device}.")

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank retrieved ``items`` against ``query`` and return the top_k.

        Each returned item gets a ``rerank_score`` field. Items are sorted by
        that score descending. On any failure the original items (truncated to
        top_k) are returned unchanged, so a reranker hiccup degrades to plain
        hybrid ranking rather than failing the query.
        """
        if not items:
            return []
        if not query or not query.strip():
            return items[:top_k]
        try:
            import time as _t

            self._load()
            pairs = [(query, _passage_text(it)) for it in items]
            _t0 = _t.time()
            with self._lock:
                scores = self._model.predict(pairs)
            _dt = _t.time() - _t0
            # Surface slow rerank passes (the usual cause is running on CPU).
            (logger.info if _dt > 2.0 else logger.debug)(
                f"[RERANK] {len(pairs)} candidates on {self.device} in {_dt:.2f}s"
            )
            for it, s in zip(items, scores):
                it["rerank_score"] = float(s)
            ranked = sorted(items, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RERANK] failed ({exc}); falling back to hybrid order")
            return items[:top_k]
