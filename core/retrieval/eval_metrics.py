"""Retrieval-only evaluation metrics (offline, no LLM judge).

Isolates *retrieval* quality from answer generation so you can tune embeddings,
the reranker, and chunking objectively and cheaply (RAGAS conflates the two and
is LLM-judge-noisy/expensive). Relevance is labeled per query by gold doc ids
and/or "must-contain" substrings — both authorable by reading the source.
"""

from typing import Any, Dict, List, Optional, Sequence


def is_relevant(chunk: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """True if a retrieved ``chunk`` satisfies the query ``item``'s relevance label.

    A chunk matches if its doc_id is in ``relevant_doc_ids`` OR its text/formula
    contains any of ``relevant_contains`` (case-insensitive substring).
    """
    doc_ids = set(item.get("relevant_doc_ids", []) or [])
    if doc_ids and chunk.get("doc_id") in doc_ids:
        return True
    subs = item.get("relevant_contains", []) or []
    if subs:
        text = (
            (chunk.get("text") or "")
            + " " + (chunk.get("latex_formula") or "")
            + " " + (chunk.get("normalized_formula") or "")
        ).lower()
        return any(str(s).lower() in text for s in subs)
    return False


def rank_of_first_relevant(ranked: Sequence[Dict[str, Any]], item: Dict[str, Any]) -> Optional[int]:
    """1-based rank of the first relevant retrieved item, or None if none match.

    Accepts either retrieval-result dicts ({"chunk": {...}}) or raw chunk dicts.
    """
    for i, r in enumerate(ranked, start=1):
        chunk = r.get("chunk", r) if isinstance(r, dict) else {}
        if is_relevant(chunk, item):
            return i
    return None


def query_metrics(rank: Optional[int], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Per-query metrics from the first-relevant rank: MRR + hit@k."""
    out: Dict[str, float] = {"mrr": (1.0 / rank) if rank else 0.0}
    for k in ks:
        out[f"hit@{k}"] = 1.0 if (rank is not None and rank <= k) else 0.0
    return out


def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-query metric dicts."""
    if not per_query:
        return {}
    keys = list(per_query[0].keys())
    return {k: round(sum(m[k] for m in per_query) / len(per_query), 4) for k in keys}
