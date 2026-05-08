"""
Formula diagnostics for indexing quality visibility.
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def build_formula_diagnostics(
    formula_chunks: List[Dict[str, Any]],
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build aggregate diagnostics for formula chunk quality.
    """
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for fc in formula_chunks:
        by_doc.setdefault(fc.get("doc_id", "unknown"), []).append(fc)

    docs = []
    for doc_id, rows in sorted(by_doc.items(), key=lambda kv: kv[0]):
        docs.append(_doc_stats(doc_id, rows))

    repeated = Counter((fc.get("normalized_formula") or fc.get("formula_text") or "").strip() for fc in formula_chunks)
    top_repeated = [{"formula": f, "count": c} for f, c in repeated.most_common(25) if f]

    out = {
        "total_formula_chunks": len(formula_chunks),
        "docs": docs,
        "global": {
            "empty_embedding_text": sum(1 for x in formula_chunks if not (x.get("embedding_text") or "").strip()),
            "caption_source": sum(1 for x in formula_chunks if x.get("is_caption_source")),
            "trivial": sum(1 for x in formula_chunks if x.get("is_trivial")),
            "strong_structure": sum(1 for x in formula_chunks if _has_strong_structure(x)),
        },
        "top_repeated": top_repeated,
        "stage_metrics": stage_metrics or {},
    }
    return out


def save_formula_diagnostics(diag: Dict[str, Any], index_dir: str) -> str:
    path = os.path.join(index_dir, "formula_diagnostics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    return path


def _doc_stats(doc_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    formulas = [(r.get("normalized_formula") or r.get("formula_text") or "").strip() for r in rows]
    repeated = Counter(formulas)
    cues = ["lennard", "jones", "sigma", "epsilon", "hamiltonian", "partition", "potential"]
    cue_hits = {cue: sum(1 for r in rows if cue in ((r.get("context_text") or "").lower() + " " + (r.get("normalized_formula") or "").lower())) for cue in cues}
    return {
        "doc_id": doc_id,
        "count": len(rows),
        "caption_source": sum(1 for r in rows if r.get("is_caption_source")),
        "trivial": sum(1 for r in rows if r.get("is_trivial")),
        "strong_structure": sum(1 for r in rows if _has_strong_structure(r)),
        "avg_quality_score": round(sum(float(r.get("quality_score", 0.0)) for r in rows) / max(1, len(rows)), 4),
        "top_repeated": [{"formula": f, "count": c} for f, c in repeated.most_common(10) if f],
        "cue_hits": cue_hits,
    }


def _has_strong_structure(row: Dict[str, Any]) -> bool:
    f = (row.get("normalized_formula") or row.get("formula_text") or "")
    return bool(re.search(r"[=^/]|(?:\([^)]+\))|[∑∫∂]", f))
