"""
Corpus context description builder for the DD+Expert agent.

Produces a short human-readable summary of which documents are indexed,
used by the DD agent during planning to decide whether a query is in-scope.
"""

import re
from typing import Any, Dict, List


def build_context_description(chunks: List[Dict[str, Any]]) -> str:
    """
    Derive a concise description of the indexed corpus for the DD agent.

    Args:
        chunks: Flat list of chunk dicts from the retriever (text + formula chunks).

    Returns:
        Multi-line string listing indexed document names.
    """
    doc_ids = sorted({c.get("doc_id", "") for c in chunks if c.get("doc_id")})
    # Strip trailing hex fingerprint added by IndexManager (e.g. "_a3f9b2")
    doc_names = [re.sub(r"_[a-f0-9]{6,}$", "", d).replace("_", " ") for d in doc_ids]
    if not doc_names:
        return "No documents are indexed."
    return "Indexed documents:\n" + "\n".join(f"  - {n}" for n in doc_names)
