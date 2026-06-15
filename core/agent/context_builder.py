"""
Corpus context description builder for the DD+Expert agent.

Produces a short human-readable summary of which documents are indexed,
used by the DD agent during planning to decide whether a query is in-scope.
"""

import re
from typing import Any, Dict, List


def build_context_description(
    chunks: List[Dict[str, Any]],
    allowed_owners: "set | frozenset | None" = None,
) -> str:
    """
    Derive a concise description of the indexed corpus for the DD agent.

    Args:
        chunks: Flat list of chunk dicts from the retriever (text + formula chunks).
        allowed_owners: if given, only chunks owned by one of these ids (a chunk
            with no owner counts as "__shared__") are described. Pass the query's
            scope ({user_id, "__shared__"}) so the description lists only the
            requesting user's documents — never other tenants' filenames.

    Returns:
        Multi-line string listing indexed document names.
    """
    if allowed_owners is not None:
        chunks = [
            c for c in chunks
            if (c.get("owner_user_id") or "__shared__") in allowed_owners
        ]
    doc_ids = sorted({c.get("doc_id", "") for c in chunks if c.get("doc_id")})
    # Strip trailing hex fingerprint added by IndexManager (e.g. "_a3f9b2")
    doc_names = [re.sub(r"_[a-f0-9]{6,}$", "", d).replace("_", " ") for d in doc_ids]
    if not doc_names:
        return "No documents are indexed."
    return "Indexed documents:\n" + "\n".join(f"  - {n}" for n in doc_names)
