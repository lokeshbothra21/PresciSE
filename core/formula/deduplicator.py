"""
Formula deduplication utilities for the Nougat background scanner.

_merge_nougat_formulas() merges Nougat-extracted formulas for a single document
into the full formula list, replacing any existing formulas on pages that Nougat
processed (Nougat visual output is treated as ground truth for those pages).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _merge_nougat_formulas(
    existing: List[Dict],
    nougat_found: List[Dict],
    doc_id: str,
) -> Tuple[List[Dict], int, int]:
    """
    Merge Nougat-found formulas for ``doc_id`` into the existing formula list.

    Strategy
    --------
    * **Page-level replacement**: For every page that Nougat processed, all
      previously extracted formulas for that page (belonging to ``doc_id``) are
      removed and replaced with Nougat's output.  Nougat read the visual
      representation, so it is the ground truth for those pages.
    * **Pages not touched by Nougat** keep their existing formulas unchanged.
    * After page-level replacement a cross-index subset check is applied: any
      *kept* existing formula whose normalised form is a proper substring of a
      Nougat formula is also dropped.

    Parameters
    ----------
    existing : list[dict]
        Full formula chunk list (all documents).
    nougat_found : list[dict]
        Formulas extracted by Nougat for *doc_id* (already deduplicated within
        the batch by ``_remove_subsets``).
    doc_id : str
        The document identifier being updated.

    Returns
    -------
    (merged_list, num_added, num_replaced)
    """
    from core.formula.nougat_scanner import _normalize_for_compare

    nougat_pages = {fc["page_number"] for fc in nougat_found}

    # Count how many existing formulas are displaced by Nougat
    replaced_count = sum(
        1
        for fc in existing
        if fc.get("doc_id") == doc_id
        and fc.get("page_number") in nougat_pages
    )

    # Remove displaced formulas
    kept_existing = [
        fc
        for fc in existing
        if not (
            fc.get("doc_id") == doc_id
            and fc.get("page_number") in nougat_pages
        )
    ]

    # Cross-index subset check: also remove kept formulas from *other* pages
    # that are a proper subset of any Nougat formula (avoids redundancy when the
    # regex pipeline captured a fragment that Nougat captured in full).
    if nougat_found:
        nougat_normed = {
            _normalize_for_compare(fc.get("latex_formula", ""))
            for fc in nougat_found
        }
        nougat_normed.discard("")   # don't match against empty strings

        def _is_subset_of_nougat(fc: Dict) -> bool:
            n = _normalize_for_compare(fc.get("latex_formula", ""))
            if not n:
                return False
            return any(n in nn and n != nn for nn in nougat_normed)

        kept_existing = [fc for fc in kept_existing if not _is_subset_of_nougat(fc)]

    merged = kept_existing + nougat_found
    added = len(nougat_found) - replaced_count
    return merged, max(added, 0), replaced_count
