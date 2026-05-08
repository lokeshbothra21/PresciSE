"""
PyMuPDF-based formula extractor for PresciSE.

Extracts formulas page-by-page directly from PDF files using PyMuPDF (fitz),
bypassing Docling's formula extraction path which suffers from math font
encoding issues in standard (non-neural) mode.

PyMuPDF handles math font encoding better than pdfminer (Docling's backend)
and runs in milliseconds with no ML models required.
"""

import re
from typing import Dict, List

import fitz  # pymupdf

from core.formula.extractor import FormulaExtractor
from core.formula.formula_schema import FormulaChunk
from core.formula.normalizer import normalize_formula, ascii_math_to_latex
from core.nlp.tokenizer import tokenize


def extract_formulas_from_pdf_pages(
    pdf_path: str,
    doc_id: str,
    min_confidence: float = 0.4,
    quality_mode: str = "balanced",
    context_window: int = 300,
) -> List[Dict]:
    """
    Extract formulas page-by-page using PyMuPDF.

    PyMuPDF handles math font encoding better than pdfminer (Docling's backend).
    Returns a list of formula chunk dicts compatible with FormulaChunk.to_dict().

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        doc_id: Document identifier (used for chunk_id and doc_id fields).
        min_confidence: Minimum extractor confidence to keep a formula.
        quality_mode: One of "strict", "balanced", "lenient".
        context_window: Characters before/after formula to capture as context.

    Returns:
        List of formula chunk dicts (same structure as FormulaChunk.to_dict()).
    """
    extractor = FormulaExtractor(min_confidence=min_confidence)
    seen_keys: set = set()
    results: List[Dict] = []
    formula_idx = 0

    fitz_doc = fitz.open(pdf_path)
    try:
        for page_num, page in enumerate(fitz_doc, start=1):
            page_text = _extract_page_text_column_aware(page)
            if not page_text or not page_text.strip():
                continue

            raw_matches = extractor.extract_with_spans(page_text)

            # Collect candidates for this page before committing to results
            page_candidates: List[Dict] = []
            for match in raw_matches:
                raw_formula = match.formula_text
                normalized = normalize_formula(raw_formula)
                context = _get_context(page_text, raw_formula, context_window)

                quality = _quality_score(normalized, context)
                is_trivial = _is_trivial_formula(normalized)

                if not _should_keep(normalized, context, quality, is_trivial, quality_mode):
                    continue

                # Deduplicate within document by (normalized formula, page).
                key = (normalized.lower(), page_num)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                page_candidates.append({
                    "raw_formula": raw_formula,
                    "normalized": normalized,
                    "context": context,
                    "quality": quality,
                    "is_trivial": is_trivial,
                })

            # Remove formulas whose normalized text is a substring of a longer formula
            # on the same page — they are incomplete fragments of the complete form.
            page_candidates = _remove_subsumed(page_candidates)

            for cand in page_candidates:
                embedding_text = _build_embedding_text(cand["normalized"], cand["context"])
                chunk = FormulaChunk(
                    chunk_id=f"{doc_id}_formula_{formula_idx}",
                    chunk_type="formula",
                    formula_text=cand["raw_formula"],
                    context_text=cand["context"],
                    normalized_formula=cand["normalized"],
                    latex_formula=ascii_math_to_latex(cand["normalized"]),
                    source_section="page_text",
                    doc_id=doc_id,
                    page_number=page_num,
                    extraction_source="pymupdf",
                    quality_score=cand["quality"],
                    is_trivial=cand["is_trivial"],
                    is_caption_source=False,
                    embedding_text=embedding_text,
                    tokens=tokenize(embedding_text),
                )
                results.append(chunk.to_dict())
                formula_idx += 1
    finally:
        fitz_doc.close()

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRIVIAL_FORMULA_RE = re.compile(
    r"^\s*[A-Za-z\u03b1-\u03c9\u0391-\u03a9][\w\s\*\.]*\s*=\s*[-+]?[\d\.]+[A-Za-z0-9]*\s*(?:,\s*[-+]?[\d\.]+[A-Za-z0-9]*)?\s*$"
)
# Two-variable assignment: p = q, d = r (single letter on each side, no structure)
_TRIVIAL_TWO_VAR_RE = re.compile(
    r"^\s*[A-Za-z\u03b1-\u03c9][;,\s\*\.]*\s*=\s*[A-Za-z\u03b1-\u03c9]\s*$"
)
# ODE-style: dot-accented variable (ṗ, q̇, ˙p, ¨r) — always physics-meaningful.
# Includes U+02D9 (modifier letter dot above) and U+00A8 (diaeresis) used in PDFs.
_HAS_COMBINING_RE = re.compile(r"[\u02d9\u00a8\u0307\u0308\u0300-\u036f]")
# Unicode Greek letters in formula
_HAS_GREEK_RE = re.compile(r"[\u03b1-\u03c9\u0391-\u03a9]")
# Garbled subscript: G; = 0, r, = r. + (punctuation adjacent to variable before =)
_TRIVIAL_GARBLED_RE = re.compile(
    r"^\s*[A-Za-z][;,\.]\s*=\s*\S{0,20}\s*$"
)
# Incomplete expression: ends with an open operator or bracket
_INCOMPLETE_FORMULA_RE = re.compile(r'[\(\[,/=\+\-]\s*$')
# Fragment: starts with a closing bracket (detached RHS)
_FRAGMENT_START_RE = re.compile(r'^\s*[\)\]]')
# Coordinate/point assignment: (x, y) = (0, 0) — purely positional, not physical
_COORDINATE_RE = re.compile(r'^\s*\(\s*\w+\s*,\s*\w+\s*\)\s*=\s*\(')
_MATH_STRUCT_RE = re.compile(r"[=^/]|(?:\([^)]+\))")


def _extract_page_text_column_aware(page) -> str:
    """
    Extract text from a PDF page preserving column reading order.

    For two-column PDFs (e.g. journal articles), sorting raw page.get_text("text")
    interleaves equations from both columns, fragmenting LHS and RHS of the same
    equation. This function detects the layout and processes columns independently.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    if not blocks:
        return ""

    # Filter to text blocks only (block_type == 0)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""

    page_width = page.rect.width
    mid_x = page_width / 2

    # Detect two-column layout: significant content both left and right of midpoint
    left_blocks  = [b for b in text_blocks if b[2] <= mid_x * 1.1]   # x1 <= 1.1 * mid
    right_blocks = [b for b in text_blocks if b[0] >= mid_x * 0.9]   # x0 >= 0.9 * mid
    is_two_column = (
        len(left_blocks) >= 2
        and len(right_blocks) >= 2
        and len(left_blocks) + len(right_blocks) > len(text_blocks)  # overlap means split
    )

    if is_two_column:
        # Sort each column by vertical position (y0), then concatenate left → right
        strict_left  = [b for b in text_blocks if b[2] <= mid_x]
        strict_right = [b for b in text_blocks if b[0] >  mid_x]
        header_footer = [b for b in text_blocks if b not in strict_left and b not in strict_right]
        strict_left.sort(key=lambda b: b[1])
        strict_right.sort(key=lambda b: b[1])
        header_footer.sort(key=lambda b: b[1])
        ordered = header_footer + strict_left + strict_right
    else:
        # Single column: sort by vertical position
        text_blocks.sort(key=lambda b: b[1])
        ordered = text_blocks

    return "\n".join(b[4] for b in ordered)


def _get_context(page_text: str, formula: str, window: int = 300) -> str:
    """
    Extract up to `window` characters before and after the formula in page_text.
    Falls back to beginning of page if formula is not found verbatim.
    """
    idx = page_text.find(formula)
    if idx == -1:
        # Formula was reconstructed by extractor; use page start as context.
        return re.sub(r"\s+", " ", page_text[:window * 2]).strip()

    start = max(0, idx - window)
    end = min(len(page_text), idx + len(formula) + window)
    context = page_text[start:end]
    return re.sub(r"\s+", " ", context).strip()


def _is_trivial_formula(formula: str) -> bool:
    # ODE shorthand (q̇=p, ṗ=-q-ζp): combining accent = always physics-meaningful
    if _HAS_COMBINING_RE.search(formula):
        return False
    if _TRIVIAL_FORMULA_RE.match(formula):
        return True
    # Simple single-variable numeric assignment: x = 1.23
    if re.match(r"^\s*[A-Za-z][\s\*\.]*\s*=\s*[-+]?[\d\.]+[A-Za-z0-9]*\s*$", formula):
        return True
    # Two-variable assignment with no structure: p = q, d = r
    if _TRIVIAL_TWO_VAR_RE.match(formula):
        # q = p is trivial ONLY if no Greek letter is present
        if _HAS_GREEK_RE.search(formula):
            return False
        return True
    # Garbled subscript patterns from PDF encoding errors: G; = 0, r, = r. +
    if _TRIVIAL_GARBLED_RE.match(formula):
        return True
    # Incomplete expression: ends with an open operator/bracket — truncated formula
    stripped = formula.strip()
    if _INCOMPLETE_FORMULA_RE.search(stripped):
        return True
    # Fragment: starts with a closing bracket — detached RHS
    if _FRAGMENT_START_RE.match(stripped):
        return True
    # Coordinate/point assignment like (x, y) = (0, 0) — positional, not physics
    if _COORDINATE_RE.match(stripped):
        return True
    return False


def _quality_score(formula: str, context: str) -> float:
    score = 0.0
    if "=" in formula:
        score += 0.25
    elif re.search(r"[∝≡∼≈]", formula):
        score += 0.20
    if "^" in formula:
        score += 0.15
    if "/" in formula:
        score += 0.10
    if "(" in formula and ")" in formula:
        score += 0.15
    if re.search(r"[∑∫∂]|sum|integral|derivative", formula, re.IGNORECASE):
        score += 0.10
    if re.search(r"(equation|potential|law|model|hamiltonian|lennard|jones)", context, re.IGNORECASE):
        score += 0.10
    symbolic = len(re.findall(r"[0-9=+\-*/^()_]", formula))
    density = symbolic / max(1, len(formula))
    score += min(0.15, density * 0.4)
    return min(1.0, score)


def _should_keep(
    normalized_formula: str,
    context_text: str,
    quality_score: float,
    is_trivial: bool,
    quality_mode: str,
) -> bool:
    if quality_mode == "lenient":
        return True

    has_structure = bool(_MATH_STRUCT_RE.search(normalized_formula))

    if quality_mode == "strict":
        if not has_structure:
            return False
        if quality_score < 0.35:
            return False
        if is_trivial:
            if re.search(
                r"(oscillator|thermostat|equation of motion|nos[eé]|hoover|langevin)",
                context_text,
                re.IGNORECASE,
            ):
                pass  # keep it despite trivial flag
            else:
                return False
        return True

    # balanced
    if not has_structure:
        return False
    if quality_score < 0.2:
        return False
    if is_trivial:
        # Physics-context rescue: short ODEs near oscillator/thermostat context
        if re.search(
            r"(oscillator|thermostat|equation of motion|nos[eé]|hoover|langevin)",
            context_text,
            re.IGNORECASE,
        ):
            pass  # keep it despite trivial flag
        else:
            return False
    return True


def _build_embedding_text(normalized_formula: str, context_text: str) -> str:
    parts = ["The equation is defined as:", normalized_formula.strip()]
    if context_text:
        parts.append(context_text.strip())
    parts.append("Keywords: equation model law theorem potential relation")
    return "\n".join(parts).strip()


def _context_key(context_text: str) -> str:
    compact = re.sub(r"\s+", " ", context_text.lower()).strip()
    return compact[:200]


def _remove_subsumed(candidates: List[Dict]) -> List[Dict]:
    """
    Remove formula candidates whose normalized text is a substring of any
    other candidate on the same page. The longer formula is the complete form;
    the shorter one is a fragment and should be discarded.
    """
    norms = [c["normalized"].lower().strip() for c in candidates]
    keep = []
    for i, cand in enumerate(candidates):
        ni = norms[i]
        subsumed = any(
            ni != nj and ni in nj
            for j, nj in enumerate(norms)
            if j != i
        )
        if not subsumed:
            keep.append(cand)
    return keep
