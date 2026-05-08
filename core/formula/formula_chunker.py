"""
Formula chunker for PresciSE.

Features:
- Span-aware formula extraction
- Deterministic normalization
- Context attachment (1-2 sentences around formula)
- Balanced quality filtering to suppress noisy formulas
- Enriched embedding text construction
"""

import re
from typing import Any, Dict, List, Union

from core.formula.extractor import FormulaExtractor
from core.formula.formula_schema import FormulaChunk
from core.formula.normalizer import normalize_formula, ascii_math_to_latex
from core.ingestion.document_schema import Document
from core.nlp.tokenizer import tokenize

_TRIVIAL_FORMULA_RE = re.compile(
    # Matches single-variable parameter assignments like: p*=0.8442, T*=0.72, p * = 0.8442, r.=2.5o
    # Allows optional whitespace and special chars (*, .) between letter and equals
    r"^\s*[A-Za-z\u03b1-\u03c9\u0391-\u03a9][\w\s\*\.]*\s*=\s*[-+]?[\d\.]+[A-Za-z0-9]*\s*(?:,\s*[-+]?[\d\.]+[A-Za-z0-9]*)?\s*$"
)
# Two-variable assignment with no structure: p = q, d = r
_TRIVIAL_TWO_VAR_RE = re.compile(
    r"^\s*[A-Za-z\u03b1-\u03c9][;,\s\*\.]*\s*=\s*[A-Za-z\u03b1-\u03c9]\s*$"
)
# ODE-style: dot-accented variable (ṗ, q̇, ˙p, ¨r) — always physics-meaningful.
# Includes U+02D9 (modifier letter dot above) and U+00A8 (diaeresis) used in PDFs.
_HAS_COMBINING_RE = re.compile(r"[\u02d9\u00a8\u0307\u0308\u0300-\u036f]")
# Unicode Greek letters in formula
_HAS_GREEK_RE = re.compile(r"[\u03b1-\u03c9\u0391-\u03a9]")
# Garbled subscript patterns from PDF encoding errors: G; = 0, r, = r. +
_TRIVIAL_GARBLED_RE = re.compile(
    r"^\s*[A-Za-z][;,\.]\s*=\s*\S{0,20}\s*$"
)
# Incomplete expression: ends with an open operator or bracket
_INCOMPLETE_FORMULA_RE = re.compile(r'[\(\[,/=\+\-]\s*$')
# Fragment: starts with a closing bracket (detached RHS)
_FRAGMENT_START_RE = re.compile(r'^\s*[\)\]]')
# Coordinate/point assignment: (x, y) = (0, 0) — purely positional, not physical
_COORDINATE_RE = re.compile(r'^\s*\(\s*\w+\s*,\s*\w+\s*\)\s*=\s*\(')
_SECTION_CAPTION_RE = re.compile(r"caption|figure|table", re.IGNORECASE)
_MATH_STRUCT_RE = re.compile(r"[=^/]|(?:\([^)]+\))")


def extract_formulas_from_doc(
    doc: Union[Document, Dict[str, Any]],
    min_confidence: float = 0.4,
    quality_mode: str = "balanced",
    extraction_source: str = "standard",
) -> List[Dict[str, Any]]:
    """
    Extract formula chunks from a normalized document.

    quality_mode:
    - strict: highest precision, lower recall
    - balanced: default
    - lenient: highest recall
    """
    extractor = FormulaExtractor(min_confidence=min_confidence)
    formula_chunks: List[Dict[str, Any]] = []
    seen_keys = set()
    formula_idx = 0

    if isinstance(doc, Document):
        doc_id = doc.document_id
        sections = doc.sections
    else:
        doc_id = doc["document_id"]
        sections = doc.get("sections", [])

    for section in sections:
        if hasattr(section, "section_type"):
            section_type = section.section_type
            content = (section.content or "").strip()
            pages = section.page_numbers
        else:
            section_type = section.get("section_type", "unknown")
            content = (section.get("content") or "").strip()
            pages = section.get("page_numbers", [])

        if not content:
            continue

        is_caption_source = bool(_SECTION_CAPTION_RE.search(str(section_type)))
        detected = extractor.extract_with_spans(content)

        for match in detected:
            raw_formula = match.formula_text
            normalized = normalize_formula(raw_formula)
            context = _build_context_from_sentences(
                section_text=content,
                formula_start=match.start,
                formula_end=match.end,
                max_sentences_each_side=2,
            )

            quality = _quality_score(normalized, context)
            is_trivial = _is_trivial_formula(normalized)
            keep = _should_keep_formula(
                normalized_formula=normalized,
                context_text=context,
                quality_score=quality,
                is_trivial=is_trivial,
                is_caption_source=is_caption_source,
                quality_mode=quality_mode,
            )
            if not keep:
                continue

            # Intra-document dedupe on normalized formula + page + short context hash.
            key = (
                normalized.lower(),
                pages[0] if pages else -1,
                _normalize_context_key(context),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            embedding_text = _build_embedding_text(normalized, context)
            chunk = FormulaChunk(
                chunk_id=f"{doc_id}_formula_{formula_idx}",
                chunk_type="formula",
                formula_text=raw_formula,
                context_text=context,
                normalized_formula=normalized,
                latex_formula=ascii_math_to_latex(normalized),
                source_section=section_type,
                doc_id=doc_id,
                page_number=pages[0] if pages else -1,
                extraction_source=extraction_source,
                quality_score=quality,
                is_trivial=is_trivial,
                is_caption_source=is_caption_source,
                embedding_text=embedding_text,
                tokens=tokenize(embedding_text),
            )
            formula_chunks.append(chunk.to_dict())
            formula_idx += 1

    return formula_chunks


def _split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    spans = []
    for m in re.finditer(r".+?(?:[.!?](?=\s+[A-Z])|$)", text, re.DOTALL):
        sentence = m.group(0).strip()
        if sentence:
            spans.append({"text": sentence, "start": m.start(), "end": m.end()})
    return spans


def _build_context_from_sentences(
    section_text: str,
    formula_start: int,
    formula_end: int,
    max_sentences_each_side: int = 2,
) -> str:
    sentences = _split_sentences_with_spans(section_text)
    if not sentences:
        return section_text[:500].strip()

    center_idx = None
    for i, s in enumerate(sentences):
        if s["start"] <= formula_start <= s["end"] or s["start"] <= formula_end <= s["end"]:
            center_idx = i
            break
    if center_idx is None:
        center_idx = min(range(len(sentences)), key=lambda i: abs(sentences[i]["start"] - formula_start))

    left = max(0, center_idx - max_sentences_each_side)
    right = min(len(sentences), center_idx + max_sentences_each_side + 1)
    context = " ".join(sentences[i]["text"] for i in range(left, right)).strip()
    return re.sub(r"\s+", " ", context)


def _build_embedding_text(normalized_formula: str, context_text: str) -> str:
    parts = ["The equation is defined as:", normalized_formula.strip()]
    if context_text:
        parts.append(context_text.strip())
    parts.append("Keywords: equation model law theorem potential relation")
    return "\n".join(parts).strip()


def _normalize_context_key(context_text: str) -> str:
    compact = re.sub(r"\s+", " ", context_text.lower()).strip()
    return compact[:200]


def _is_trivial_formula(formula: str) -> bool:
    # ODE shorthand (q̇=p, ṗ=-q-ζp): combining accent = always physics-meaningful
    if _HAS_COMBINING_RE.search(formula):
        return False
    if _TRIVIAL_FORMULA_RE.match(formula):
        return True
    # Catch any remaining short parameter assignments with optional spaces/special chars.
    # Examples: p*=0.8442, T * = 0.72, r. = 2.5o, x=1, n=5
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


def _should_keep_formula(
    normalized_formula: str,
    context_text: str,
    quality_score: float,
    is_trivial: bool,
    is_caption_source: bool,
    quality_mode: str,
) -> bool:
    if quality_mode == "lenient":
        return True

    has_structure = bool(_MATH_STRUCT_RE.search(normalized_formula))
    context_has_equation_cues = bool(
        re.search(r"(equation|potential|law|model|where|defined|hamiltonian|lennard|jones)", context_text, re.IGNORECASE)
    )

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
        if is_caption_source and not context_has_equation_cues:
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
    if is_caption_source and not context_has_equation_cues:
        return False
    return True

