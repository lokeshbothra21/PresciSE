"""
Formula extraction and indexing module for PresciSE.

Provides:
    - FormulaChunk schema for structured formula storage
    - Regex-based formula extractor from scientific text
    - Formula chunker for building formula-specific indexes
"""

from core.formula.formula_schema import FormulaChunk
from core.formula.extractor import FormulaExtractor
from core.formula.formula_chunker import extract_formulas_from_doc
from core.formula.diagnostics import build_formula_diagnostics, save_formula_diagnostics
from core.formula.normalizer import normalize_formula

__all__ = [
    "FormulaChunk",
    "FormulaExtractor",
    "extract_formulas_from_doc",
    "build_formula_diagnostics",
    "save_formula_diagnostics",
    "normalize_formula",
]
