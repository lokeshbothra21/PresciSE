"""
FormulaChunk schema for structured formula storage.

Each formula is stored as a knowledge unit with:
- The raw formula text
- Surrounding context for semantic embedding
- Normalized representation for matching stability
- Source metadata for citation
"""

from typing import List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class FormulaChunk:
    """
    A single formula extracted from a scientific document.
    
    Designed to be embedded using SPECTER alongside text chunks
    but stored in a separate index for independent retrieval.
    
    Embedding strategy:
        SPECTER embeds `context_text + " " + formula_text` together,
        NOT the formula alone. This allows conceptual queries like
        "theory of relativity" to match "E = mc^2" via context.
    """
    
    chunk_id: str                          # e.g. "hoover1985_formula_0"
    chunk_type: str = "formula"            # always "formula"
    formula_text: str = ""                 # raw: "E = mc^2"
    context_text: str = ""                 # surrounding paragraph
    normalized_formula: str = ""           # cleaned ASCII-math: "E = mc^2"
    latex_formula: str = ""               # renderable LaTeX: "E = mc^{2}"
    source_section: str = ""               # section type from Document
    doc_id: str = ""                       # parent document ID
    page_number: int = -1                  # page where formula appears
    extraction_source: str = "standard"    # standard|formula_enrichment|vlm_fallback
    quality_score: float = 0.0             # heuristic quality score [0,1]
    is_trivial: bool = False               # scalar/simple assignment noise marker
    is_caption_source: bool = False        # extracted from caption-like section
    embedding_text: str = ""               # enriched text used for embedding
    embedding: Optional[List[float]] = None  # SPECTER vector (768-dim)
    tokens: List[str] = field(default_factory=list)  # for BM25 compatibility
    
    def to_dict(self) -> dict:
        """Convert to dict for pickle serialization and index storage."""
        d = asdict(self)
        # Build combined text for retrieval display
        d["text"] = self._build_display_text()
        return d
    
    def _build_display_text(self) -> str:
        """
        Build text representation for display in evidence blocks.

        Emits LaTeX wrapped in [FORMULA]$...$[/FORMULA] when available so the
        LLM can reproduce it verbatim and the frontend can render it via KaTeX.
        Falls back to the raw formula text when no LaTeX is stored.

        Format:
            [FORMULA]$E = mc^{2}$[/FORMULA]
            Context: Einstein showed that energy and mass...
        """
        parts = []
        if self.latex_formula:
            parts.append(f"[FORMULA]${self.latex_formula}$[/FORMULA]")
        elif self.formula_text:
            parts.append(f"[FORMULA]{self.formula_text}[/FORMULA]")
        if self.context_text:
            parts.append(f"Context: {self.context_text}")
        return "\n".join(parts)
    
    def get_embedding_text(self) -> str:
        """
        Build the text that SPECTER will embed.
        
        Embeds normalized formula + explanatory context so conceptual
        queries can match both equation structure and semantics.
        """
        if self.embedding_text:
            return self.embedding_text

        equation = self.normalized_formula or self.formula_text
        parts = []
        parts.append("The equation is defined as:")
        if equation:
            parts.append(equation)
        if self.context_text:
            parts.append(self.context_text)
        parts.append("Keywords: equation model law theorem potential relation")
        return "\n".join(parts).strip()
