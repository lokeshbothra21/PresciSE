"""Fast rule-based search router for PresciSE.

Replaces Qwen2.5-1.5B with a sub-millisecond regex + NLP rule engine
that keeps the same ``route(query) -> dict`` interface.

Layered signal detection:
  1. Corpus-specific physics named-entity regex
  2. spaCy NER (PERSON/ORG) for author names Qwen missed
  3. Formula-cue keywords / math symbols
  4. classify_query() NLP fallback

Weight table:
  named entity + formula cue → exact_formula   (BM25 0.75, FAISS 0.25)
  named entity only          → exact_match      (BM25 0.65, FAISS 0.35)
  formula cue only           → formula_search   (BM25 0.70, FAISS 0.30)
  NLP → definition           → definition_lookup(BM25 0.50, FAISS 0.50)
  NLP → methodology          → methodology      (BM25 0.35, FAISS 0.65)
  NLP → comparison           → comparison       (BM25 0.30, FAISS 0.70)
  default                    → exploratory      (BM25 0.45, FAISS 0.55)
"""

import re
from typing import Dict, Optional

from loguru import logger

from core.nlp.query_classifier import classify_query
from core.nlp.ner import extract_entities


class SearchRouter:
    """
    Rule-based router for dynamic hybrid search weight determination.

    Uses a layered signal approach to classify queries and assign
    BM25/FAISS weights without loading any ML model.  Runs in < 1 ms.

    Args:
        default_weights: Optional override for the fallback weight dict
                         (useful in unit tests).
    """

    # ------------------------------------------------------------------
    # Corpus-specific physics / method terms that should boost BM25
    # ------------------------------------------------------------------
    _PHYSICS_ENTITY_RE = re.compile(
        r"\b(?:"
        r"lennard[- ]?jones|"
        r"nos[eé][- ]?hoover|"
        r"ewald|"
        r"verlet|"
        r"lammps|"
        r"hamiltonian|"
        r"partition[ \-]?function|"
        r"nvt|nve|npt|"
        r"langevin|"
        r"amber|charmm|gromacs|namd|openmm|"
        r"ergodic(?:ity)?|"
        r"equipartition|"
        r"boltzmann|"
        r"maxwell[- ]boltzmann|"
        r"born[- ]oppenheimer|"
        r"rdf|"
        r"pair[ \-]?distribution|"
        r"radial[ \-]?distribution|"
        r"potential[ \-]?energy|"
        r"kinetic[ \-]?energy|"
        r"thermostat|barostat|"
        r"cutoff|"
        r"periodic[ \-]?boundary|pbc|"
        r"force[ \-]?field|"
        r"trajectory|"
        r"microcanonical|canonical|grand[ \-]?canonical"
        r")\b",
        re.IGNORECASE | re.UNICODE,
    )

    # ------------------------------------------------------------------
    # Formula / equation cues (math symbols + domain keywords)
    # ------------------------------------------------------------------
    _FORMULA_CUE_RE = re.compile(
        r"(?:"
        r"equation|formula|expression|"
        r"potential(?!\s+energy)|"   # "potential energy" caught by entity regex
        r"hamiltonian|lagrangian|"
        r"partition[ \-]?function|"
        r"deriv(?:ative)?|integral|"
        r"function\s+of|"
        r"[∂∫∑∇∆Σ×·±≈≡∝∼]"
        r")",
        re.IGNORECASE | re.UNICODE,
    )

    # ------------------------------------------------------------------
    # Intent → (intent_label, bm25_weight, faiss_weight)
    # ------------------------------------------------------------------
    _WEIGHTS: Dict[str, tuple] = {
        "exact_formula":     ("exact_formula",     0.75, 0.25),
        "exact_match":       ("exact_match",       0.65, 0.35),
        "formula_search":    ("formula_search",    0.70, 0.30),
        "definition_lookup": ("definition_lookup", 0.50, 0.50),
        "methodology":       ("methodology",       0.35, 0.65),
        "comparison":        ("comparison",        0.30, 0.70),
        "exploratory":       ("exploratory",       0.45, 0.55),
    }

    def __init__(self, default_weights: Optional[Dict] = None) -> None:
        self._default_weights = default_weights or {
            "bm25_weight": 0.45,
            "faiss_weight": 0.55,
            "intent": "exploratory",
            "reason": "Default exploratory weights",
        }

    # ------------------------------------------------------------------
    # Public API — same signature as the old Qwen-based router
    # ------------------------------------------------------------------

    def route(self, query: str) -> Dict[str, object]:
        """
        Analyze *query* and return optimal BM25 / FAISS weights.

        Returns:
            {
                "bm25_weight": float,
                "faiss_weight": float,
                "intent": str,
                "reason": str,
            }
        """
        try:
            return self._route(query)
        except Exception as exc:
            logger.warning(f"SearchRouter fallback ({exc}): {query!r}")
            return {**self._default_weights, "reason": f"Router error: {exc}"}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _route(self, query: str) -> Dict[str, object]:
        has_entity = bool(self._PHYSICS_ENTITY_RE.search(query))

        # spaCy NER catches author names (Nosé, Hoover) that the regex misses
        if not has_entity:
            try:
                ents = extract_entities(query)
                has_entity = any(e["label"] in ("PERSON", "ORG") for e in ents)
            except Exception:
                pass  # spaCy model not loaded — continue without it

        has_formula_cue = bool(self._FORMULA_CUE_RE.search(query))

        # --- Priority cascade ---
        if has_entity and has_formula_cue:
            key = "exact_formula"
            reason = "Physics entity + formula cue → exact formula lookup"
        elif has_entity:
            key = "exact_match"
            reason = "Physics named entity → BM25-weighted term matching"
        elif has_formula_cue:
            key = "formula_search"
            reason = "Formula/equation cue → BM25-weighted formula search"
        else:
            nlp_type = classify_query(query)
            key = {
                "definition":  "definition_lookup",
                "methodology": "methodology",
                "comparison":  "comparison",
                "exploratory": "exploratory",
            }.get(nlp_type, "exploratory")
            reason = f"NLP classification: {nlp_type}"

        intent, bm25, faiss = self._WEIGHTS[key]
        result = {
            "bm25_weight": bm25,
            "faiss_weight": faiss,
            "intent": intent,
            "reason": reason,
        }
        logger.debug(f"Router: intent={intent} bm25={bm25} faiss={faiss} | {query!r}")
        return result
