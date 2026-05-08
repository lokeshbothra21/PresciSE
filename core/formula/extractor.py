"""
Regex-based formula extractor for scientific text.

Detects equation-like substrings in inline paragraphs and multiline blocks.
No LLM/OCR required.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from core.formula.normalizer import normalize_formula

# Greek letters (Unicode + mojibake variants often seen in PDF extraction)
GREEK_LETTERS = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩÎÏ")

# Math operators and symbols
MATH_SYMBOLS = set("=±×÷∇∫∑∏√∂∞≈≠≤≥<>∝≡∼~")

# Known math function names (do not treat as natural language words)
MATH_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "ln",
    "max",
    "min",
    "sum",
    "lim",
    "sqrt",
    "det",
    "div",
    "grad",
    "curl",
}

# Derivative patterns: dX/dt, ∂X/∂t — multi-char tokens like dp_i/dt, d(ps)/dt
DERIVATIVE_PATTERN = re.compile(
    r"[d∂]\(?[A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω_0-9]{0,3}\)?\s*/\s*[d∂][A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω_0-9]{0,3}"
)

# ODE pattern: complete derivative equation dX/dt = RHS
ODE_PATTERN = re.compile(
    r"[d∂]\(?[A-Za-z_][A-Za-z_0-9]{0,4}\)?\s*/\s*[d∂][A-Za-z_][A-Za-z_0-9]{0,3}\s*=\s*[^\n.;]{5,200}"
)

# Summation/integral/product patterns
SUMMATION_PATTERN = re.compile(r"[∑∫∏]\s*[A-Za-z0-9α-ωΑ-Ω_\s\(\)\[\]\^*/+\-=]+")

# Inline or multiline equation-like fragments with structure cues.
# Includes U+02D9 (˙, modifier dot above) and U+00A8 (¨, diaeresis) so that
# time-derivative notation like "˙p" and "¨r" are captured as valid LHS tokens.
EQUATION_FRAGMENT_PATTERN = re.compile(
    r"([\u02d9\u00a8\u0307\u0308A-Za-z0-9\)\]\}α-ωΑ-Ω]\s*[\u02d9\u00a8\u0307\u0308A-Za-z0-9α-ωΑ-Ω\(\)\[\]\{\}\s\+\-\*/\^_]*=\s*[A-Za-z0-9α-ωΑ-Ω\(\)\[\]\{\}\s\+\-\*/\^_\.]{2,600})",
    re.MULTILINE,
)

# Words that commonly appear *inside* formula descriptions and should not halt extraction.
# e.g. "V(r) = 4ε[...] where σ is the diameter"
FORMULA_CONNECTORS = {
    "where", "with", "and", "for", "of", "the", "in", "at", "by",
    "is", "are", "as", "an", "a", "or", "to", "be",
}


@dataclass(frozen=True)
class FormulaMatch:
    formula_text: str
    confidence: float
    start: int
    end: int


class FormulaExtractor:
    """
    Extract formula-like text from scientific content.
    """

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def extract(self, text: str) -> List[Tuple[str, float]]:
        """Backward-compatible API: returns (formula_text, confidence)."""
        matches = self.extract_with_spans(text)
        return [(m.formula_text, m.confidence) for m in matches]

    def extract_with_spans(self, text: str) -> List[FormulaMatch]:
        """
        Extract formulas with confidence and text spans.
        """
        if not text:
            return []

        normalized_text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
        candidates = []
        seen = set()

        # Strategy 1: equals-sign anchored parsing for inline formulas.
        for match in re.finditer(r"=", normalized_text):
            pos = match.start()

            # Skip comparison operators.
            if pos > 0 and normalized_text[pos - 1] in "!<>=":
                continue
            if pos < len(normalized_text) - 1 and normalized_text[pos + 1] == "=":
                continue

            lhs_window = normalized_text[max(0, pos - 100) : pos]
            rhs_window = normalized_text[pos + 1 : min(len(normalized_text), pos + 500)]
            lhs = self._extract_lhs(lhs_window)
            rhs = self._extract_rhs(rhs_window)

            if lhs and rhs:
                formula = normalize_formula(f"{lhs} = {rhs}")
                if self._is_valid_formula(formula):
                    start = max(0, pos - len(lhs))
                    end = min(len(normalized_text), pos + 1 + len(rhs))
                    key = (formula, start, end)
                    if key not in seen:
                        seen.add(key)
                        candidates.append((formula, start, end))

        # Strategy 2: equation fragments for inline/multiline extraction.
        for m in EQUATION_FRAGMENT_PATTERN.finditer(normalized_text):
            formula = self._compact_equation_candidate(m.group(1))
            if self._is_valid_formula(formula):
                key = (formula, m.start(), m.end())
                if key not in seen:
                    seen.add(key)
                    candidates.append((formula, m.start(), m.end()))

        # Strategy 3: derivative expressions.
        for m in DERIVATIVE_PATTERN.finditer(normalized_text):
            formula = normalize_formula(m.group().strip())
            if self._is_valid_formula(formula):
                key = (formula, m.start(), m.end())
                if key not in seen:
                    seen.add(key)
                    candidates.append((formula, m.start(), m.end()))

        # Strategy 4: summation/integral/product expressions.
        for m in SUMMATION_PATTERN.finditer(normalized_text):
            formula = normalize_formula(m.group().strip())
            if len(formula) > 3 and self._is_valid_formula(formula):
                key = (formula, m.start(), m.end())
                if key not in seen:
                    seen.add(key)
                    candidates.append((formula, m.start(), m.end()))

        # Strategy 5: complete ODE equations — dX/dt = RHS.
        for m in ODE_PATTERN.finditer(normalized_text):
            formula = normalize_formula(m.group().strip())
            if self._is_valid_formula(formula):
                key = (formula, m.start(), m.end())
                if key not in seen:
                    seen.add(key)
                    candidates.append((formula, m.start(), m.end()))

        # Strategy 6: proportionality / equivalence operators (∝, ≡, ∼, ≈).
        for m in re.finditer(r"[∝≡∼≈]", normalized_text):
            pos = m.start()
            lhs = self._extract_lhs(normalized_text[max(0, pos - 100): pos])
            rhs = self._extract_rhs(normalized_text[pos + 1: pos + 500])
            if lhs and rhs:
                formula = normalize_formula(f"{lhs} {m.group()} {rhs}")
                if self._is_valid_formula(formula):
                    start = max(0, pos - len(lhs))
                    end = min(len(normalized_text), pos + 1 + len(rhs))
                    key = (formula, start, end)
                    if key not in seen:
                        seen.add(key)
                        candidates.append((formula, start, end))

        results: List[FormulaMatch] = []
        for formula, start, end in candidates:
            confidence = self._score_formula(formula)
            if confidence >= self.min_confidence:
                results.append(
                    FormulaMatch(
                        formula_text=formula,
                        confidence=confidence,
                        start=max(0, start),
                        end=max(start, end),
                    )
                )

        # Deduplicate by normalized formula text and keep highest-confidence span.
        dedup = {}
        for r in results:
            prev = dedup.get(r.formula_text)
            if prev is None or r.confidence > prev.confidence:
                dedup[r.formula_text] = r
        results = list(dedup.values())

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def _extract_lhs(self, window: str) -> str:
        window = window.rstrip()
        if not window:
            return ""

        tokens = window.split()
        formula_tokens = []
        for token in reversed(tokens):
            if self._is_formula_token(token):
                formula_tokens.insert(0, token)
            else:
                break
        return " ".join(formula_tokens).strip()

    def _extract_rhs(self, window: str) -> str:
        window = window.lstrip()
        if not window:
            return ""

        for sep in [". ", ".\n", ".\r", ";\n", "; "]:
            idx = window.find(sep)
            if idx >= 0:
                window = window[:idx]

        tokens = window.split()
        formula_tokens = []
        consecutive_non_formula = 0

        for token in tokens:
            if self._is_formula_token(token):
                formula_tokens.append(token)
                consecutive_non_formula = 0
            elif token.lower().rstrip(".,;:") in FORMULA_CONNECTORS:
                # Connector word (e.g. "where", "with") — keep it but count it
                formula_tokens.append(token)
                consecutive_non_formula += 1
                if consecutive_non_formula >= 3:
                    # Three connectors in a row means we've left the formula
                    formula_tokens = formula_tokens[:-3]
                    break
            else:
                # True prose word — stop here
                break

        # Trim any trailing connector words that were added speculatively
        while formula_tokens and formula_tokens[-1].lower().rstrip(".,;:") in FORMULA_CONNECTORS:
            formula_tokens.pop()

        return " ".join(formula_tokens).strip()

    def _compact_equation_candidate(self, fragment: str) -> str:
        """
        Reduce a broad equation fragment to its core LHS=RHS expression.
        """
        if "=" not in fragment:
            return normalize_formula(fragment)
        pos = fragment.find("=")
        lhs = self._extract_lhs(fragment[max(0, pos - 100) : pos])
        rhs = self._extract_rhs(fragment[pos + 1 : min(len(fragment), pos + 500)])
        if lhs and rhs:
            return normalize_formula(f"{lhs} = {rhs}")
        return normalize_formula(fragment)

    def _is_formula_token(self, token: str) -> bool:
        clean = token.strip("(),[]{}:;")
        if not clean:
            return False

        if clean in {"+", "-", "*", "/", "^", "=", "±", "×", "÷"}:
            return True

        # Accent characters (˙, ¨) attached to a letter form ODE variables.
        if len(clean) >= 1 and clean[0] in "\u02d9\u00a8\u0307\u0308":
            return True

        if len(clean) == 1:
            return clean.isalpha() or clean.isdigit() or clean in GREEK_LETTERS or clean in MATH_SYMBOLS

        if clean.lower() in MATH_FUNCTIONS:
            return True

        if any(c.isdigit() for c in clean):
            return True

        if any(c in "^_*/+∂∇∫∑" for c in clean):
            return True

        if any(c in GREEK_LETTERS for c in clean):
            return True

        if len(clean) <= 3 and clean[0].isupper():
            return True

        if len(clean) == 2 and clean.isalpha() and clean.islower():
            # Allow derivative shorthand: dp, dq, dr, ds, dt, dv, dx, dy, dz
            if clean[0] == 'd':
                return True
            return False

        if len(clean) > 2 and clean.isalpha() and clean[0].islower():
            return False

        return True

    def _is_valid_formula(self, formula: str) -> bool:
        if len(formula) < 3 or len(formula) > 600:
            return False

        has_operator = bool(re.search(r"[=+\-*/^∂∇∫∑()∝≡∼≈]", formula))
        if not has_operator:
            return False

        # Reject natural-language-heavy fragments.
        words = re.findall(r"[A-Za-z]{3,}", formula)
        non_math_words = [w for w in words if w.lower() not in MATH_FUNCTIONS]
        symbol_count = len(re.findall(r"[0-9=+\-*/^()_]", formula))
        density = symbol_count / max(1, len(formula))
        if len(non_math_words) > 6 and density < 0.22:
            return False
        if "=" in formula and density < 0.08 and len(non_math_words) > 3:
            return False

        # Patterns anchored to start of formula
        start_patterns = [r"^(i\.e\.|e\.g\.|et al|Fig\.|Table|Ref)"]
        for pattern in start_patterns:
            if re.match(pattern, formula, re.IGNORECASE):
                return False
        # Patterns that can appear anywhere — figure references and non-math text
        anywhere_patterns = [
            r"\bin\s*[Ff]ig",      # "in Fig.", "inFig." (no space)
            r"\b[Ff]igs?\.\s*\d",  # "Figs. 1", "Fig.1"
            r"https?://",
            r"@",
        ]
        for pattern in anywhere_patterns:
            if re.search(pattern, formula, re.IGNORECASE):
                return False

        return True

    def _score_formula(self, formula: str) -> float:
        score = 0.0

        if "=" in formula:
            score += 0.35
        elif re.search(r"[∝≡∼≈]", formula):
            score += 0.30

        if any(c in GREEK_LETTERS for c in formula):
            score += 0.15

        extra_math = set(formula) & (MATH_SYMBOLS - {"="})
        if extra_math:
            score += 0.10

        if "^" in formula or re.search(r"[²³⁴⁵⁶⁷⁸⁹⁰]", formula):
            score += 0.10

        if "_" in formula:
            score += 0.05

        if "/" in formula:
            score += 0.05

        alpha_count = sum(1 for c in formula if c.isalpha())
        total = len(formula.replace(" ", ""))
        if total > 0:
            symbolic_ratio = 1 - (alpha_count / total)
            score += symbolic_ratio * 0.20

        return min(score, 1.0)

    @staticmethod
    def normalize(formula: str) -> str:
        """Backward-compatible normalizer wrapper."""
        return normalize_formula(formula)
