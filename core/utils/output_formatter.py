"""
Standard output formatter for PresciSE query results.

Provides consistent, readable formatting for retrieved evidence and answers.
"""

import re
import sys
from typing import List, Dict, Any

# Mapping from ASCII names (used by normalizer) back to Unicode symbols for display.
_GREEK_ASCII_TO_UNICODE = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}

# LaTeX command → Unicode for terminal rendering (longest first avoids partial hits)
_LATEX_GREEK_TO_UNICODE = {
    "\\varepsilon": "ε", "\\vartheta": "θ", "\\varphi": "φ", "\\varpi": "π",
    "\\varrho": "ρ", "\\varsigma": "σ",
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ",
    "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ",
    "\\pi": "π", "\\rho": "ρ", "\\sigma": "σ", "\\tau": "τ",
    "\\phi": "φ", "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
    "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Phi": "Φ",
    "\\Psi": "Ψ", "\\Omega": "Ω",
}

_LATEX_OPS_TO_UNICODE = {
    "\\propto": "∝", "\\equiv": "≡", "\\approx": "≈", "\\sim": "~",
    "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\nabla": "∇",
    "\\partial": "∂", "\\sum": "Σ", "\\prod": "Π", "\\int": "∫",
    "\\sqrt": "√", "\\pm": "±", "\\times": "×", "\\div": "÷",
    "\\infty": "∞", "\\cdot": "·",
}

# Ensure stdout accepts full Unicode on Windows consoles (CP1252 → UTF-8).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _latex_to_readable(latex: str) -> str:
    """
    Convert a LaTeX math string to a readable Unicode/ASCII form for terminal display.

    Strips $...$ delimiters, converts commands (\\frac, \\dot, Greek letters,
    operators) to Unicode, and simplifies ^{...}/_{...} notation.
    """
    s = latex.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Time derivatives first (before any brace-stripping touches them)
    # \dot{x} → ẋ,  \ddot{x} → ẍ
    s = re.sub(r"\\dot\{([A-Za-z])\}", lambda m: m.group(1) + "\u0307", s)
    s = re.sub(r"\\ddot\{([A-Za-z])\}", lambda m: m.group(1) + "\u0308", s)

    # Iteratively simplify ^{...} and _{...}, innermost first, until stable.
    # This lets complex nested exponents like f^{ - (q^{2} + p^{2}) / 2}
    # resolve in two passes: inner ^{2} → ^2 first, then outer ^{...} → ^(...).
    def _simplify_super(m: re.Match) -> str:
        inner = m.group(1)
        return f"^{inner}" if len(inner) == 1 else f"^({inner})"

    def _simplify_sub(m: re.Match) -> str:
        inner = m.group(1)
        return f"_{inner}" if len(inner) == 1 else f"_({inner})"

    for _ in range(6):   # enough passes for typical nesting depth
        prev = s
        s = re.sub(r"\^\{([^{}]+)\}", _simplify_super, s)
        s = re.sub(r"_\{([^{}]+)\}", _simplify_sub, s)
        if s == prev:
            break

    # Fractions: \frac{a}{b} → (a)/(b).
    # After the ^{...} passes above, numerator/denominator are brace-free.
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)

    # Greek letters (longest first to avoid \varepsilon matching before \epsilon)
    for cmd, ch in sorted(_LATEX_GREEK_TO_UNICODE.items(), key=lambda x: -len(x[0])):
        s = s.replace(cmd, ch)

    # Math operators
    for cmd, ch in sorted(_LATEX_OPS_TO_UNICODE.items(), key=lambda x: -len(x[0])):
        s = s.replace(cmd, ch)

    # Remove spacing/layout commands (\left, \right, \,, etc.)
    s = re.sub(r"\\(?:left|right|,|;|!|quad|qquad|text)\b\s*", " ", s)
    # Strip any remaining unknown backslash commands (e.g. \mathrm, \mathbf)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # Tidy whitespace
    s = re.sub(r" +", " ", s).strip()
    return s


def format_clean_evidence(retrieved: List[Dict[str, Any]], top_k: int = 6) -> None:
    """
    Print ONLY the evidence text and source for clean output.

    Args:
        retrieved: List of retrieved chunks
        top_k: Number of results to display
    """
    print("-" * 60)
    print(f"RETRIEVED CHUNKS ({len(retrieved[:top_k])})")
    print("-" * 60)

    for i, item in enumerate(retrieved[:top_k], 1):
        chunk = item["chunk"]
        doc_id = chunk.get("doc_id", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        text = chunk["text"].strip()

        print(f"\n[CHUNK {i}] Source: {doc_id} (Page {pages})")
        print(f"{text}")
        print("-" * 30)
    print("\n")


def print_retrieved_evidence(retrieved: List[Dict[str, Any]], top_k: int = 6) -> None:
    """
    Print retrieved evidence in standard format.

    Format:
        [1] Score: 0.6000 | Section: text | Pages: [6] | Doc: doc_name
            Preview: First 100 characters...

    Args:
        retrieved: List of retrieved chunks with scores
        top_k: Number of results to display
    """
    print("Top retrieved evidence:")
    for i, item in enumerate(retrieved[:top_k], 1):
        chunk = item["chunk"]
        score = item["score"]
        doc_id = chunk.get("doc_id", "unknown")
        section_type = chunk.get("metadata", {}).get("section_type", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]

        print(f"  [{i}] Score: {score:.4f} | Section: {section_type} | Pages: {pages} | Doc: {doc_id}")
        enc = sys.stdout.encoding or "utf-8"
        safe_preview = text_preview.encode(enc, errors="replace").decode(enc, errors="replace")
        print(f"      Preview: {safe_preview}")
    print()


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.

    Args:
        title: Header text
        width: Total width of header line
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width + "\n")


def print_final_answer(answer: str, width: int = 60) -> None:
    """
    Print the final answer with formatting.

    Args:
        answer: The generated answer text
        width: Width for separator lines
    """
    print_section_header("FINAL ANSWER", width)
    print(answer)
    print("\n" + "=" * width + "\n")


def format_query_session(query: str, retrieved: List[Dict[str, Any]], answer: str, top_k: int = 6) -> str:
    """
    Format a complete query session as a string (for logging/saving).

    Args:
        query: The user query
        retrieved: Retrieved evidence chunks
        answer: Generated answer
        top_k: Number of evidence items to include

    Returns:
        Formatted string with entire session
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"QUERY: {query}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Top retrieved evidence:")
    for i, item in enumerate(retrieved[:top_k], 1):
        chunk = item["chunk"]
        score = item["score"]
        doc_id = chunk.get("doc_id", "unknown")
        section_type = chunk.get("metadata", {}).get("section_type", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]

        lines.append(f"  [{i}] Score: {score:.4f} | Section: {section_type} | Pages: {pages} | Doc: {doc_id}")
        lines.append(f"      Preview: {text_preview}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("FINAL ANSWER")
    lines.append("=" * 60)
    lines.append("")
    lines.append(answer)
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _cleanup_prose_latex(text: str) -> str:
    """
    Convert stray LaTeX commands in prose text (outside any formula block)
    to readable Unicode. Handles \\partial, \\frac, Greek letters, and operators
    that the LLM may write directly in the answer text.
    """
    # Fractions in prose: \frac{a}{b} → (a)/(b), a few passes for nesting
    for _ in range(3):
        text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)

    # Greek and math operators
    all_cmds = {**_LATEX_GREEK_TO_UNICODE, **_LATEX_OPS_TO_UNICODE}
    for cmd, ch in sorted(all_cmds.items(), key=lambda x: -len(x[0])):
        text = text.replace(cmd, ch)

    # Remove spacing/layout commands
    text = re.sub(r"\\(?:left|right|,|;|!|quad|qquad|text)\b\s*", " ", text)

    # Strip remaining unknown backslash-word sequences (e.g. \partialf → partialf)
    # Only strip if followed by a letter (to avoid removing real \ in file paths etc.)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)

    return text


def render_formula_answer(answer: str) -> str:
    """
    Post-process the LLM's raw answer for terminal display.

    - [FORMULA]$...$[/FORMULA] (LaTeX): converted to readable Unicode via
      _latex_to_readable() — Greek symbols, operators, fractions, dot-accents.
    - [FORMULA]...[/FORMULA] where content has LaTeX \\ commands but no $:
      also passed through _latex_to_readable() (LLM forgot the $ delimiters).
    - [FORMULA]...[/FORMULA] (legacy ASCII-math): Greek ASCII names → Unicode,
      spaces around ^ removed.
    - Stray LaTeX commands in prose (\\partial, \\frac, etc.) are cleaned up.

    Note: the API path does NOT call this function — it returns the raw
    [FORMULA]$...$[/FORMULA] blocks so the frontend can render them with KaTeX.

    Args:
        answer: Raw answer string from the LLM.

    Returns:
        Terminal-display-ready answer string.
    """
    def _render_block(m: re.Match) -> str:
        content = m.group(1).strip()
        # LaTeX path A: properly wrapped as $...$
        if content.startswith("$") and content.endswith("$"):
            return f"\n    {_latex_to_readable(content)}\n"
        # LaTeX path B: LLM wrote LaTeX commands but forgot the $ delimiters
        if re.search(r"\\[A-Za-z]", content):
            return f"\n    {_latex_to_readable(content)}\n"
        # Legacy ASCII-math path
        f = content
        for ascii_name, unicode_char in sorted(_GREEK_ASCII_TO_UNICODE.items(), key=lambda x: -len(x[0])):
            f = re.sub(rf"\b{ascii_name}", unicode_char, f)
        f = re.sub(r"\s*\^\s*", "^", f)
        return f"\n    {f.strip()}\n"

    result = re.sub(r"\[FORMULA\](.*?)\[/FORMULA\]", _render_block, answer, flags=re.DOTALL)
    # Clean up any stray LaTeX commands the LLM wrote directly in prose text
    return _cleanup_prose_latex(result)
