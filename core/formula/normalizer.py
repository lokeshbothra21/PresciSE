"""
Deterministic formula normalizer for scientific text extraction.

Focus:
- Structural repair (exponents, implicit multiplication)
- Greek symbol normalization to ASCII names
- Operator and whitespace cleanup for readable ASCII math
"""

import re


_GREEK_TO_ASCII = {
    "alpha": ("α", "Α", "Î±", "Î‘"),
    "beta": ("β", "Β", "Î²", "Î’"),
    "gamma": ("γ", "Γ", "Î³", "Î“"),
    "delta": ("δ", "Δ", "Î´", "Î”"),
    "epsilon": ("ε", "Ε", "Îµ", "Î•"),
    "zeta": ("ζ", "Ζ", "Î¶", "Î–"),
    "eta": ("η", "Η", "Î·", "Î—"),
    "theta": ("θ", "Θ", "Î¸", "Î˜"),
    "lambda": ("λ", "Λ", "Î»", "Î›"),
    "mu": ("μ", "Μ", "Î¼", "Îœ"),
    "nu": ("ν", "Ν", "Î½", "Î"),
    "xi": ("ξ", "Ξ", "Î¾", "Îž"),
    "pi": ("π", "Π", "Ï€", "Î "),
    "rho": ("ρ", "Ρ", "Ï", "Î¡"),
    "sigma": ("σ", "Σ", "Ïƒ", "Î£"),
    "tau": ("τ", "Τ", "Ï„", "Î¤"),
    "phi": ("φ", "Φ", "Ï†", "Î¦"),
    "chi": ("χ", "Χ", "Ï‡", "Î§"),
    "psi": ("ψ", "Ψ", "Ïˆ", "Î¨"),
    "omega": ("ω", "Ω", "Ï‰", "Î©"),
}

_SUPERSCRIPT_DIGITS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)

_OPERATORS = r"=+\-*/^∝≡∼≈"
_MATH_FUNC_PREFIXES = {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "max", "min"}

# Greek ASCII names that should be expanded by _replace_var_digit even though
# they are longer than 3 characters (e.g. zeta2 → zeta^2).
_GREEK_ASCII_NAMES = frozenset([
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "chi",
    "psi", "omega",
])


def _replace_var_digit(match: re.Match) -> str:
    token = match.group(0)
    var = match.group(1)
    exp = match.group(2)

    # Skip common function-like tokens and likely chemical all-caps tokens.
    if var.lower() in _MATH_FUNC_PREFIXES:
        return token
    if len(var) > 1 and var.isupper():
        return token
    # Greek ASCII names are always expanded (zeta2 → zeta^2).
    if var.lower() in _GREEK_ASCII_NAMES:
        return f"{var}^{exp}"
    if len(var) > 3 and var.islower():
        return token

    return f"{var}^{exp}"


def _is_math_dense(text: str) -> bool:
    if not text:
        return False
    has_equation_cue = "=" in text or "^" in text or "/" in text or "(" in text or ")" in text
    symbols = len(re.findall(r"[0-9=+\-*/^()_\[\]{}]", text))
    ratio = symbols / max(1, len(text))
    return has_equation_cue or ratio >= 0.18


def _normalize_greek_math_only(text: str) -> str:
    if not _is_math_dense(text):
        return text
    result = text
    for ascii_name, variants in _GREEK_TO_ASCII.items():
        for ch in variants:
            result = result.replace(ch, ascii_name)
    return result


def normalize_formula(formula: str) -> str:
    """
    Normalize formula text into ASCII-readable math representation.
    """
    if not formula:
        return ""

    s = formula.strip()

    # Remove surrounding LaTeX wrappers.
    s = re.sub(r"^\$+|\$+$", "", s)

    # Pre-pass A: collapse PDF-split combining/modifier accents separated from
    # their base letter by whitespace.  "˙ p" → "˙p", "¨ r" → "¨r".
    s = re.sub(r"([\u02d9\u00a8\u0307\u0308])\s+([A-Za-z])", r"\1\2", s)

    # Pre-pass B: detect any variable/function immediately followed by a negative
    # parenthesized exponent expression and rewrite BEFORE superscript digits are
    # stripped (so structure is preserved).
    # Handles: "e -(q²+p²)/2"  → "e^(-(q^2+p^2)/2)"
    #          "f -(x²+y²)/2"  → "f^(-(x^2+y^2)/2)"
    #          "A -(r²+s²)"    → "A^(-(r^2+s^2))"
    #          "e -H/2"        → "e^(-H/2)"  (bare token form)
    # A strong indicator of an implicit exponent is: var <space> sign (...)/num
    # or simply var <space> sign (...) — both are rewritten.
    def _rewrite_implicit_exponent(m: re.Match) -> str:
        var = m.group(1)
        sign = m.group(2)
        expr = m.group(3).strip()
        return f"{var}^({sign}{expr})"

    s = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+([-\u2212])\s*"
        r"(\([^)]*\)(?:\s*/\s*[0-9]+)?|[A-Za-z0-9_\^]+(?:\s*/\s*[0-9]+)?)",
        _rewrite_implicit_exponent,
        s,
    )

    # Convert common superscript digits to caret notation later.
    s = s.replace("Â²", "2").replace("Â³", "3")
    s = s.translate(_SUPERSCRIPT_DIGITS)

    # Exponent normalization:
    # (x)12 -> (x)^12
    s = re.sub(r"(\))([0-9]{1,4})(?=\b)", r"\1^\2", s)
    # r2 -> r^2, mc2 -> mc^2 for variable-like tokens.
    s = re.sub(r"\b([A-Za-z]{1,4})([0-9]{1,3})(?=\b)", _replace_var_digit, s)

    # Multiplication insertion for common implicit cases:
    # 4( -> 4 * (
    s = re.sub(r"([0-9])\s*\(", r"\1 * (", s)
    # )r -> ) * r
    s = re.sub(r"\)\s*([A-Za-z])", r") * \1", s)

    # Normalize Greek-like symbols only in math-dense strings.
    s = _normalize_greek_math_only(s)

    # Operator normalization.
    s = s.replace("**", "^")
    s = s.replace("×", "*").replace("Ã—", "*")
    s = s.replace("÷", "/").replace("Ã·", "/")
    s = s.replace("−", "-").replace("–", "-")

    # Spacing around operators.
    s = re.sub(rf"\s*([{_OPERATORS}])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".,;:")

    # Remove accidental spaces after unary minus in exponents etc.
    s = re.sub(r"\^ - ([0-9]+)", r"^-\1", s)

    return s


def _replace_paren_group(s: str, trigger: str, open_brace: str, close_brace: str) -> str:
    """
    Replace `trigger(...)` → `open_brace...close_brace` with nested-paren awareness.

    E.g. trigger="^", open_brace="^{", close_brace="}" turns
    "f^(-(q^2+p^2)/2)" → "f^{-(q^2+p^2)/2}".
    """
    result: list[str] = []
    i = 0
    t_len = len(trigger)
    while i < len(s):
        # Check for trigger immediately followed by '('
        if s[i:i + t_len] == trigger and i + t_len < len(s) and s[i + t_len] == "(":
            result.append(open_brace)
            i += t_len + 1  # skip trigger + '('
            depth = 1
            while i < len(s) and depth > 0:
                if s[i] == "(":
                    depth += 1
                    result.append(s[i])
                elif s[i] == ")":
                    depth -= 1
                    if depth == 0:
                        result.append(close_brace)
                    else:
                        result.append(s[i])
                else:
                    result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _replace_caret_parens(s: str) -> str:
    return _replace_paren_group(s, "^", "^{", "}")


def _replace_underscore_parens(s: str) -> str:
    return _replace_paren_group(s, "_", "_{", "}")


def ascii_math_to_latex(formula: str) -> str:
    """
    Convert a normalised ASCII-math formula string to renderable LaTeX.

    The output is suitable for wrapping in $...$ and passing to KaTeX/MathJax.
    """
    s = formula

    # 1. Time derivatives: ˙q → \dot{q},  ¨r → \ddot{r}
    s = re.sub(r"[\u02d9\u0307]([A-Za-z])", r"\\dot{\1}", s)
    s = re.sub(r"[\u00a8\u0308]([A-Za-z])", r"\\ddot{\1}", s)

    # 2. Greek ASCII names → LaTeX commands (longest first to avoid partial matches)
    for name in sorted(_GREEK_ASCII_NAMES, key=len, reverse=True):
        s = re.sub(rf"\b{name}\b", rf"\\{name}", s)

    # 3. Relational / proportionality operators
    _OPS = {
        "\u221d": r"\propto", "\u2261": r"\equiv", "\u223c": r"\sim",
        "\u2248": r"\approx", "\u2264": r"\leq", "\u2265": r"\geq",
        "\u2260": r"\neq", "\u2207": r"\nabla", "\u2202": r"\partial",
        "\u2211": r"\sum", "\u220f": r"\prod", "\u222b": r"\int",
        "\u221a": r"\sqrt", "\u00b1": r"\pm", "\u00d7": r"\times",
        "\u00f7": r"\div", "\u221e": r"\infty",
    }
    for sym, cmd in _OPS.items():
        s = s.replace(sym, cmd)

    # 4. Normalise spaces around ^ and _ (added by normalizer's spacing pass)
    s = re.sub(r"\s*\^\s*", "^", s)
    s = re.sub(r"\s*_\s*", "_", s)

    # 5. Superscripts: x^(expr) → x^{expr} with nested-paren awareness;
    #    then bare x^token → x^{token}
    s = _replace_caret_parens(s)
    s = re.sub(r"\^([A-Za-z0-9\\]+)", r"^{\1}", s)

    # 6. Subscripts: x_(expr) → x_{expr} with nested-paren awareness;
    #    then bare x_token → x_{token}
    s = _replace_underscore_parens(s)
    s = re.sub(r"_([A-Za-z0-9\\]+)", r"_{\1}", s)

    # 7. Simple fractions: a/b → \frac{a}{b} (single-token only)
    s = re.sub(
        r"([A-Za-z0-9_{}\\]+)\s*/\s*([A-Za-z0-9_{}\\]+)",
        r"\\frac{\1}{\2}",
        s,
    )

    # 8. Multiplication: * → \cdot
    s = s.replace("*", r"\cdot ")

    return s.strip()
