import re

_TRIVIAL_FORMULA_RE = re.compile(
    r"^\s*[A-Za-zα-ωΑ-Ω][\w\*]?\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)*\s*,?\s*$"
)

test_formulas = [
    "p * = 0.8442",
    "p*=0.8442", 
    "T * = 0.72",
    "T*=0.72",
    "r. = 2.5o",
    "r.=2.5o",
    "x=1",
    "n=5",
    "V(r) = 4ε[(σ/r)^12 - (σ/r)^6]"
]

print("Testing _TRIVIAL_FORMULA_RE:")
print("=" * 60)
for formula in test_formulas:
    match = _TRIVIAL_FORMULA_RE.match(formula)
    print(f"{formula:30} → {'TRIVIAL' if match else 'NOT TRIVIAL'}")

print("\n\nTesting secondary pattern:")
print("=" * 60)
secondary_re = re.compile(r"^\s*[A-Za-z]\s*=\s*[A-Za-z0-9.]+\s*$")
for formula in test_formulas:
    match = secondary_re.match(formula)
    print(f"{formula:30} → {'TRIVIAL' if match else 'NOT TRIVIAL'}")
