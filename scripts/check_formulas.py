"""
Check actual formula formats from the index
"""
import pickle
import re

# Load formula chunks
with open("data/index/formula_chunks.pkl", "rb") as f:
    formula_chunks = pickle.load(f)

# Find Lennard-Jones related formulas
lj_formulas = []
for fc in formula_chunks:
    text = fc.get("formula_text", "").lower()
    context = fc.get("context_text", "").lower()
    if "lennard" in text or "lennard" in context or "jones" in text or "jones" in context:
        lj_formulas.append(fc)

print("=" * 70)
print(f"FOUND {len(lj_formulas)} LENNARD-JONES RELATED FORMULAS")
print("=" * 70)

# Test trivial detection
_TRIVIAL_FORMULA_RE = re.compile(
    r"^\s*[A-Za-zα-ωΑ-Ω][\w\*]?\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)*\s*,?\s*$"
)

for i, fc in enumerate(lj_formulas[:5]):
    print(f"\n[FORMULA {i+1}]")
    formula_text = fc.get("formula_text", "")
    normalized = fc.get("normalized_formula", "")
    print(f"  Raw: {repr(formula_text)}")
    print(f"  Normalized: {repr(normalized)}")
    print(f"  Is Trivial (stored): {fc.get('is_trivial')}")
    
    # Test both patterns
    match1 = _TRIVIAL_FORMULA_RE.match(normalized)
    match2 = re.match(r"^\s*[A-Za-z]\s*=\s*[A-Za-z0-9.]+\s*$", normalized)
    
    print(f"  Pattern 1 match: {bool(match1)}")
    print(f"  Pattern 2 match: {bool(match2)}")
    print(f"  Context: {fc.get('context_text', '')[:100]}...")
