"""
Analyze the exact formula formats from lennard_jones_search.json
"""
import json

with open("lennard_jones_search.json") as f:
    data = json.load(f)

print("ANALYZING FORMULAS FROM JSON")
print("=" * 70)

for i, fc in enumerate(data['sample_formula_chunks']):
    formula = fc.get('normalized_formula', fc.get('formula_text', ''))
    print(f"\n[FORMULA {i+1}]: {repr(formula)}")
    print(f"  Trivial: {fc.get('is_trivial')}")
    print(f"  Quality: {fc.get('quality_score')}")
    
# The issue is clear from the lj_analysis.txt:
# - "p * = 0.8442" has SPACES
# - "T * = 0.72" has SPACES  
# - "r. = 2.5o" has a DOT in variable name

print("\n\n" + "=" * 70)
print("THE PROBLEM:")
print("=" * 70)
print("1. Formulas have SPACES: 'p * = 0.8442' not 'p*=0.8442'")
print("2. Regex doesn't handle spaces between variable and equals")
print("3. Variable names have special chars: 'p*', 'r.'")
print("4. Values have special chars: '2.5o' (invalid number)")
