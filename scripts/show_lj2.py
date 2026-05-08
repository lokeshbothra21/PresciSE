import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open("lennard_jones_search.json", encoding="utf-8") as f:
    data = json.load(f)
def s(x, n=300): return str(x).encode('ascii','replace').decode('ascii')[:n]
print(f"TEXT: {data['text_chunks_count']}   FORMULA: {data['formula_chunks_count']}")
print()
print("=== FORMULA CHUNKS ===")
for i, fc in enumerate(data["sample_formula_chunks"]):
    formula = fc.get("normalized_formula") or fc.get("formula_text", "")
    print(f"[F{i+1}] formula = {s(formula)}")
    print(f"     source  = {fc.get('extraction_source')}")
    print(f"     trivial = {fc.get('is_trivial')}   quality = {fc.get('quality_score')}")
    print(f"     context = {s(fc.get('context_text', ''), 250)}")
    print()
print("=== TEXT CHUNKS ===")
for i, tc in enumerate(data["sample_text_chunks"][:3]):
    print(f"[T{i+1}] doc = {s(tc.get('doc_id', ''))}")
    print(f"     text = {s(tc.get('text', ''), 400)}")
    print()
