"""
Show Lennard-Jones chunks cleanly (ASCII only, no encoding issues).
"""
import json

with open("lennard_jones_search.json", encoding="utf-8") as f:
    data = json.load(f)

def safe(s, limit=300):
    return s.encode("ascii", errors="replace").decode("ascii")[:limit]

print("=" * 70)
print(f"TEXT CHUNKS: {data['text_chunks_count']}   FORMULA CHUNKS: {data['formula_chunks_count']}")
print("=" * 70)

print("\n--- FORMULA CHUNKS ---")
for i, fc in enumerate(data["sample_formula_chunks"]):
    formula = fc.get("normalized_formula") or fc.get("formula_text", "")
    context = fc.get("context_text", "")
    source  = fc.get("extraction_source", "?")
    trivial = fc.get("is_trivial", "?")
    quality = fc.get("quality_score", "?")
    doc     = fc.get("doc_id", "?")
    page    = fc.get("page_number", "?")
    print(f"\n[F{i+1}] {safe(formula)}")
    print(f"  Doc    : {safe(doc)}")
    print(f"  Page   : {page}")
    print(f"  Source : {source}")
    print(f"  Trivial: {trivial}  Quality: {quality}")
    print(f"  Context: {safe(context, 200)}...")

print("\n--- TEXT CHUNKS ---")
for i, tc in enumerate(data["sample_text_chunks"][:3]):
    text = tc.get("text", "")
    doc  = tc.get("doc_id", "?")
    page = tc.get("metadata", {}).get("pages", "?")
    print(f"\n[T{i+1}] Doc: {safe(doc)}  Page: {page}")
    print(f"  Text: {safe(text, 300)}...")
