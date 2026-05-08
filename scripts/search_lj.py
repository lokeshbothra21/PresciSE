"""
Search for Lennard-Jones content in indexed chunks.
"""
import pickle
import json

print("="*70)
print("SEARCHING FOR LENNARD-JONES CONTENT")
print("="*70)

# Load chunks
try:
    with open("data/index/chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
    print(f"\n✅ Loaded {len(text_chunks)} text chunks")
except Exception as e:
    print(f"\n❌ Could not load text chunks: {e}")
    text_chunks = []

try:
    with open("data/index/formula_chunks.pkl", "rb") as f:
        formula_chunks = pickle.load(f)
    print(f"✅ Loaded {len(formula_chunks)} formula chunks")
except Exception as e:
    print(f"❌ Could not load formula chunks: {e}")
    formula_chunks = []

print("\n" + "="*70)
print("TEXT CHUNKS WITH 'LENNARD' OR 'JONES'")
print("="*70)

lj_text_chunks = []
for i, chunk in enumerate(text_chunks):
    text = chunk.get("text", "").lower()
    if "lennard" in text or "jones" in text:
        lj_text_chunks.append(chunk)
        print(f"\n[TEXT CHUNK {i}]")
        print(f"Doc: {chunk.get('doc_id', 'N/A')}")
        print(f"Page: {chunk.get('page_number', 'N/A')}")
        print(f"Text: {chunk.get('text', '')[:300]}...")
        if len(lj_text_chunks) >= 5:  # Limit to first 5
            break

print(f"\n\nTotal text chunks with Lennard-Jones: {len(lj_text_chunks)}")

print("\n" + "="*70)
print("FORMULA CHUNKS WITH 'LENNARD' OR 'JONES'")
print("="*70)

lj_formula_chunks = []
for i, chunk in enumerate(formula_chunks):
    formula = chunk.get("formula_text", "").lower()
    context = chunk.get("context_text", "").lower()
    embedding_text = chunk.get("embedding_text", "").lower()
    
    if any(kw in text for kw in ["lennard", "jones"] for text in [formula, context, embedding_text]):
        lj_formula_chunks.append(chunk)
        print(f"\n[FORMULA CHUNK {i}]")
        print(f"Doc: {chunk.get('doc_id', 'N/A')}")
        print(f"Page: {chunk.get('page_number', 'N/A')}")
        print(f"Formula: {chunk.get('normalized_formula', chunk.get('formula_text', 'N/A'))}")
        print(f"Context: {chunk.get('context_text', '')[:200]}...")
        print(f"Is Trivial: {chunk.get('is_trivial', False)}")
        print(f"Quality Score: {chunk.get('quality_score', 'N/A')}")
        if len(lj_formula_chunks) >= 5:  # Limit to first 5
            break

print(f"\n\nTotal formula chunks with Lennard-Jones: {len(lj_formula_chunks)}")

print("\n" + "="*70)
print("EMBEDDING TEXT SAMPLE (for formula search)")
print("="*70)

if lj_formula_chunks:
    print("\nFirst Lennard-Jones formula chunk embedding text:")
    print(lj_formula_chunks[0].get("embedding_text", "N/A")[:500])

# Save detailed results
with open("lennard_jones_search.json", "w") as f:
    json.dump({
        "text_chunks_count": len(lj_text_chunks),
        "formula_chunks_count": len(lj_formula_chunks),
        "sample_text_chunks": lj_text_chunks[:3],
        "sample_formula_chunks": lj_formula_chunks[:3]
    }, f, indent=2, default=str)

print(f"\n\n✅ Detailed results saved to: lennard_jones_search.json")
