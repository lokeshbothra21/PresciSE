"""Quick diagnostic: test one PDF through the full pipeline with logging."""
import sys, os, time
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

print("=== DIAGNOSTIC: Single PDF Pipeline Test ===\n")

# Step 1: Load PDF
print("[1] Loading PDF...")
t0 = time.time()
from core.ingestion.pdf_loader import load_pdf_as_document
doc = load_pdf_as_document("data/pdfs/Verlet.pdf")
print(f"    ✅ Loaded in {time.time()-t0:.1f}s")
print(f"    Sections: {len(doc.sections) if hasattr(doc, 'sections') else 'N/A'}")

# Step 2: Text chunks
print("[2] Creating text chunks...")
t0 = time.time()
from core.chunking.pdf_chunker import make_chunks_from_doc
chunks = make_chunks_from_doc(doc)
print(f"    ✅ {len(chunks)} chunks in {time.time()-t0:.1f}s")

# Step 3: Formula extraction
print("[3] Extracting formulas...")
t0 = time.time()
from core.formula.formula_chunker import extract_formulas_from_doc
formulas = extract_formulas_from_doc(doc)
print(f"    ✅ {len(formulas)} formulas in {time.time()-t0:.1f}s")
for i, f in enumerate(formulas[:3]):
    print(f"    [{i+1}] {f.get('formula_text', '')[:60]}")

# Step 4: Embeddings
print("[4] Embedding text chunks...")
t0 = time.time()
from core.embeddings.embedder import Embedder
embedder = Embedder()
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_texts(texts)
for c, emb in zip(chunks, embeddings):
    c["embedding"] = emb
print(f"    ✅ Embedded {len(chunks)} text chunks in {time.time()-t0:.1f}s")

# Step 5: Embed formulas
if formulas:
    print("[5] Embedding formula chunks...")
    t0 = time.time()
    formula_texts = [
        f"{fc.get('context_text', '')} {fc.get('formula_text', '')}".strip()
        for fc in formulas
    ]
    formula_embeddings = embedder.embed_texts(formula_texts)
    for fc, emb in zip(formulas, formula_embeddings):
        fc["embedding"] = emb
    print(f"    ✅ Embedded {len(formulas)} formula chunks in {time.time()-t0:.1f}s")
else:
    print("[5] No formulas to embed, skipping.")

# Step 6: Build retriever
print("[6] Building HybridRetriever...")
t0 = time.time()
from core.retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever(chunks, formula_chunks=formulas)
print(f"    ✅ Retriever built in {time.time()-t0:.1f}s")

# Step 7: Save indexes
print("[7] Saving indexes...")
os.makedirs("data/index", exist_ok=True)
t0 = time.time()
import pickle
with open("data/index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
with open("data/index/formula_chunks.pkl", "wb") as f:
    pickle.dump(formulas, f)
retriever.save("data/index")
print(f"    ✅ Saved in {time.time()-t0:.1f}s")

print("\n🎉 ALL STEPS COMPLETE - Pipeline is working!")
