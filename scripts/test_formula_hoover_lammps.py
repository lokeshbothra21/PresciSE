"""
Formula extraction and retrieval test for Hoover and LAMMPS PDFs.
No emojis - safe for Windows cp1252 terminals.
"""
import sys
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

INDEX_DIR = Path("data/index")
PDF_DIR = Path("data/pdfs")

TARGET_PDFS = {
    "hoover": "Hoover2024_NoseHoover_canonical_temperature_control.pdf",
    "lammps": "LAMMPS_tutorials_Kohlmeyer2025.pdf",
}

FORMULA_QUERIES = [
    "equation of motion Nose-Hoover thermostat",
    "Nose-Hoover extended Lagrangian",
    "canonical ensemble temperature control",
    "LAMMPS parallel molecular dynamics force calculation",
    "Newton equations of motion integration",
    "velocity Verlet algorithm",
]


# ============================================================
# CHECK 1: Formula Extraction from PDFs
# ============================================================
print("=" * 70)
print("  FORMULA EXTRACTION TEST (Hoover + LAMMPS)")
print("=" * 70)

from core.ingestion.pdf_loader import load_pdf_as_document
from core.formula.formula_chunker import extract_formulas_from_doc

results = {}
for label, filename in TARGET_PDFS.items():
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        print(f"\n[{label.upper()}] MISSING: {filename}")
        results[label] = []
        continue

    print(f"\n[{label.upper()}] {filename}")
    try:
        doc = load_pdf_as_document(str(pdf_path))
        formulas = extract_formulas_from_doc(doc)
        results[label] = formulas
        print(f"  Formulas extracted: {len(formulas)}")
        for i, fc in enumerate(formulas[:5]):
            ft = fc.get("formula_text", "")[:80]
            nf = fc.get("normalized_formula", "")[:60]
            ctx = fc.get("context_text", "")[:100].replace("\n", " ")
            print(f"  [{i+1}] {ft}")
            print(f"       norm: {nf}")
            print(f"       ctx:  {ctx}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results[label] = []

hoover_count = len(results.get("hoover", []))
lammps_count = len(results.get("lammps", []))
print(f"\nSummary: Hoover={hoover_count} formulas, LAMMPS={lammps_count} formulas")


# ============================================================
# CHECK 2: Index Integrity
# ============================================================
print("\n" + "=" * 70)
print("  INDEX INTEGRITY CHECK")
print("=" * 70)

index_files = {
    "chunks.pkl": INDEX_DIR / "chunks.pkl",
    "formula_chunks.pkl": INDEX_DIR / "formula_chunks.pkl",
    "bm25_index.pkl": INDEX_DIR / "bm25_index.pkl",
    "faiss_index.bin": INDEX_DIR / "faiss_index.bin",
    "formula_faiss_index.bin": INDEX_DIR / "formula_faiss_index.bin",
}

index_ready = all(p.exists() for p in index_files.values())
for name, path in index_files.items():
    status = "OK" if path.exists() else "MISSING"
    size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else ""
    print(f"  {status:7s} {name:35s} {size}")

if not index_ready:
    print("\nIndex not fully built. Skipping retrieval test.")
    print("Run: python -m scripts.pdf_demo  to build the index first.")
    sys.exit(0)


# ============================================================
# CHECK 3: Formula Chunks per Document
# ============================================================
print("\n" + "=" * 70)
print("  FORMULA CHUNKS IN INDEX (per source document)")
print("=" * 70)

with open(INDEX_DIR / "formula_chunks.pkl", "rb") as f:
    formula_chunks = pickle.load(f)

print(f"Total formula chunks: {len(formula_chunks)}")

from collections import Counter
doc_counts = Counter(c.get("doc_id", "unknown") for c in formula_chunks)
for doc_id, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
    marker = " <-- Hoover" if "Hoover" in doc_id or "hoover" in doc_id else \
             " <-- LAMMPS" if "LAMMPS" in doc_id or "lammps" in doc_id or "Parallelization" in doc_id else ""
    print(f"  {count:4d}  {doc_id}{marker}")


# ============================================================
# CHECK 4: Retrieval Test
# ============================================================
print("\n" + "=" * 70)
print("  FORMULA RETRIEVAL TEST")
print("=" * 70)

from core.retrieval.hybrid_retriever import HybridRetriever
from core.embeddings.embedder import Embedder
from core.nlp.tokenizer import tokenize

with open(INDEX_DIR / "chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

retriever = HybridRetriever(text_chunks, formula_chunks=formula_chunks)
retriever.load(str(INDEX_DIR))

embedder = Embedder()

for query in FORMULA_QUERIES:
    print(f"\nQuery: \"{query}\"")
    q_tokens = tokenize(query)
    q_emb = embedder.embed_text(query)

    results_ret = retriever.retrieve_with_formulas(q_tokens, q_emb, top_k=6)

    formula_hits = [r for r in results_ret if r.get("source") == "formula_index"]
    text_hits    = [r for r in results_ret if r.get("source") != "formula_index"]
    print(f"  text={len(text_hits)}  formula={len(formula_hits)}")

    for fh in formula_hits[:3]:
        chunk = fh["chunk"]
        ft    = chunk.get("formula_text", "")[:70]
        doc   = chunk.get("doc_id", "")
        score = fh["score"]
        print(f"  FORMULA [{score:.4f}] {ft}  | {doc}")

    for th in text_hits[:2]:
        chunk = th["chunk"]
        txt   = chunk.get("text", "")[:80].replace("\n", " ")
        doc   = chunk.get("doc_id", "")
        score = th["score"]
        print(f"  TEXT    [{score:.4f}] {txt}  | {doc}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
