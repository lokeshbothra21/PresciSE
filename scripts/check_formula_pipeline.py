"""
Sanity check script for the formula-aware retrieval pipeline.

Validates:
    1. Formula extraction count per PDF
    2. Formula embedding (dimensions match SPECTER 768)
    3. Retrieval of formula chunks for conceptual queries
    4. Formula index integrity

Usage:
    python -m scripts.check_formula_pipeline
"""

import sys
import os
sys.path.insert(0, ".")

import pickle
from pathlib import Path
from dotenv import load_dotenv


def main():
    load_dotenv()
    
    print("=" * 70)
    print("  PRESCISE FORMULA PIPELINE SANITY CHECK")
    print("=" * 70)
    
    index_dir = Path("data/index")
    pdf_dir = Path("data/pdfs")
    
    # ============================================================
    # CHECK 1: Formula Extraction
    # ============================================================
    print("\n[1/4] FORMULA EXTRACTION TEST")
    print("-" * 50)
    
    from core.ingestion.pdf_loader import load_pdf_as_document
    from core.formula.formula_chunker import extract_formulas_from_doc
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("  ❌ No PDFs found in data/pdfs/")
        return
    
    total_formulas = 0
    for pdf_path in pdf_files:
        print(f"\n  📄 {pdf_path.name}")
        try:
            doc = load_pdf_as_document(str(pdf_path))
            formulas = extract_formulas_from_doc(doc)
            total_formulas += len(formulas)
            print(f"     Formulas found: {len(formulas)}")
            
            # Show first 3 formulas
            for i, fc in enumerate(formulas[:3]):
                formula_text = fc.get("formula_text", "")[:60]
                confidence = fc.get("normalized_formula", "")[:40]
                print(f"     [{i+1}] {formula_text}")
                print(f"         Normalized: {confidence}")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print(f"\n  📊 Total formulas extracted: {total_formulas}")
    if total_formulas == 0:
        print("  ⚠️  No formulas detected. Check PDF content or extractor patterns.")
    else:
        print("  ✅ Formula extraction working")
    
    # ============================================================
    # CHECK 2: Formula Embeddings
    # ============================================================
    print(f"\n[2/4] FORMULA EMBEDDING TEST")
    print("-" * 50)
    
    formula_chunks_path = index_dir / "formula_chunks.pkl"
    
    if not formula_chunks_path.exists():
        print("  ⚠️  No formula_chunks.pkl found. Run pdf_demo.py first to build index.")
        print("  Skipping embedding check...")
    else:
        with open(formula_chunks_path, "rb") as f:
            formula_chunks = pickle.load(f)
        
        print(f"  Loaded {len(formula_chunks)} formula chunks")
        
        if formula_chunks:
            first = formula_chunks[0]
            emb = first.get("embedding")
            if emb is None:
                print("  ❌ Formula chunks have no embeddings!")
            else:
                dim = len(emb)
                print(f"  Embedding dimension: {dim}")
                if dim == 768:
                    print("  ✅ Matches SPECTER (768-dim)")
                else:
                    print(f"  ⚠️  Expected 768 (SPECTER), got {dim}")
            
            # Check required fields
            required = ["chunk_id", "chunk_type", "formula_text", "context_text", "doc_id", "text"]
            missing = [f for f in required if f not in first]
            if missing:
                print(f"  ❌ Missing fields: {missing}")
            else:
                print(f"  ✅ All required fields present")
        else:
            print("  ⚠️  Formula chunks file is empty")
    
    # ============================================================
    # CHECK 3: Formula Retrieval
    # ============================================================
    print(f"\n[3/4] FORMULA RETRIEVAL TEST")
    print("-" * 50)
    
    chunks_path = index_dir / "chunks.pkl"
    if not chunks_path.exists() or not formula_chunks_path.exists():
        print("  ⚠️  Index not built yet. Skipping retrieval test.")
    else:
        from core.retrieval.hybrid_retriever import HybridRetriever
        from core.embeddings.embedder import Embedder
        from core.nlp.tokenizer import tokenize
        
        with open(chunks_path, "rb") as f:
            text_chunks = pickle.load(f)
        with open(formula_chunks_path, "rb") as f:
            formula_chunks = pickle.load(f)
        
        if not formula_chunks:
            print("  ⚠️  No formula chunks to test retrieval")
        else:
            retriever = HybridRetriever(text_chunks, formula_chunks=formula_chunks)
            retriever.load(str(index_dir))
            
            embedder = Embedder()
            
            # Test queries
            test_queries = [
                "What is the equation of motion?",
                "How does temperature affect molecular dynamics?",
                "Explain the Hamiltonian formulation",
            ]
            
            for query in test_queries:
                print(f"\n  🔍 Query: \"{query}\"")
                q_tokens = tokenize(query)
                q_emb = embedder.embed_text(query)
                
                results = retriever.retrieve_with_formulas(q_tokens, q_emb, top_k=5)
                
                formula_hits = [r for r in results if r.get("source") == "formula_index"]
                text_hits = [r for r in results if r.get("source") != "formula_index"]
                
                print(f"     Text chunks: {len(text_hits)}, Formula chunks: {len(formula_hits)}")
                
                for fh in formula_hits[:2]:
                    ft = fh["chunk"].get("formula_text", "")[:60]
                    score = fh["score"]
                    print(f"     📐 Formula: {ft}  (score: {score:.4f})")
    
    # ============================================================
    # CHECK 4: Index Integrity
    # ============================================================
    print(f"\n[4/4] INDEX INTEGRITY CHECK")
    print("-" * 50)
    
    files_to_check = {
        "chunks.pkl": chunks_path,
        "formula_chunks.pkl": formula_chunks_path,
        "bm25_index.pkl": index_dir / "bm25_index.pkl",
        "faiss_index.bin": index_dir / "faiss_index.bin",
        "formula_faiss_index.bin": index_dir / "formula_faiss_index.bin",
    }
    
    for name, path in files_to_check.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {name:30s} ({size_kb:.1f} KB)")
        else:
            status = "⚠️" if "formula" in name else "❌"
            print(f"  {status} {name:30s} (missing)")
    
    print("\n" + "=" * 70)
    print("  SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
