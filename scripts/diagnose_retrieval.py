import sys
import os
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persistence import IndexManager
from core.embeddings import Embedder
from core.nlp.tokenizer import tokenize

def diagnose(query):
    with open("diagnosis.txt", "w", encoding="utf-8", errors="replace") as out:
        out.write(f"Diagnosing retrieval for query: '{query}'\n")
        
        # Load index
        manager = IndexManager()
        if not manager._indexes_exist():
            out.write("Indexes do not exist!\n")
            return

        retriever = manager.load_or_build()
        out.write("\nIndexes loaded.\n")
        out.write(f"Formula Weight: {retriever.formula_weight}\n")
        out.write(f"Formula Threshold: {retriever.formula_threshold}\n")
        
        # Embed query
        embedder = Embedder()
        query_emb = embedder.embed_text(query)
        query_tokens = tokenize(query)
        
        out.write("\n--- FAISS Retrieval (Formulas only) ---\n")
        if retriever.formula_faiss:
            raw_formulas = retriever.formula_faiss.retrieve(query_emb, top_k=20)
            
            for i, item in enumerate(raw_formulas):
                score = item["score"]
                chunk = item["chunk"]
                out.write(f"[{i}] Score: {score:.4f}\n")
                out.write(f"    Formula: {chunk.get('formula_text', '')[:100]}...\n")
                
                # Check threshold
                if score < retriever.formula_threshold:
                    out.write(f"    [REJECTED] Below threshold {retriever.formula_threshold}\n")
                else:
                    weighted = score * retriever.formula_weight
                    out.write(f"    [ACCEPTED] Weighted Score: {weighted:.4f} (Weight: {retriever.formula_weight})\n")
                out.write("-" * 40 + "\n")
        else:
            out.write("No formula index loaded.\n")

        out.write("\n--- Full Retrieval (Top 15) ---\n")
        results = retriever.retrieve_with_formulas(query_tokens, query_embedding=query_emb, top_k=15)
        
        for i, res in enumerate(results):
            score = res["score"]
            ctype = "FORMULA" if res.get("source") == "formula_index" else "TEXT"
            out.write(f"[{i}] {ctype} | Score: {score:.4f} | ID: {res['chunk']['chunk_id']}\n")

if __name__ == "__main__":
    query = "Explain the extended Hamiltonian"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    diagnose(query)
