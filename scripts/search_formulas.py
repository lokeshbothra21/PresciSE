import pickle
import sys
import os

def search_formulas(query):
    path = "data/index/formula_chunks.pkl"
    if not os.path.exists(path):
        print("Formula index not found.")
        return

    with open(path, "rb") as f:
        chunks = pickle.load(f)
    
    with open("found_formulas.txt", "w", encoding="utf-8", errors="replace") as out:
        out.write(f"Searching {len(chunks)} extracted formulas for: '{query}'\n")
        
        matches = []
        for c in chunks:
            ft = c.get("formula_text", "")
            ct = c.get("context_text", "")
            # Search in both formula text and context text
            if query.lower() in ft.lower() or query.lower() in ct.lower():
                matches.append((ft, ct, c))
        
        out.write(f"Found {len(matches)} matches.\n\n")
        for i, (ft, ct, chunk) in enumerate(matches[:20]):
            out.write(f"[{i}]\n")
            out.write(f"  Formula: {ft}\n")
            out.write(f"  Context: {ct[:300]}...\n")
            out.write("-" * 40 + "\n")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Lennard-Jones"
    search_formulas(query)
