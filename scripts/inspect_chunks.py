import pickle
import sys
import os
from pprint import pprint

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_chunks(search_term):
    # Load text chunks
    chunks_path = "data/index/chunks.pkl"
    formula_chunks_path = "data/index/formula_chunks.pkl"
    
    found_count = 0
    
    try:
        with open("found_chunks.txt", "w", encoding="utf-8", errors="replace") as out:
            out.write(f"Searching for '{search_term}' (actually 'hamiltonian')\n")
            
            # Text Chunks
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                out.write(f"Loaded {len(chunks)} text chunks.\n")
                
                if chunks:
                    try:
                        out.write(f"First chunk keys: {list(chunks[0].keys())}\n")
                        out.write(f"First chunk content: {str(chunks[0])}\n")
                    except Exception as e:
                        out.write(f"Error printing first chunk: {e}\n")

                count = 0
                for i, chunk in enumerate(chunks):
                    try:
                        # Search in full string representation just in case
                        full_text = str(chunk)
                        if "hamiltonian" in full_text.lower():
                            count += 1
                            out.write(f"\n[TEXT MATCH {count}] Chunk ID: {chunk.get('chunk_id', 'unknown')}\n")
                            out.write(f"Source: {chunk.get('doc_id')}, Page: {chunk.get('metadata', {}).get('pages')}\n")
                            out.write("-" * 40 + "\n")
                            text = chunk.get("text", "")
                            out.write(text + "\n")
                            out.write("-" * 40 + "\n")
                            if count >= 3: break
                    except Exception as e:
                        print(f"Error processing chunk {i}: {e}")
            
            # Formula Chunks
            if os.path.exists(formula_chunks_path):
                with open(formula_chunks_path, "rb") as f:
                    f_chunks = pickle.load(f)
                out.write(f"\nLoaded {len(f_chunks)} formula chunks.\n")
                
                count = 0
                for i, chunk in enumerate(f_chunks):
                    try:
                        if isinstance(chunk, dict):
                            text = chunk.get("text", "")
                        else:
                            text = getattr(chunk, "text", "")
                        
                        if "hamiltonian" in text.lower():
                            count += 1
                            out.write(f"\n[FORMULA MATCH {count}]\n")
                            out.write(text + "\n")
                            out.write("-" * 40 + "\n")
                            if count >= 3: break
                    except Exception as e:
                        print(f"Error processing formula chunk {i}: {e}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    term = "extended hamiltonian"
    if len(sys.argv) > 1:
        term = sys.argv[1]
    inspect_chunks(term)
