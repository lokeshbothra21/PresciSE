import pickle
import faiss
import os
import sys
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.formula.formula_schema import FormulaChunk

def verify_formula_stats():
    index_dir = "data/index"
    formula_chunks_path = os.path.join(index_dir, "formula_chunks.pkl")
    formula_index_path = os.path.join(index_dir, "formula_faiss_index.bin")
    
    if not os.path.exists(formula_chunks_path):
        logger.error(f"❌ Formula chunks file not found: {formula_chunks_path}")
        return
    
    logger.info(f"Loading formula chunks from {formula_chunks_path}...")
    with open(formula_chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    count = len(chunks)
    logger.info(f"✅ Loaded {count} formula chunks")
    
    if count > 0:
        first = chunks[0]
        logger.info(f"debug: type={type(first)}")
        logger.info(f"debug: dir={dir(first)}")
        if hasattr(first, '__dict__'):
            logger.info(f"debug: dict keys={first.__dict__.keys()}")
    
    source_stats = {}
    for i, chunk in enumerate(chunks):
        # Handle dict or object
        if isinstance(chunk, dict):
            doc = chunk.get('doc_id', 'unknown')
        else:
             # Try safe access
            doc = getattr(chunk, 'doc_id', 'unknown')
            
        source_stats[doc] = source_stats.get(doc, 0) + 1
    
    logger.info(f"📊 Formula distribution across {len(source_stats)} documents:")
    for doc, c in sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"   - {doc}: {c} formulas")
    
    # Verify FAISS index
    if os.path.exists(formula_index_path):
        index = faiss.read_index(formula_index_path)
        logger.info(f"✅ FAISS Index loaded. Size: {index.ntotal}")
        if index.ntotal != count:
            logger.error(f"❌ Mismatch! FAISS has {index.ntotal} vectors, but chunks has {count}")
        else:
            logger.info("✅ FAISS index size matches chunk count")
    else:
        logger.error(f"❌ FAISS index file not found: {formula_index_path}")

if __name__ == "__main__":
    verify_formula_stats()
