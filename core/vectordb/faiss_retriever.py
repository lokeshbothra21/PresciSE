from typing import List, Dict
import faiss
import os
from core.vectordb.faiss_index import FAISSIndex


class FAISSRetriever:
    """
    Semantic retriever over chunks.

    Requires chunks to already contain embeddings.
    """

    def __init__(self, chunks: List[Dict]):
        """
        chunks: list of dict OR Chunk objects converted to dict with:
        - chunk_id
        - text
        - embedding
        - metadata (optional)
        """
        self.chunks = chunks

        # detect embedding dimension from first chunk
        first_emb = chunks[0].get("embedding")
        if first_emb is None:
            raise ValueError("Chunks must contain embeddings before building FAISS index.")
        embedding_dim = len(first_emb)

        self.index = FAISSIndex(embedding_dim)

        embeddings = []
        for c in chunks:
            emb = c.get("embedding")
            if emb is None:
                raise ValueError("All chunks must have embeddings.")
            embeddings.append(emb)

        self.index.add(embeddings)

    def retrieve(self, query_embedding: List[float], top_k: int = 5):
        scores, indices = self.index.search(query_embedding, top_k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"chunk": chunk, "score": float(score)})

        return results
    
    def save_retriever(self, filepath: str) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            filepath: Path to save FAISS index file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index.get_persistable_index(), filepath)
    
    def load_retriever(self, filepath: str) -> None:
        """
        Load FAISS index from disk.
        
        Args:
            filepath: Path to FAISS index file
        """
        cpu_index = faiss.read_index(filepath)
        self.index.load_from_cpu_index(cpu_index)
