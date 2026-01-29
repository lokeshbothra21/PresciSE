from typing import List
import numpy as np
import faiss


class FAISSIndex:
    """
    Simple FAISS wrapper for semantic search.

    We use cosine similarity by storing normalized embeddings
    and using Inner Product similarity (IndexFlatIP).
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings: List[List[float]]):
        vecs = np.array(embeddings, dtype="float32")
        if vecs.ndim != 2 or vecs.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings with dim={self.embedding_dim}, got {vecs.shape}")
        self.index.add(vecs)

    def search(self, query_embedding: List[float], top_k: int = 5):
        q = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(q, top_k)
        return scores[0].tolist(), indices[0].tolist()
