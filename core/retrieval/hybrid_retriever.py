from typing import List, Dict, Any
from .bm25_retriever import BM25Retriever
from core.vectordb.faiss_retriever import FAISSRetriever


class HybridRetriever:
    """
    Hybrid retrieval = BM25 (lexical) + FAISS (semantic).

    Returns a unified ranked list of chunks with combined scores.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25_top_k: int = 5,
        faiss_top_k: int = 5,
        bm25_weight: float = 0.6,
        faiss_weight: float = 0.4,
    ):
        self.chunks = chunks

        self.bm25_top_k = bm25_top_k
        self.faiss_top_k = faiss_top_k

        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight

        # BM25 uses tokens
        self.bm25 = BM25Retriever(chunks)

        # FAISS uses embeddings
        self.faiss = FAISSRetriever(chunks)

    def retrieve(self, query_tokens: List[str], query_embedding: List[float], top_k: int = 5):
        bm25_ranked = self.bm25.retrieve(query_tokens, top_k=self.bm25_top_k)
        faiss_ranked = self.faiss.retrieve(query_embedding, top_k=self.faiss_top_k)

        # Convert both outputs to a consistent score dict (chunk_id → score)
        bm25_scores = {}
        for doc, score in bm25_ranked:
            bm25_scores[doc["chunk_id"]] = float(score)

        faiss_scores = {}
        for item in faiss_ranked:
            chunk = item["chunk"]
            score = item["score"]
            faiss_scores[chunk["chunk_id"]] = float(score)

        # Normalize BM25 scores into [0,1]
        if bm25_scores:
            bm25_max = max(bm25_scores.values())
            bm25_min = min(bm25_scores.values())
            denom = (bm25_max - bm25_min) if bm25_max != bm25_min else 1.0
            for k in bm25_scores:
                bm25_scores[k] = (bm25_scores[k] - bm25_min) / denom

        # Normalize FAISS scores into [0,1]
        if faiss_scores:
            faiss_max = max(faiss_scores.values())
            faiss_min = min(faiss_scores.values())
            denom = (faiss_max - faiss_min) if faiss_max != faiss_min else 1.0
            for k in faiss_scores:
                faiss_scores[k] = (faiss_scores[k] - faiss_min) / denom

        # Merge + deduplicate
        merged = {}
        for c in self.chunks:
            cid = c["chunk_id"]
            merged[cid] = {
                "chunk": c,
                "bm25_score": bm25_scores.get(cid, 0.0),
                "faiss_score": faiss_scores.get(cid, 0.0),
            }

        # Final weighted score
        results = []
        for cid, item in merged.items():
            final_score = (
                self.bm25_weight * item["bm25_score"] +
                self.faiss_weight * item["faiss_score"]
            )
            results.append({
                "chunk": item["chunk"],
                "score": final_score,
                "bm25_score": item["bm25_score"],
                "faiss_score": item["faiss_score"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
