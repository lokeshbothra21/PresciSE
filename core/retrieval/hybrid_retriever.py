from typing import List, Dict, Any
import os
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
        bm25_top_k: int = 15,
        faiss_top_k: int = 15,
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

    def retrieve(self, query_tokens: List[str], query_embedding: List[float], top_k: int = 15):
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
    
    def save(self, index_dir: str) -> None:
        """
        Save both BM25 and FAISS indexes to disk.
        
        Args:
            index_dir: Directory to save indexes
        """
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        
        self.bm25.save_index(bm25_path)
        self.faiss.save_retriever(faiss_path)
    
    def load(self, index_dir: str) -> None:
        """
        Load both BM25 and FAISS indexes from disk.
        
        Args:
            index_dir: Directory containing saved indexes
        """
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        
        self.bm25.load_index(bm25_path)
        self.faiss.load_retriever(faiss_path)
    
    def retrieve_with_router(self, query_tokens: List[str], query_embedding: List[float], 
                            query_str: str, top_k: int = 15) -> Dict[str, Any]:
        """
        Retrieve with dynamic BM25/FAISS weights determined by router.
        
        Uses Qwen2.5-based router to analyze query intent and assign optimal weights.
        Falls back to static weights if router fails.
        
        Args:
            query_tokens: Tokenized query for BM25
            query_embedding: Query embedding for FAISS
            query_str: Original query string for router analysis
            top_k: Number of results to return
            
        Returns:
            {
                "results": [...],           # Retrieved chunks with scores
                "router_decision": {        # Router's weight decision
                    "bm25_weight": float,
                    "faiss_weight": float,
                    "intent": str,
                    "reason": str
                }
            }
        """
        # Lazy load router
        if not hasattr(self, 'router') or self.router is None:
            from core.retrieval.search_router import SearchRouter
            self.router = SearchRouter()
        
        # Get dynamic weights from router
        router_decision = self.router.route(query_str)
        
        print(f"\n  🤖 Router Decision:")
        print(f"     Intent: {router_decision['intent']}")
        print(f"     BM25: {router_decision['bm25_weight']:.2f} | FAISS: {router_decision['faiss_weight']:.2f}")
        print(f"     Reason: {router_decision['reason']}\n")
        
        # Temporarily override weights
        original_bm25_weight = self.bm25_weight
        original_faiss_weight = self.faiss_weight
        
        self.bm25_weight = router_decision["bm25_weight"]
        self.faiss_weight = router_decision["faiss_weight"]
        
        # Run retrieval with dynamic weights
        results = self.retrieve(query_tokens, query_embedding, top_k)
        
        # Restore original weights (for safety)
        self.bm25_weight = original_bm25_weight
        self.faiss_weight = original_faiss_weight
        
        return {
            "results": results,
            "router_decision": router_decision
        }

    def retrieve_progressive(self, query_tokens: List[str], query_embedding: List[float], top_k: int = 15) -> Dict[str, List]:
        """
        Two-phase progressive retrieval for faster perceived latency.
        
        Phase 1: Fast keyword match on metadata (~2ms)
        Phase 2: Full BM25 + FAISS search (~50ms)
        
        Args:
            query_tokens: Tokenized query
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            
        Returns:
            {
                "fast_results": [...],    # from metadata search
                "full_results": [...],    # from BM25+FAISS
                "merged_results": [...]   # deduplicated & ranked
            }
        """
        from core.retrieval.metadata_searcher import search_by_keywords
        
        # Reconstruct query string for metadata search
        query_str = ' '.join(query_tokens)
        
        # PHASE 1: Fast metadata keyword search (~2ms)
        fast_results = search_by_keywords(query_str, self.chunks, top_k=5)
        
        # PHASE 2: Full BM25 + FAISS search (existing method)
        full_results = self.retrieve(query_tokens, query_embedding, top_k=top_k)
        
        # Merge and deduplicate results
        seen_ids = set()
        merged = []
        
        # Add fast results first (prioritize quick matches)
        for item in fast_results:
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in seen_ids:
                merged.append({
                    "chunk": item["chunk"],
                    "score": item["metadata_score"],
                    "source": "metadata_fast",
                    "matched_keywords": item.get("matched_keywords", [])
                })
                seen_ids.add(chunk_id)
        
        # Add full search results (avoid duplicates)
        for item in full_results:
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in seen_ids:
                merged.append({
                    "chunk": item["chunk"],
                    "score": item["score"],
                    "source": "full_search"
                })
                seen_ids.add(chunk_id)
        
        # Re-rank by score (full search scores are typically more reliable)
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "fast_results": fast_results,
            "full_results": full_results,
            "merged_results": merged[:top_k]
        }
