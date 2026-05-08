from typing import List, Dict, Any, Optional
import os
import threading
from loguru import logger
from .bm25_retriever import BM25Retriever
from core.vectordb.faiss_retriever import FAISSRetriever


class HybridRetriever:
    """
    Hybrid retrieval = BM25 (lexical) + FAISS (semantic) + Formula FAISS.

    Returns a unified ranked list of chunks with combined scores.
    Formula chunks are stored in a separate index and merged at query time.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        formula_chunks: List[Dict[str, Any]] = None,
        bm25_top_k: int = 15,
        faiss_top_k: int = 15,
        bm25_weight: float = 0.6,
        faiss_weight: float = 0.4,
        formula_weight: float = 0.5,
        formula_threshold: float = 0.15,
    ):
        self.chunks = chunks
        self.formula_chunks = formula_chunks or []

        self.bm25_top_k = bm25_top_k
        self.faiss_top_k = faiss_top_k

        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.formula_weight = formula_weight
        self.formula_threshold = formula_threshold

        # Lock protecting formula index hot-swaps from the Nougat background thread
        self._formula_lock = threading.Lock()

        # BM25 uses tokens
        self.bm25 = BM25Retriever(chunks)

        # FAISS uses embeddings (text chunks)
        self.faiss = FAISSRetriever(chunks)

        # Formula indexes — BM25 (lexical on embedding_text tokens) + FAISS (semantic)
        self.formula_faiss = None
        self.formula_bm25 = None
        if self.formula_chunks and len(self.formula_chunks) > 0:
            self.formula_bm25 = BM25Retriever(self.formula_chunks)
            if self.formula_chunks[0].get("embedding") is not None:
                self.formula_faiss = FAISSRetriever(self.formula_chunks)
                logger.info(f"Formula index: {len(self.formula_chunks)} formulas indexed")

    def update_formula_index(self, new_formula_chunks: List[Dict[str, Any]]) -> None:
        """
        Hot-swap the formula index with a new list of formula chunks.

        Thread-safe: acquires _formula_lock before touching any formula state.
        Called by the Nougat background scanner after each PDF is processed.

        Parameters
        ----------
        new_formula_chunks : list[dict]
            Fully-embedded formula chunks (must each have an ``embedding`` key).
        """
        with self._formula_lock:
            self.formula_chunks = new_formula_chunks

            # Rebuild BM25 index (fast, pure-Python)
            self.formula_bm25 = BM25Retriever(new_formula_chunks) if new_formula_chunks else None

            # Rebuild FAISS index (GPU-resident, requires embeddings)
            embedded = [fc for fc in new_formula_chunks if fc.get("embedding") is not None]
            if embedded:
                self.formula_faiss = FAISSRetriever(embedded)
                logger.info(
                    f"[VLM SCAN] Formula index hot-swapped: "
                    f"{len(new_formula_chunks)} chunks ({len(embedded)} embedded)"
                )
            else:
                self.formula_faiss = None

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
        Save BM25, FAISS, and Formula FAISS indexes to disk.
        
        Args:
            index_dir: Directory to save indexes
        """
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        formula_faiss_path = os.path.join(index_dir, "formula_faiss_index.bin")
        
        self.bm25.save_index(bm25_path)
        self.faiss.save_retriever(faiss_path)
        
        if self.formula_faiss is not None:
            self.formula_faiss.save_retriever(formula_faiss_path)
    
    def load(self, index_dir: str) -> None:
        """
        Load BM25, FAISS, and Formula FAISS indexes from disk.
        
        Args:
            index_dir: Directory containing saved indexes
        """
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        formula_faiss_path = os.path.join(index_dir, "formula_faiss_index.bin")
        
        self.bm25.load_index(bm25_path)
        self.faiss.load_retriever(faiss_path)
        
        if self.formula_faiss is not None and os.path.exists(formula_faiss_path):
            self.formula_faiss.load_retriever(formula_faiss_path)
    
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
        
        logger.info(f"Router Decision: Intent={router_decision['intent']}, BM25={router_decision['bm25_weight']:.2f}, FAISS={router_decision['faiss_weight']:.2f}, Reason={router_decision['reason']}")
        
        # Temporarily override weights
        original_bm25_weight = self.bm25_weight
        original_faiss_weight = self.faiss_weight
        
        self.bm25_weight = router_decision["bm25_weight"]
        self.faiss_weight = router_decision["faiss_weight"]
        
        # Run retrieval with dynamic weights + formula merge
        results = self.retrieve_with_formulas(query_tokens, query_embedding, top_k)
        
        # Restore original weights (for safety)
        self.bm25_weight = original_bm25_weight
        self.faiss_weight = original_faiss_weight
        
        return {
            "results": results,
            "router_decision": router_decision
        }
    
    def retrieve_text(
        self,
        query_tokens: List[str],
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        SR1 — BM25 + FAISS on text chunks only. Returns top_k text results.

        Uses self.bm25_weight and self.faiss_weight (set by router at call time).
        Formula index is not touched.
        """
        results = self.retrieve(query_tokens, query_embedding, top_k=top_k)
        for item in results:
            item.setdefault("source", "text_index")
        logger.debug(f"[SR1] Text retrieval: {len(results)} results")
        return results

    def retrieve_formulas(
        self,
        query_tokens: List[str],
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        SR2 — BM25 + FAISS on formula chunks only. Returns top_k formula results.

        Uses self.bm25_weight and self.faiss_weight (set by router at call time).
        Text index is not touched. Includes same-page co-retrieval so full equation
        systems surface together.
        """
        formula_results = []
        formula_top_k = 20  # wider candidate pool for scoring

        if self.formula_faiss is not None or self.formula_bm25 is not None:
            # FAISS scores
            faiss_scores: Dict[str, float] = {}
            if self.formula_faiss is not None:
                for item in self.formula_faiss.retrieve(query_embedding, top_k=formula_top_k):
                    faiss_scores[item["chunk"]["chunk_id"]] = float(item["score"])

            # BM25 scores (lexical match on embedding_text tokens)
            bm25_scores: Dict[str, float] = {}
            if self.formula_bm25 is not None:
                for doc, score in self.formula_bm25.retrieve(query_tokens, top_k=formula_top_k):
                    bm25_scores[doc["chunk_id"]] = float(score)

            # Normalise each to [0, 1]
            if bm25_scores:
                bmax, bmin = max(bm25_scores.values()), min(bm25_scores.values())
                denom = (bmax - bmin) if bmax != bmin else 1.0
                bm25_scores = {k: (v - bmin) / denom for k, v in bm25_scores.items()}

            if faiss_scores:
                fmax, fmin = max(faiss_scores.values()), min(faiss_scores.values())
                denom = (fmax - fmin) if fmax != fmin else 1.0
                faiss_scores = {k: (v - fmin) / denom for k, v in faiss_scores.items()}

            # Build chunk lookup and combine scores
            formula_lookup = {c["chunk_id"]: c for c in self.formula_chunks}
            all_ids = set(faiss_scores) | set(bm25_scores)

            for cid in all_ids:
                b = bm25_scores.get(cid, 0.0)
                f = faiss_scores.get(cid, 0.0)
                combined = self.bm25_weight * b + self.faiss_weight * f
                if combined >= self.formula_threshold and cid in formula_lookup:
                    formula_results.append({
                        "chunk": formula_lookup[cid],
                        "score": combined * self.formula_weight,
                        "bm25_score": b,
                        "faiss_score": f,
                        "source": "formula_index",
                    })

            formula_results.sort(key=lambda x: x["score"], reverse=True)
            formula_results = formula_results[:formula_top_k]

            # Same-page co-retrieval: surface sibling formulas from the same page
            # so full equation systems (e.g. the three Nosé–Hoover ODEs) appear together.
            CO_RETRIEVAL_ANCHOR_THRESHOLD = 0.25
            CO_RETRIEVAL_CAP_PER_PAGE = 5
            anchor_pages: Dict[tuple, float] = {}
            for item in formula_results:
                if item["score"] >= CO_RETRIEVAL_ANCHOR_THRESHOLD:
                    key = (item["chunk"].get("doc_id", ""), item["chunk"].get("page_number", -1))
                    if key[1] >= 0:
                        anchor_pages[key] = max(anchor_pages.get(key, 0.0), item["score"])

            if anchor_pages:
                seen_co = {item["chunk"]["chunk_id"] for item in formula_results}
                for key, anchor_score in anchor_pages.items():
                    doc_id, page = key
                    siblings = [
                        c for c in self.formula_chunks
                        if c.get("doc_id") == doc_id
                        and c.get("page_number") == page
                        and c["chunk_id"] not in seen_co
                    ]
                    siblings.sort(key=lambda c: c.get("quality_score", 0.0), reverse=True)
                    for sib in siblings[:CO_RETRIEVAL_CAP_PER_PAGE]:
                        formula_results.append({
                            "chunk": sib,
                            "score": anchor_score * 0.85,
                            "bm25_score": 0.0,
                            "faiss_score": 0.0,
                            "source": "formula_index",
                        })
                        seen_co.add(sib["chunk_id"])

                formula_results.sort(key=lambda x: x["score"], reverse=True)

            if formula_results:
                logger.info(f"[SR2] Formula retrieval: {len(formula_results)} formulas above threshold ({self.formula_threshold})")

        return formula_results[:top_k]

    def retrieve_with_formulas(
        self,
        query_tokens: List[str],
        query_embedding: List[float],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve text chunks + formula chunks.

        Thin wrapper over retrieve_text (SR1) + retrieve_formulas (SR2).
        Returns 10 text + 10 formula results (20 total) regardless of top_k,
        so the DD agent always sees a balanced evidence set.
        """
        text_results = self.retrieve_text(query_tokens, query_embedding, top_k=10)
        formula_results = self.retrieve_formulas(query_tokens, query_embedding, top_k=10)
        return text_results + formula_results

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
