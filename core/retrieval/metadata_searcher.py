"""
Fast metadata-based keyword search.

Uses existing keyword extraction from core.nlp.keyword_extraction
for consistency and reusability.
"""

from typing import List, Dict
from core.nlp.keyword_extraction import extract_keywords


def search_by_keywords(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Fast keyword-based search on chunk metadata.
    
    Uses existing keyword_extraction.py for query processing.
    
    Args:
        query: User query string
        chunks: List of chunks with metadata
        top_k: Number of top results to return
        
    Returns:
        List of chunks ranked by keyword overlap score
    """
    # Use existing keyword extraction for query
    query_keywords = set(kw.lower() for kw in extract_keywords(query))
    
    if not query_keywords:
        return []
    
    matches = []
    for chunk in chunks:
        # Get chunk keywords from metadata
        chunk_keywords = chunk.get("metadata", {}).get("keywords", [])
        if not chunk_keywords:
            continue
        
        # Normalize to lowercase and flatten to individual tokens
        chunk_kw_tokens = set()
        for kw in chunk_keywords:
            # Split phrases into tokens
            chunk_kw_tokens.update(kw.lower().split())
        
        # Calculate overlap score
        overlap = len(query_keywords & chunk_kw_tokens)
        
        if overlap > 0:
            # Normalize by query size (Jaccard-like similarity)
            score = overlap / len(query_keywords)
            matches.append({
                "chunk": chunk,
                "metadata_score": score,
                "matched_keywords": list(query_keywords & chunk_kw_tokens)
            })
    
    # Sort by overlap score descending
    matches.sort(key=lambda x: x["metadata_score"], reverse=True)
    
    return matches[:top_k]
