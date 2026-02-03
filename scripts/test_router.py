"""
Test script for Search Router.

Tests Qwen2.5 router with various query types to validate weight decisions.
"""

from core.retrieval.search_router import SearchRouter


def test_router():
    """Test router with different query intents."""
    print("="*70)
    print("SEARCH ROUTER TEST")
    print("="*70)
    
    # Initialize router
    router = SearchRouter()
    
    # Test queries with different intents
    test_queries = [
        # Exact match queries (should favor BM25)
        "What is the definition of molecular dynamics?",
        "MD simulation acronym meaning",
        
        # Semantic queries (should favor FAISS)
        "How do proteins fold in cells?",
        "Explain the relationship between entropy and protein stability",
        
        # Comparison queries (should favor FAISS)
        "Compare classical MD with quantum MD simulations",
        "Difference between explicit and implicit solvent models",
        
        # Methodology queries (should favor FAISS)
        "How does Langevin dynamics work?",
        "Describe the process of protein aggregation",
        
        # Definition lookup (should be balanced)
        "What is free energy calculation?",
        "What are force fields in molecular simulations?"
    ]
    
    print("\nTesting with various query types:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(test_queries)}] Query: \"{query}\"")
        print(f"{'='*70}")
        
        # Get router decision
        decision = router.route(query)
        
        # Display results
        print(f"\n  Intent:       {decision['intent']}")
        print(f"  BM25 Weight:  {decision['bm25_weight']:.3f}")
        print(f"  FAISS Weight: {decision['faiss_weight']:.3f}")
        print(f"  Reason:       {decision['reason']}")
        
        # Validate
        assert 0.0 <= decision['bm25_weight'] <= 1.0
        assert 0.0 <= decision['faiss_weight'] <= 1.0
        assert abs(decision['bm25_weight'] + decision['faiss_weight'] - 1.0) < 0.01
        print(f"\n  ✅ Weights valid (sum={decision['bm25_weight'] + decision['faiss_weight']:.3f})")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_router()
