"""
Query Enhancement Agent using direct tool calling.

This agent:
1. Classifies queries as Type A (exploratory) or Type B (multi-case)
2. Generates appropriate subqueries
3. Returns enhanced query set with weights
"""

from typing import Dict, List, Any
from core.agent.enhancement_tools import (
    classify_query_type,
    generate_exploratory_subqueries,
    generate_comparative_subqueries
)


class QueryEnhancer:
    """
    Simple query enhancer using direct tool calls.
    
    Uses tools to:
    - Classify query type
    - Generate subqueries  
    - Assign weights
    """
    
    def __init__(self):
        """Initialize query enhancer."""
        pass
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance query with subqueries.
        
        Args:
            query: Original user query
            
        Returns:
            {
                "main_query": str,
                "type": "exploratory" or "comparative",
                "subqueries": List[str],
                "weights": {
                    "main_query": 1.0,
                    "subquery_0": 0.7,
                    ...
                },
                "unfulfilled_subqueries": []  # Will be populated after retrieval
            }
        """
        # Classify query type
        query_type = classify_query_type.invoke({"query": query})
        
        # Generate subqueries based on type
        if query_type == "exploratory":
            subqueries = generate_exploratory_subqueries.invoke({"query": query})
        else:
            subqueries = generate_comparative_subqueries.invoke({"query": query})
        
        # Build weights
        weights = {"main_query": 1.0}
        for i in range(len(subqueries)):
            weights[f"subquery_{i}"] = 0.7
        
        return {
            "main_query": query,
            "type": query_type,
            "subqueries": subqueries,
            "weights": weights,
            "unfulfilled_subqueries": []
        }


# Simpler wrapper function for easy integration
def enhance_query(query: str) -> Dict[str, Any]:
    """
    Enhance a query with subqueries.
    
    Args:
        query: Original user query
        
    Returns:
        Enhanced query dict with main query, subqueries, and weights
    """
    enhancer = QueryEnhancer()
    return enhancer.enhance_query(query)
