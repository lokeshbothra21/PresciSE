"""
LangChain tools for query enhancement.

Provides classification and subquery generation capabilities.
"""

import json
from typing import List
from langchain_core.tools import tool
from core.llm import GeminiClient
from core.agent.enhancement_prompts import (
    CLASSIFICATION_PROMPT,
    EXPLORATORY_GENERATION_PROMPT,
    COMPARATIVE_GENERATION_PROMPT
)


# Initialize LLM for enhancement (using same Gemini client)
_enhancement_llm = None

def get_enhancement_llm():
    """Get or create enhancement LLM instance."""
    global _enhancement_llm
    if _enhancement_llm is None:
        _enhancement_llm = GeminiClient(temperature=0.3)  # Slightly higher for creativity
    return _enhancement_llm


@tool
def classify_query_type(query: str) -> str:
    """
    Classify query as 'exploratory' or 'comparative'.
    
    EXPLORATORY: Broad, context-aware queries needing comprehensive coverage.
    Examples: "What makes X special?", "How does X work?"
    
    COMPARATIVE: Queries with multiple scenarios, exceptions, or conditions.
    Examples: "What are exceptions to X?", "How does X behave under Y?"
    
    Args:
        query: The user's question
        
    Returns:
        "exploratory" or "comparative"
    """
    llm = get_enhancement_llm()
    prompt = CLASSIFICATION_PROMPT.format(query=query)
    
    response = llm.generate(prompt).strip().lower()
    
    # Ensure valid response
    if "comparative" in response:
        return "comparative"
    else:
        return "exploratory"  # Default to exploratory


@tool
def generate_exploratory_subqueries(query: str) -> List[str]:
    """
    Generate 3-5 contextual subqueries for an EXPLORATORY query.
    
    Subqueries explore different aspects, facets, or related information
    to provide comprehensive coverage of the topic.
    
    Args:
        query: The main user query
        
    Returns:
        List of 3-5 subquery strings
    """
    llm = get_enhancement_llm()
    prompt = EXPLORATORY_GENERATION_PROMPT.format(query=query)
    
    response = llm.generate(prompt).strip()
    
    # Debug logging
    print(f"\n[DEBUG] EXPLORATORY Generation Response:")
    print(f"{response[:200]}..." if len(response) > 200 else response)
    print()
    
    # Parse JSON response with regex
    import re
    try:
        # Try to find JSON array using regex (more robust)
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            json_str = json_match.group()
            subqueries = json.loads(json_str)
            
            # Validate
            if isinstance(subqueries, list) and len(subqueries) > 0:
                # Ensure all items are strings
                subqueries = [str(sq) for sq in subqueries if sq]
                # Limit to 3-5
                if len(subqueries) > 5:
                    subqueries = subqueries[:5]
                print(f"[DEBUG] Successfully parsed {len(subqueries)} subqueries\n")
                return subqueries
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parsing failed: {e}")
    except Exception as e:
        print(f"[WARNING] Unexpected error: {e}")
    
    # Fallback: Return empty list if parsing fails
    print(f"[WARNING] Failed to parse subqueries. Full response:\n{response}\n")
    return []


@tool
def generate_comparative_subqueries(query: str) -> List[str]:
    """
    Generate 3-4 case-specific subqueries for a COMPARATIVE query.
    
    Each subquery represents a different scenario, condition, exception,
    or perspective that needs synthesis.
    
    Args:
        query: The main user query
        
    Returns:
        List of 3-4 subquery strings
    """
    llm = get_enhancement_llm()
    prompt = COMPARATIVE_GENERATION_PROMPT.format(query=query)
    
    response = llm.generate(prompt).strip()
    
    # Debug logging
    print(f"\n[DEBUG] COMPARATIVE Generation Response:")
    print(f"{response[:200]}..." if len(response) > 200 else response)
    print()
    
    # Parse JSON response with regex
    import re
    try:
        # Try to find JSON array using regex (more robust)
        json_match = re.search(r'\[[\s\S]*?\]', response)
        if json_match:
            json_str = json_match.group()
            subqueries = json.loads(json_str)
            
            # Validate
            if isinstance(subqueries, list) and len(subqueries) > 0:
                # Ensure all items are strings
                subqueries = [str(sq) for sq in subqueries if sq]
                # Limit to 3-4
                if len(subqueries) > 4:
                    subqueries = subqueries[:4]
                print(f"[DEBUG] Successfully parsed {len(subqueries)} subqueries\n")
                return subqueries
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parsing failed: {e}")
    except Exception as e:
        print(f"[WARNING] Unexpected error: {e}")
    
    # Fallback: Return empty list if parsing fails
    print(f"[WARNING] Failed to parse subqueries. Full response:\n{response}\n")
    return []


# Export all tools for agent creation
ENHANCEMENT_TOOLS = [
    classify_query_type,
    generate_exploratory_subqueries,
    generate_comparative_subqueries
]
