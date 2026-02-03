"""
Standard output formatter for PresciSE query results.

Provides consistent, readable formatting for retrieved evidence and answers.
"""

from typing import List, Dict, Any


def print_retrieved_evidence(retrieved: List[Dict[str, Any]], top_k: int = 6) -> None:
    """
    Print retrieved evidence in standard format.
    
    Format:
        [1] Score: 0.6000 | Section: text | Pages: [6] | Doc: doc_name
            Preview: First 100 characters...
    
    Args:
        retrieved: List of retrieved chunks with scores
        top_k: Number of results to display
    """
    print("Top retrieved evidence:")
    for i, item in enumerate(retrieved[:top_k], 1):
        chunk = item["chunk"]
        score = item["score"]
        doc_id = chunk.get("doc_id", "unknown")
        section_type = chunk.get("metadata", {}).get("section_type", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        
        print(f"  [{i}] Score: {score:.4f} | Section: {section_type} | Pages: {pages} | Doc: {doc_id}")
        print(f"      Preview: {text_preview}")
    print()


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Header text
        width: Total width of header line
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width + "\n")


def print_final_answer(answer: str, width: int = 60) -> None:
    """
    Print the final answer with formatting.
    
    Args:
        answer: The generated answer text
        width: Width for separator lines
    """
    print_section_header("FINAL ANSWER", width)
    print(answer)
    print("\n" + "=" * width + "\n")


def format_query_session(query: str, retrieved: List[Dict[str, Any]], answer: str, top_k: int = 6) -> str:
    """
    Format a complete query session as a string (for logging/saving).
    
    Args:
        query: The user query
        retrieved: Retrieved evidence chunks
        answer: Generated answer
        top_k: Number of evidence items to include
        
    Returns:
        Formatted string with entire session
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"QUERY: {query}")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append("Top retrieved evidence:")
    for i, item in enumerate(retrieved[:top_k], 1):
        chunk = item["chunk"]
        score = item["score"]
        doc_id = chunk.get("doc_id", "unknown")
        section_type = chunk.get("metadata", {}).get("section_type", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        text_preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        
        lines.append(f"  [{i}] Score: {score:.4f} | Section: {section_type} | Pages: {pages} | Doc: {doc_id}")
        lines.append(f"      Preview: {text_preview}")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("FINAL ANSWER")
    lines.append("=" * 60)
    lines.append("")
    lines.append(answer)
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
