"""
LangChain-based prompt templates for PresciSE scientific answering.

Replaces manual f-string prompts with LangChain PromptTemplate for:
- Reusable templates
- Input validation
- Easier composition
- Better maintainability
"""

from langchain_core.prompts import PromptTemplate


# Main scientific answering prompt template (standard mode)
SCIENTIFIC_ANSWER_TEMPLATE = """You are PresciSE: a scientific assistant.

TASK:
Answer the user query using ONLY the evidence provided.
If evidence is insufficient, clearly say so and state what is missing.

USER QUERY:
{query}

EVIDENCE (retrieved chunks):
{evidence}

INSTRUCTIONS:
- Write a clear, technical explanation.
- Cite sources using the format <doc_id, page X> where doc_id is the document name and X is the page number from the evidence.
- Do not invent facts not present in evidence.
- If multiple sources support a statement, cite all: <doc1, page 7><doc2, page 12>.
"""

# Create LangChain PromptTemplate
scientific_prompt_template = PromptTemplate(
    input_variables=["query", "evidence"],
    template=SCIENTIFIC_ANSWER_TEMPLATE.strip()
)


def build_scientific_prompt(query: str, evidence_blocks: str) -> str:
    """
    Builds a grounded prompt for scientific answering using retrieved evidence.
    
    This is a backward-compatible wrapper function that uses LangChain PromptTemplate
    internally while maintaining the same signature as the original implementation.
    
    Args:
        query: User's question
        evidence_blocks: Formatted evidence text with citations
        
    Returns:
        Formatted prompt string ready for LLM
    """
    return scientific_prompt_template.format(
        query=query,
        evidence=evidence_blocks
    )


def build_enhanced_prompt(
    main_query: str,
    subqueries: list,
    evidence_blocks: str,
    enhancement_type: str,
    unfulfilled_subqueries: list = None
) -> str:
    """
    Build prompt for enhanced queries with subqueries.
    
    Args:
        main_query: Original user query
        subqueries: List of generated subqueries
        evidence_blocks: Formatted evidence from all queries
        enhancement_type: "exploratory" or "comparative"
        unfulfilled_subqueries: Subqueries that yielded no results
        
    Returns:
        Formatted prompt for LLM
    """
    # Import here to avoid circular dependency
    from core.agent.enhancement_prompts import EXPLORATORY_ANSWER_PROMPT, COMPARATIVE_ANSWER_PROMPT
    
    # Format subqueries list
    subqueries_list = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(subqueries)])
    
    # Choose template based on type
    if enhancement_type == "exploratory":
        template = EXPLORATORY_ANSWER_PROMPT
    else:
        template = COMPARATIVE_ANSWER_PROMPT
    
    # Format prompt
    prompt = template.format(
        main_query=main_query,
        subqueries_list=subqueries_list,
        evidence_blocks=evidence_blocks
    )
    
    # Add knowledge gap suggestions if applicable
    if unfulfilled_subqueries and len(unfulfilled_subqueries) > 0:
        gap_suggestions = "\n".join([f"- {sq}" for sq in unfulfilled_subqueries])
        # Note: The template already includes knowledge gap section,
        # but we ensure unfulfilled queries are highlighted
        prompt += f"\n\n**Note**: The following subqueries had no supporting evidence:\n{gap_suggestions}"
    
    return prompt


# Alternative: Get the template object directly for LangChain chains
def get_prompt_template() -> PromptTemplate:
    """
    Returns the LangChain PromptTemplate object for use in chains.
    
    Returns:
        Configured PromptTemplate instance
    """
    return scientific_prompt_template
