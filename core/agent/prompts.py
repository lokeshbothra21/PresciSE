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
SCIENTIFIC_ANSWER_TEMPLATE = """You are PresciSE, a scientific literature assistant. Your ONLY job is to extract and explain information that is explicitly present in the provided evidence. You do not have independent scientific knowledge for the purposes of this task.

EVIDENCE (retrieved from scientific literature):
{evidence}

USER QUERY:
{query}

STRICT RULES — you must follow every one of these:
1. EVIDENCE ONLY: Base your entire answer exclusively on the text in the EVIDENCE section above. Do not add facts, equations, or explanations from your training knowledge.
2. NO HALLUCINATED FORMULAS: Only reproduce a formula if it appears verbatim in the evidence inside [FORMULA]...[/FORMULA] tags. If a formula is not in the evidence, state explicitly that the retrieved documents do not contain that equation.
3. EXACT REPRODUCTION: When including a formula, copy it character-for-character from the evidence. If the formula is stored as [FORMULA]$...$[/FORMULA], reproduce it on its own line exactly as [FORMULA]$exact LaTeX here$[/FORMULA]. Do not complete, guess, or infer missing characters.
4. FORMULA NOTATION: Formulas in evidence are LaTeX wrapped in $...$. Reproduce them exactly as [FORMULA]$exact LaTeX$[/FORMULA] on their own line. Do NOT use $$...$$, \begin{equation}, or any other math environment.
5. NO CITATIONS IN TEXT: Do not include [doc, page N] tags inline. Citations are shown separately.
6. MISSING INFORMATION: If the evidence does not contain enough to answer the query fully, say clearly what is missing rather than filling the gap with your own knowledge.
7. SINGLE FORMULA: Include AT MOST ONE formula in your answer — the one that most directly and completely answers the question. Do NOT list or compare multiple formulas from the evidence. Exceptions: (a) if the user explicitly asks about different conditions or cases where the equation meaningfully differs per case, you may include a second formula; (b) if the user explicitly asks to "write the equations" or "give the equations" for a system (e.g. a set of coupled ODEs, equations of motion, conservation laws), include ALL equations that form that system — they are a single logical unit, not separate formulas. Even in exception (b), do not add extra formulas beyond the stated system. When in doubt, choose one and omit the rest.
8. EVIDENCE PRIORITY: Evidence blocks are ranked by relevance — [Evidence 1] is the most relevant to the query, higher numbers are less relevant. When choosing which formula or fact to include, prefer the highest-ranked evidence that directly answers the question.

ANSWER:"""

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
