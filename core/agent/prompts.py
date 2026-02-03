def build_scientific_prompt(query: str, evidence_blocks: str) -> str:
    """
    Builds a grounded prompt for scientific answering using retrieved evidence.
    """
    return f"""
You are PresciSE: a scientific assistant.

TASK:
Answer the user query using ONLY the evidence provided.
If evidence is insufficient, clearly say so and state what is missing.

USER QUERY:
{query}

EVIDENCE (retrieved chunks):
{evidence_blocks}

INSTRUCTIONS:
- Write a clear, technical explanation.
- Cite sources using the format <doc_id, page X> where doc_id is the document name and X is the page number from the evidence.
- Do not invent facts not present in evidence.
- If multiple sources support a statement, cite all: <doc1, page 7><doc2, page 12>.
""".strip()
