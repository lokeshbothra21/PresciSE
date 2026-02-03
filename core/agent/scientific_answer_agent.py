from typing import List, Dict, Any
from core.llm import GeminiClient
from core.agent.prompts import build_scientific_prompt


class ScientificAnswerAgent:
    """
    Single-agent workflow:
    Query -> Evidence formatting -> Gemini answer
    """

    def __init__(self, llm: GeminiClient):
        self.llm = llm

    def _format_evidence(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for item in retrieved_chunks:
            chunk = item["chunk"]
            # Extract doc_id and page numbers from chunk
            doc_id = chunk.get("doc_id", "unknown")
            pages = chunk.get("metadata", {}).get("pages", [])
            # Use first page number if available
            page_num = pages[0] if pages and pages[0] > 0 else "unknown"
            citation = f"[{doc_id}, page {page_num}]"
            blocks.append(f"{citation} {chunk['text']}")
        return "\n\n".join(blocks)

    def answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        evidence_text = self._format_evidence(retrieved_chunks)
        prompt = build_scientific_prompt(query=query, evidence_blocks=evidence_text)
        return self.llm.generate(prompt)
