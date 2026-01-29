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
            blocks.append(f"[{chunk['chunk_id']}] {chunk['text']}")
        return "\n\n".join(blocks)

    def answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        evidence_text = self._format_evidence(retrieved_chunks)
        prompt = build_scientific_prompt(query=query, evidence_blocks=evidence_text)
        return self.llm.generate(prompt)
