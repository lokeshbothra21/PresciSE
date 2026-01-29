from typing import List, Dict
from core.retrieval.bm25_index import BM25Index

class BM25Retriever:
    def __init__(self, documents: List[Dict]):
        """
        documents: List of dicts with keys:
        - chunk_id
        - tokens
        - text
        """
        self.documents = documents
        corpus_tokens = [doc["tokens"] for doc in documents]
        self.index = BM25Index(corpus_tokens)

    def retrieve(self, query_tokens: List[str], top_k: int = 5):
        scores = self.index.score(query_tokens)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
