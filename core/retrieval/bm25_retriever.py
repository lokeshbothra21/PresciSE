from typing import List, Dict
import pickle
import os
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
    
    def save_index(self, filepath: str) -> None:
        """
        Save BM25 index to disk using pickle.
        
        Args:
            filepath: Path to save index file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.index, f)
    
    def load_index(self, filepath: str) -> None:
        """
        Load BM25 index from disk.
        
        Args:
            filepath: Path to index file
        """
        with open(filepath, "rb") as f:
            self.index = pickle.load(f)
