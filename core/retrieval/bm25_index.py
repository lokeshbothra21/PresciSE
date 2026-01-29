from typing import List
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, corpus_tokens: List[List[str]]):
        self.bm25 = BM25Okapi(corpus_tokens)

    def score(self, query_tokens: List[str]):
        return self.bm25.get_scores(query_tokens)
