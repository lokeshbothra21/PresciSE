from typing import List
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, corpus_tokens: List[List[str]]):
        # Guard against empty corpus which causes ZeroDivisionError in BM25Okapi
        if not corpus_tokens or len(corpus_tokens) == 0:
            raise ValueError(
                "BM25Index cannot be initialized with empty corpus. "
                "Check that PDF processing generated chunks with tokens."
            )
        self.bm25 = BM25Okapi(corpus_tokens)

    def score(self, query_tokens: List[str]):
        return self.bm25.get_scores(query_tokens)
