from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


def extract_keywords(text: str) -> List[str]:
    """
    Extract key terms for lexical retrieval.

    Builds a fresh vectorizer per call. The vectorizer is re-fit on each call
    anyway, so a module-level singleton only added shared mutable state that
    was unsafe under concurrent use; a local instance is thread-safe by
    construction with identical behaviour.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20,
    )
    vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out().tolist()
