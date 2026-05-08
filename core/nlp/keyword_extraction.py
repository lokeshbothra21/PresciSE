from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20
)

def extract_keywords(text: str) -> List[str]:
    """
    Extract key terms for lexical retrieval.
    """
    tfidf = _vectorizer.fit_transform([text])
    return _vectorizer.get_feature_names_out().tolist()
