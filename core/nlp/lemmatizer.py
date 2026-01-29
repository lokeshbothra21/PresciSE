import spacy
from typing import List

_nlp = None

def _load_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")

def lemmatize(tokens: List[str]) -> List[str]:
    _load_model()
    doc = _nlp(" ".join(tokens))
    return [token.lemma_.lower() for token in doc]
