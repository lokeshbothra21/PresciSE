import spacy
from typing import List, Dict

_nlp = None

def _load_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Dict]:
    """
    Extract named entities from scientific queries.
    """
    _load_model()
    doc = _nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    return entities
