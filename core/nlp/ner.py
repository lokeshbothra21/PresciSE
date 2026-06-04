import threading
import spacy
from typing import List, Dict

_nlp = None
# Guards both the lazy load and inference: a spaCy pipeline is not safe to
# call concurrently from multiple threads on the same object.
_lock = threading.Lock()

def _load_model():
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                _nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Dict]:
    """
    Extract named entities from scientific queries.
    """
    _load_model()
    with _lock:
        doc = _nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
    return entities
