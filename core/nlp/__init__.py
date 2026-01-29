from core.nlp.tokenizer import tokenize
from core.nlp.lemmatizer import lemmatize
from core.nlp.ner import extract_entities
from core.nlp.keyword_extraction import extract_keywords
from core.nlp.query_classifier import classify_query

def process_query(text: str) -> dict:
    tokens = tokenize(text)
    lemmas = lemmatize(tokens)
    entities = extract_entities(text)
    keywords = extract_keywords(text)
    query_type = classify_query(text)

    return {
        "original_text": text,
        "tokens": tokens,
        "lemmas": lemmas,
        "entities": entities,
        "keywords": keywords,
        "query_type": query_type
    }
