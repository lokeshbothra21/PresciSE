"""
Metadata extraction for chunk keywords.

Uses TF-IDF + Named Entity Recognition to extract distinctive keywords
without pre-filtering or manual constraints.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import spacy

# Lazy-load spacy model (expensive)
_nlp = None


def get_nlp():
    """Load SpaCy model once and cache it."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    return _nlp


def extract_tfidf_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top keywords using TF-IDF.
    
    Let the algorithm decide which terms are most important
    without manual filtering.
    
    Args:
        text: Chunk text
        top_n: Number of keywords to extract
        
    Returns:
        List of top keywords (can be unigrams or bigrams)
    """
    vectorizer = TfidfVectorizer(
        max_features=top_n,
        stop_words='english',
        ngram_range=(1, 2),  # Allow both single words and phrases
        min_df=1,
        max_df=1.0
    )
    
    try:
        vectorizer.fit([text])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except ValueError:
        # Text too short or all stopwords
        return []


def extract_entities_spacy(text: str, top_n: int = 5) -> List[str]:
    """
    Extract named entities using SpaCy.
    
    Let SpaCy decide what's important - no manual entity type filtering.
    
    Args:
        text: Chunk text
        top_n: Max number of entities to extract
        
    Returns:
        List of entity texts
    """
    nlp = get_nlp()
    
    # Limit text to first 1000 chars for speed
    # (entities are usually frontloaded)
    doc = nlp(text[:1000])
    
    # Extract all entities, let SpaCy decide what's important
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Deduplicate while preserving order
    unique_entities = list(dict.fromkeys(entities))
    
    return unique_entities[:top_n]


def generate_chunk_metadata(text: str, max_keywords: int = 5) -> Dict[str, Any]:
    """
    Generate keyword metadata for a chunk.
    
    Combines TF-IDF (distinctive terms) + NER (important entities)
    without manual filtering or including unnecessary data.
    
    Args:
        text: Chunk text
        max_keywords: Maximum total keywords to extract
        
    Returns:
        {
            "keywords": ["protein folding", "molecular dynamics", ...]
        }
    """
    # Extract keywords from both methods
    tfidf_kw = extract_tfidf_keywords(text, top_n=max_keywords)
    ner_kw = extract_entities_spacy(text, top_n=max_keywords)
    
    # Combine and deduplicate (TF-IDF first for priority)
    all_keywords = tfidf_kw + ner_kw
    unique_keywords = []
    seen = set()
    
    for kw in all_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            unique_keywords.append(kw)
            seen.add(kw_lower)
    
    # Return only top max_keywords
    return {
        "keywords": unique_keywords[:max_keywords]
    }
