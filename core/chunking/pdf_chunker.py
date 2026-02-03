from typing import List, Dict, Any, Union
from core.ingestion.document_schema import Document
from core.nlp.tokenizer import tokenize
from core.nlp.metadata_extractor import generate_chunk_metadata


def make_chunks_from_doc(doc: Union[Document, Dict[str, Any]], max_chars: int = 900, overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Create retrieval chunks from a normalized PresciSE Document (Pydantic model or dict).

    MVP strategy:
    - Iterate sections
    - Split section text into smaller pieces (by character count)
    - Attach metadata (doc_id, section, pages, keywords)
    - Use overlapping chunks to preserve context across boundaries
    
    Args:
        doc: Document to chunk (Pydantic model or dict)
        max_chars: Maximum characters per chunk (default: 900)
        overlap: Overlap between consecutive chunks in characters (default: 150)
    """
    chunks = []
    
    # Handle both Pydantic Document objects and dicts
    if isinstance(doc, Document):
        doc_id = doc.document_id
        sections = doc.sections
    else:
        doc_id = doc["document_id"]
        sections = doc.get("sections", [])

    for i, section in enumerate(sections):
        # Handle both Pydantic Section objects and dicts
        if hasattr(section, 'section_type'):
            section_type = section.section_type
            text = (section.content or "").strip()
            pages = section.page_numbers
        else:
            section_type = section.get("section_type", "unknown")
            text = (section.get("content") or "").strip()
            pages = section.get("page_numbers", [])

        if not text:
            continue

        start = 0
        part = 0

        while start < len(text):
            # Extract chunk with max_chars length
            piece = text[start:start + max_chars].strip()
            if not piece:
                break

            chunk_id = f"{doc_id}_{section_type}_{i}_{part}"
            
            # Generate keyword metadata for fast search
            keyword_metadata = generate_chunk_metadata(piece)

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": piece,
                "tokens": tokenize(piece),
                "metadata": {
                    "section_type": section_type,
                    "pages": pages,
                    **keyword_metadata  # Add keywords from TF-IDF + NER
                }
            })

            # Move forward by (max_chars - overlap) to create overlapping chunks
            # This ensures consecutive chunks share 'overlap' characters
            start += max_chars - overlap
            part += 1

    return chunks

