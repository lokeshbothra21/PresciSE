from typing import List, Dict, Any, Union
from core.nlp.tokenizer import tokenize
from core.ingestion.document_schema import Document


def make_chunks_from_doc(doc: Union[Document, Dict[str, Any]], max_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    Create retrieval chunks from a normalized PresciSE Document (Pydantic model or dict).

    MVP strategy:
    - Iterate sections
    - Split section text into smaller pieces (by character count)
    - Attach metadata (doc_id, section, pages)
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
            piece = text[start:start + max_chars].strip()
            if not piece:
                break

            chunk_id = f"{doc_id}_{section_type}_{i}_{part}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": piece,
                "tokens": tokenize(piece),
                "metadata": {
                    "section_type": section_type,
                    "pages": pages,
                }
            })

            start += max_chars
            part += 1

    return chunks
