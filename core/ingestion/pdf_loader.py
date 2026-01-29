import os
from core.ingestion.docling_loader import load_with_docling
from core.ingestion.normalizer import normalize_docling_output


def load_pdf_as_document(pdf_path: str):
    """
    Converts a PDF into PresciSE normalized document format.
    """
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]

    docling_result = load_with_docling(pdf_path)
    normalized = normalize_docling_output(docling_result, document_id=document_id)

    # convert Pydantic model to dict for easy processing
    return normalized
