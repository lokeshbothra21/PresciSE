import os
from core.ingestion.docling_loader import (
    load_with_docling,
    load_with_docling_ocr_aggressive,
    load_with_docling_vlm,
)
from core.ingestion.normalizer import normalize_docling_output


def load_pdf_as_document(
    pdf_path: str,
    use_vlm_fallback: bool = False,
    mode: str = "standard",
    prefer_gpu: bool = True,
):
    """
    Converts a PDF into PresciSE normalized document format.
    """
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]

    selected_mode = (mode or "standard").strip().lower()
    if use_vlm_fallback:
        selected_mode = "vlm"

    if selected_mode == "vlm":
        docling_result = load_with_docling_vlm(pdf_path, prefer_gpu=prefer_gpu)
    elif selected_mode == "ocr":
        docling_result = load_with_docling_ocr_aggressive(pdf_path, prefer_gpu=prefer_gpu)
    else:
        docling_result = load_with_docling(pdf_path, prefer_gpu=prefer_gpu)
    normalized = normalize_docling_output(docling_result, document_id=document_id)

    # convert Pydantic model to dict for easy processing
    return normalized
