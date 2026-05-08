import logging
import os

# ---------------------------------------------------------------------------
# Compatibility shim: timm 0.5.4 (required by nougat-ocr) is missing
# ImageNetInfo, which transformers>=4.46 tries to import via its timm_wrapper
# module.  Adding a stub prevents the lazy-import failure that would otherwise
# crash docling's layout pipeline at first use.
# ---------------------------------------------------------------------------
try:
    import timm.data as _timm_data
    if not hasattr(_timm_data, "ImageNetInfo"):
        class _ImageNetInfoStub:
            """Stub for timm>=0.6 ImageNetInfo (not present in timm 0.5.4)."""
            def __init__(self, *a, **kw): pass
        _timm_data.ImageNetInfo = _ImageNetInfoStub
    if not hasattr(_timm_data, "infer_imagenet_subset"):
        _timm_data.infer_imagenet_subset = lambda *a, **kw: None
except Exception:
    pass

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    FormatOption,
    PdfFormatOption,
)
from docling.exceptions import ConversionError
from docling.pipeline.vlm_pipeline import VlmPipeline

# Suppress noisy logs from docling internals.
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()], force=True)
for logger_name in ["docling", "pdfminer", "pypdf", "pypdfium2", "rapidocr", "RapidOCR"]:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.CRITICAL)
    _logger.addHandler(logging.NullHandler())
    _logger.propagate = False


def _apply_device_policy(prefer_gpu: bool = True) -> None:
    """
    Best-effort device policy for Docling subprocess execution.
    """
    if prefer_gpu:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("DOCLING_DEVICE", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["DOCLING_DEVICE"] = "cpu"


def load_with_docling(pdf_path: str, prefer_gpu: bool = True):
    """
    Extract structured content from a PDF using the standard Docling pipeline.

    Formula/code enrichment is disabled by default for runtime stability.
    """
    try:
        _apply_device_policy(prefer_gpu=prefer_gpu)
        pipeline_options = PdfPipelineOptions()
        enable_formula_enrichment = os.getenv("PRESCISE_ENABLE_FORMULA_ENRICHMENT", "0") == "1"
        enable_code_enrichment = os.getenv("PRESCISE_ENABLE_CODE_ENRICHMENT", "0") == "1"
        pipeline_options.do_formula_enrichment = enable_formula_enrichment
        pipeline_options.do_code_enrichment = enable_code_enrichment

        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(pdf_path)
    except ConversionError:
        filename = os.path.basename(pdf_path)
        raise ConversionError(
            f"Failed to convert PDF: {filename}. "
            "The PDF may be malformed or missing required metadata."
        ) from None

    return result.document


def load_with_docling_ocr_aggressive(pdf_path: str, prefer_gpu: bool = True):
    """
    OCR-focused fallback path for glyph-rendered math where text extraction is weak.
    """
    try:
        _apply_device_policy(prefer_gpu=prefer_gpu)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_formula_enrichment = (
            os.getenv("PRESCISE_ENABLE_FORMULA_ENRICHMENT", "0") == "1"
        )
        pipeline_options.do_code_enrichment = False

        if getattr(pipeline_options, "ocr_options", None) is not None:
            setattr(pipeline_options.ocr_options, "force_full_page_ocr", True)

        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(pdf_path)
    except Exception as e:
        filename = os.path.basename(pdf_path)
        raise ConversionError(
            f"OCR fallback conversion failed for PDF: {filename}. Error: {e}"
        ) from None

    return result.document


def load_with_docling_vlm(pdf_path: str, prefer_gpu: bool = True):
    """
    Extract content from a PDF using Docling VLM pipeline (Granite Docling).

    This is a fallback path for pages/documents with poor formula coverage.
    """
    try:
        _apply_device_policy(prefer_gpu=prefer_gpu)
        vlm_options = VlmPipelineOptions()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_cls=VlmPipeline,
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_options=vlm_options,
                )
            }
        )
        result = converter.convert(pdf_path)
    except Exception as e:
        filename = os.path.basename(pdf_path)
        raise ConversionError(
            f"VLM fallback conversion failed for PDF: {filename}. Error: {e}"
        ) from None

    return result.document
