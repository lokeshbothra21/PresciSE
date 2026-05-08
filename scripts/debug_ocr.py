"""
Quick debug script to test OCR extraction directly and see the actual error.
"""
import sys
import os

# Set environment variables
os.environ["PRESCISE_FALLBACK_STAGE_TIMEOUT_SEC"] = "300"
os.environ["PRESCISE_FALLBACK_PREFER_GPU"] = "0"
os.environ["PRESCISE_FORMULA_QUALITY_MODE"] = "balanced"

# Test OCR loading
from core.ingestion.pdf_loader import load_pdf_as_document
from core.formula.formula_chunker import extract_formulas_from_doc

# Pick a test PDF
pdf_path = "data/pdfs/2010-05-21_Hoover1985.pdf"

print("=" * 60)
print("Testing OCR Extraction Directly")
print("=" * 60)

print("\n1. Testing standard extraction...")
try:
    doc_standard = load_pdf_as_document(pdf_path, mode="standard", prefer_gpu=False)
    formulas_standard = extract_formulas_from_doc(doc_standard, extraction_source="standard")
    print(f"   ✅ Standard extraction: {len(formulas_standard)} formulas")
except Exception as e:
    print(f"   ❌ Standard extraction failed: {e}")

print("\n2. Testing OCR extraction...")
try:
    doc_ocr = load_pdf_as_document(pdf_path, mode="ocr", prefer_gpu=False)
    formulas_ocr = extract_formulas_from_doc(doc_ocr, extraction_source="ocr_fallback")
    print(f"   ✅ OCR extraction: {len(formulas_ocr)} formulas")
    if formulas_ocr:
        print("\n   Sample OCR formulas:")
        for i, f in enumerate(formulas_ocr[:3], 1):
            print(f"   [{i}] {f.get('normalized_formula', f.get('formula_text', 'N/A'))}")
except Exception as e:
    print(f"   ❌ OCR extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
