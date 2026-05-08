"""Get the FULL traceback from Docling to see exactly where rt_detr_v2 fails."""
import sys, traceback
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

print("=== DOCLING rt_detr_v2 DIAGNOSTIC ===\n")

# Step 1: Check model auto-mapping
print("[1] Checking transformers auto-mapping...")
from transformers.models.auto.configuration_auto import MODEL_NAMES_MAPPING
print(f"    rt_detr_v2 in MODEL_NAMES_MAPPING: {'rt_detr_v2' in MODEL_NAMES_MAPPING}")
print(f"    rt_detr in MODEL_NAMES_MAPPING: {'rt_detr' in MODEL_NAMES_MAPPING}")
detr_keys = [k for k in MODEL_NAMES_MAPPING if 'detr' in k]
print(f"    DETR related keys: {detr_keys}")

# Step 2: Check AutoConfig mapping
print("\n[2] Checking AutoConfig._model_mapping...")
from transformers import AutoConfig
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    print(f"    rt_detr_v2 in CONFIG_MAPPING_NAMES: {'rt_detr_v2' in CONFIG_MAPPING_NAMES}")
    detr_configs = {k: v for k, v in CONFIG_MAPPING_NAMES.items() if 'detr' in k.lower()}
    print(f"    DETR configs: {detr_configs}")
except Exception as e:
    print(f"    Error: {e}")

# Step 3: Try loading the exact model Docling uses
print("\n[3] Attempting to load Docling's layout model...")
try:
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    print(f"    StandardPdfPipeline imported OK")
except Exception as e:
    print(f"    Error importing pipeline: {e}")

# Step 4: Try actual PDF conversion with FULL traceback
print("\n[4] Attempting PDF conversion (Verlet.pdf)...")
try:
    from core.ingestion.docling_loader import load_with_docling
    doc = load_with_docling("data/pdfs/Verlet.pdf")
    print(f"    ✅ SUCCESS! Sections: {len(doc.sections) if hasattr(doc, 'sections') else 'N/A'}")
except Exception as e:
    print(f"\n    ❌ FAILED with: {type(e).__name__}: {e}")
    print("\n    FULL TRACEBACK:")
    traceback.print_exc()

print("\n=== DIAGNOSTIC COMPLETE ===")
