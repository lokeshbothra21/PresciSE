$ErrorActionPreference = "Stop"
Remove-Item -Path "data/index/*" -Force -Recurse -ErrorAction SilentlyContinue
$env:PRESCISE_ENABLE_FORMULA_ENRICHMENT='1'
$env:PRESCISE_ENABLE_CODE_ENRICHMENT='0'
$env:PRESCISE_ENABLE_OCR_FALLBACK='0'
$env:PRESCISE_ENABLE_VLM_FALLBACK='0'
$env:PRESCISE_FORMULA_QUALITY_MODE='balanced'
python -m scripts.pdf_demo "What is the Lennard-Jones potential?"
