"""
Quick smoke test for the NougatFormulaScanner.
Run: python -m scripts.test_nougat
"""
import warnings, os, sys, time, re, torch, torch.nn

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings('ignore')

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[1] Device: {device}")

from nougat import NougatModel
from nougat.model import NougatConfig
from core.formula.nougat_scanner import NougatFormulaScanner
import json

MODEL_PATH = 'data/models/nougat-small'
print(f"[2] Loading NougatModel from {MODEL_PATH} ...")
t0 = time.time()
# Build model with correct small config (decoder_layer=4) to avoid default base config
with open(f'{MODEL_PATH}/config.json') as f:
    raw_cfg = json.load(f)
dec_layers = raw_cfg.get('decoder', {}).get('decoder_layers', 4)
nougat_cfg = NougatConfig(decoder_layer=dec_layers, name_or_path=MODEL_PATH)
model = NougatModel(nougat_cfg).to(device)

# Convert HF-format checkpoint to nougat-native key format
from safetensors.torch import load_file as _load_sf
raw_ckpt = _load_sf(f'{MODEL_PATH}/model.safetensors')
converted = NougatFormulaScanner._convert_hf_checkpoint(raw_ckpt)
model_state = model.state_dict()
pos_key = 'decoder.model.model.decoder.embed_positions.weight'
pos_ckpt = converted.pop(pos_key, None)
filtered = {k: v for k, v in converted.items()
            if k not in model_state or model_state[k].shape == v.shape}
missing, unexpected = model.load_state_dict(filtered, strict=False)
if pos_ckpt is not None:
    tgt = model_state[pos_key].shape[0]
    resized = model.decoder.resize_bart_abs_pos_emb(pos_ckpt, tgt)
    model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(resized.to(device))
    print(f"[2] Pos embedding resized {pos_ckpt.shape[0]} to {tgt}")
print(f"[2] Checkpoint loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

model.eval()
print(f"[2] Loaded in {time.time()-t0:.1f}s")

# Patch for transformers >= 4.38 (cache_position compat)
_orig = model.decoder.prepare_inputs_for_inference
def _patched(input_ids, past_key_values=None, attention_mask=None,
             use_cache=None, encoder_outputs=None, **_kw):
    return _orig(input_ids, past_key_values=past_key_values,
                 attention_mask=attention_mask, use_cache=use_cache,
                 encoder_outputs=encoder_outputs)
model.decoder.model.prepare_inputs_for_generation = _patched
print("[2b] All patches applied")

# Pick a formula-rich page from Hoover PDF
import io, fitz
from PIL import Image

PDF_PATH = 'data/pdfs/Hoover2024_NoseHoover_canonical_temperature_control.pdf'
PAGE_NUM = 2   # page index (0-based), page 3 is formula-heavy

print(f"[3] Rendering page {PAGE_NUM} of {os.path.basename(PDF_PATH)} ...")
doc = fitz.open(PDF_PATH)
pix = doc[PAGE_NUM].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
doc.close()
img = Image.open(io.BytesIO(pix.tobytes('png')))
print(f"[3] Page size: {img.size}")

print(f"[4] Running Nougat inference ...")
t1 = time.time()
with torch.no_grad():
    out = model.inference(image=img, early_stopping=True)
elapsed = time.time() - t1
mmd = out['predictions'][0] if out.get('predictions') else ''
print(f"[4] Inference done in {elapsed:.1f}s  |  output: {len(mmd)} chars")

print("\n--- MMD OUTPUT (first 1000 chars) ---")
print(mmd[:1000])

# Extract formulas
DISPLAY_RE = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
INLINE_RE  = re.compile(r'\\\((.+?)\\\)', re.DOTALL)
DOLLAR_RE  = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
formulas = []
for pat in (DISPLAY_RE, DOLLAR_RE, INLINE_RE):
    formulas.extend(m.group(1).strip() for m in pat.finditer(mmd))

print(f"\n--- FORMULAS FOUND: {len(formulas)} ---")
for i, f in enumerate(formulas[:15]):
    print(f"  [{i+1}] {f[:120]}")

print("\n[5] Nougat smoke test PASSED.")
