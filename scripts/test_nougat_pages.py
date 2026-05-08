"""Test Nougat on formula-rich pages to validate formula extraction."""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import sys, json, torch, io, fitz, re
sys.path.insert(0, '.')
from PIL import Image
from nougat.model import NougatConfig, NougatModel
from safetensors.torch import load_file
from core.formula.nougat_scanner import NougatFormulaScanner

MODEL_PATH = 'data/models/nougat-small'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[1] Device: {device}")

with open(f'{MODEL_PATH}/config.json') as f:
    raw_cfg = json.load(f)
dec_layers = raw_cfg.get('decoder', {}).get('decoder_layers', 4)
cfg = NougatConfig(decoder_layer=dec_layers, name_or_path=MODEL_PATH)

print(f"[2] Building NougatModel (decoder_layer={dec_layers})...")
model = NougatModel(cfg).to(device).eval()

raw = load_file(f'{MODEL_PATH}/model.safetensors')
conv = NougatFormulaScanner._convert_hf_checkpoint(raw)
ms = model.state_dict()
pos_key = 'decoder.model.model.decoder.embed_positions.weight'
pos_ckpt = conv.pop(pos_key, None)
filtered = {k: v for k, v in conv.items()
            if k not in ms or ms[k].shape == v.shape}
missing, _ = model.load_state_dict(filtered, strict=False)
# Interpolate positional embedding
if pos_ckpt is not None:
    tgt = ms[pos_key].shape[0]
    resized = model.decoder.resize_bart_abs_pos_emb(pos_ckpt, tgt)
    model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(resized.to(device))
    print(f"[2] Pos embedding resized {pos_ckpt.shape[0]} to {tgt}")
print(f"[2] Weights loaded — missing: {len(missing)} (attn_mask buffers, expected)")

_orig = model.decoder.prepare_inputs_for_inference
def _p(input_ids, past_key_values=None, attention_mask=None,
       use_cache=None, encoder_outputs=None, **_kw):
    return _orig(input_ids, past_key_values=past_key_values,
                 attention_mask=attention_mask, use_cache=use_cache,
                 encoder_outputs=encoder_outputs)
model.decoder.model.prepare_inputs_for_generation = _p
print("[2] compat-patch applied")

DISPLAY_RE = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
INLINE_RE = re.compile(r'\\\((.+?)\\\)', re.DOTALL)

test_pages = [
    ('Hoover2024_NoseHoover_canonical_temperature_control.pdf', 3),
    ('Hoover2024_NoseHoover_canonical_temperature_control.pdf', 4),
    ('Hoover2024_NoseHoover_canonical_temperature_control.pdf', 5),
    ('Verlet.pdf', 1),
    ('Verlet.pdf', 2),
    ('a-unified-formulation-of-the-constant-temperature-molecular-dn26c8wsl1.pdf', 2),
]

print(f"\n[3] Scanning {len(test_pages)} pages...\n")
for pdf_name, page_num in test_pages:
    pdf_path = f'data/pdfs/{pdf_name}'
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        print(f"  {pdf_name} p{page_num}: page out of range")
        continue
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    doc.close()
    img = Image.open(io.BytesIO(pix.tobytes('png')))

    with torch.no_grad():
        out = model.inference(image=img, early_stopping=True)
    mmd = out['predictions'][0] if out.get('predictions') else ''
    formulas = list(DISPLAY_RE.findall(mmd)) + list(INLINE_RE.findall(mmd))
    preview = mmd[:100].replace('\n', ' ').strip()

    print(f"PDF: {pdf_name}  Page {page_num}")
    print(f"  Output: {len(mmd)} chars | {len(formulas)} formulas | {preview!r}")
    for fi, formula in enumerate(formulas[:5]):
        print(f"  [formula {fi}] {formula[:100]}")
    print()

print("[4] Done.")
