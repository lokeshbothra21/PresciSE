"""
Nougat background formula scanner.

Runs Meta's Nougat model (academic-paper OCR → LaTeX) as a background pass over
each PDF to extract high-quality LaTeX formulas that the regex/PyMuPDF pipeline
missed.  Results are hot-swapped into the live formula index via
HybridRetriever.update_formula_index().

Public API
----------
NougatFormulaScanner(embedder, index_dir)  — main class
_normalize_nougat_latex(latex)             — clean / validate a raw Nougat string
_normalize_for_compare(latex)              — strip noise for substring comparison
_remove_subsets(chunks)                    — drop redundant sub-formulas in a batch
_make_formula_chunk(latex, doc_id, ...)    — build a FormulaChunk-compatible dict
"""

from __future__ import annotations

import json
import re
import time
import torch
import torch.nn
from pathlib import Path
from typing import Dict, List

from loguru import logger


# ---------------------------------------------------------------------------
# Nougat OCR Greek-letter misreading corrections
# ---------------------------------------------------------------------------

# Each entry: (compiled_pattern, replacement, description)
# Only SAFE corrections where the pattern is never valid LaTeX.
_NOUGAT_GREEK_FIXES = [
    (re.compile(r'<\|>'),                        r'\\phi',  'phi misread as <|>'),
    (re.compile(r'<\\?\|\\?>'),                  r'\\phi',  'phi variant'),
    (re.compile(r'\\uparrow\s*rac\b', re.I),     r'\\psi',  'psi misread as uparrow rac'),
    (re.compile(r'\barrow\s*up\s*rac\b', re.I),  r'\\psi',  'psi misread as arrow up rac'),
    (re.compile(r'\\uprac\b', re.I),             r'\\psi',  'psi shorthand misread'),
    # Note: bare "3" → "\beta" is NOT added — too ambiguous with the digit 3.
    # Additional patterns can be appended here as new misreadings are discovered.
]


def _fix_nougat_greek(latex: str) -> str:
    """Apply known Nougat OCR misreadings of Greek letters. Safe patterns only."""
    for pattern, replacement, _ in _NOUGAT_GREEK_FIXES:
        latex = pattern.sub(replacement, latex)
    return latex


# ---------------------------------------------------------------------------
# Normalisation helpers (used by deduplicator too, hence module-level)
# ---------------------------------------------------------------------------

def _normalize_nougat_latex(latex: str) -> str:
    """
    Clean a raw Nougat LaTeX string.
    Returns an empty string if the formula is invalid or trivial after cleaning.
    """
    s = latex.strip()

    # 0. Fix known Nougat Greek-letter misreadings before any other checks
    s = _fix_nougat_greek(s)

    # 1. Remove stray block delimiters Nougat sometimes leaves inside
    s = re.sub(r"^\\\[|\\\]$|^\\\(|\\\)$|^\$\$|\$\$$", "", s).strip()

    # 2. Collapse excessive whitespace
    s = re.sub(r"\s{2,}", " ", s)

    # 3. Reject hallucination: same token repeated 3+ times consecutively
    if re.search(r"(\b\S+\b)(?:\s+\1){2,}", s):
        return ""

    # 4. Reject if braces are unbalanced (sign of truncated output)
    if s.count("{") != s.count("}"):
        return ""

    # 5. Reject trivial: pure integers/floats or single characters
    if re.fullmatch(r"[0-9]+\.?[0-9]*\s*", s) or len(s) <= 1:
        return ""

    # 6. Reject if no letters or LaTeX commands at all (just digits/punctuation)
    if not re.search(r"[A-Za-z\\]", s):
        return ""

    # 7. Cap extremely long outputs (> 500 chars → likely garbled)
    if len(s) > 500:
        return ""

    return s


def _normalize_for_compare(latex: str) -> str:
    """Strip whitespace, braces, and backslashes for substring comparison."""
    return re.sub(r"[\s\\{}]", "", latex).lower()


def _remove_subsets(chunks: List[Dict]) -> List[Dict]:
    """
    Within a list of formula chunks (same doc+page batch):
    if normalised(A) is a proper substring of normalised(B), drop A.
    Keeps the most complete formula when partial duplicates exist.
    """
    normed = [_normalize_for_compare(fc.get("latex_formula", "")) for fc in chunks]
    keep = []
    for i, fc in enumerate(chunks):
        dominated = any(
            i != j
            and normed[i]
            and normed[i] in normed[j]
            and normed[i] != normed[j]
            for j in range(len(chunks))
        )
        if not dominated:
            keep.append(fc)
    return keep


# ---------------------------------------------------------------------------
# Chunk factory
# ---------------------------------------------------------------------------

def _make_formula_chunk(
    latex: str, doc_id: str, page_num: int, idx: int, context_text: str = ""
) -> Dict:
    """Build a FormulaChunk-compatible dict from a Nougat-extracted LaTeX string."""
    from core.nlp.tokenizer import tokenize

    embedding_text = f"The equation is:\n{latex}"
    if context_text:
        embedding_text += f"\nContext: {context_text}"

    return {
        "chunk_id": f"{doc_id}_nougat_p{page_num}_{idx}",
        "chunk_type": "formula",
        "formula_text": latex,
        "normalized_formula": latex,   # Nougat outputs LaTeX directly
        "latex_formula": latex,
        "context_text": context_text,
        "doc_id": doc_id,
        "page_number": page_num,
        "extraction_source": "nougat",
        "quality_score": 0.85,
        "is_trivial": False,
        "text": f"[FORMULA]${latex}$[/FORMULA]",
        "embedding_text": embedding_text,
        "tokens": tokenize(embedding_text),  # includes context for BM25 too
        "embedding": None,                   # filled after embed_texts()
        "metadata": {"pages": [page_num], "section_type": "formula"},
    }


# ---------------------------------------------------------------------------
# Main scanner class
# ---------------------------------------------------------------------------

class NougatFormulaScanner:
    """
    Scans PDFs with Meta's Nougat model to extract high-quality LaTeX formulas.

    Usage
    -----
    scanner = NougatFormulaScanner(embedder, "data/index")
    if scanner.needs_scan(doc_id):
        result = scanner.scan_pdf(pdf_path, doc_id, existing_formula_chunks)
        # result = {"merged": [...], "added": int, "replaced": int,
        #           "pages_scanned": int, "elapsed_s": float}
    """

    MODEL_ID = "facebook/nougat-small"

    # Regex patterns for extracting formulas from .mmd output
    _INLINE_RE  = re.compile(r"\\\((.+?)\\\)",  re.DOTALL)
    _DISPLAY_RE = re.compile(r"\\\[(.+?)\\\]",  re.DOTALL)
    _DOLLAR_RE  = re.compile(r"\$\$(.+?)\$\$",  re.DOTALL)

    def __init__(self, embedder, index_dir: str):
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.scan_status_path = self.index_dir / "vlm_scan_status.json"
        self._model = None   # lazy-loaded on first scan
        self._device = None

    # ------------------------------------------------------------------
    # Scan state tracking
    # ------------------------------------------------------------------

    def needs_scan(self, doc_id: str) -> bool:
        """Return True if this doc has never been scanned by Nougat."""
        return doc_id not in self._load_status()

    def mark_scanned(
        self,
        doc_id: str,
        pages_scanned: int,
        added: int,
        replaced: int,
        elapsed_s: float,
    ) -> None:
        import datetime
        status = self._load_status()
        status[doc_id] = {
            "scanned_at": datetime.datetime.utcnow().isoformat() + "Z",
            "pages_scanned": pages_scanned,
            "formulas_added": added,
            "formulas_replaced": replaced,
            "elapsed_s": round(elapsed_s, 1),
        }
        self.scan_status_path.write_text(json.dumps(status, indent=2))

    def _load_status(self) -> dict:
        if self.scan_status_path.exists():
            try:
                return json.loads(self.scan_status_path.read_text())
            except Exception:
                return {}
        return {}

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # HF-format → nougat-native weight conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_hf_checkpoint(ckpt: dict) -> dict:
        """
        Re-key a HuggingFace-format facebook/nougat-small checkpoint so that
        its weights can be loaded into NougatModel (which uses timm's
        SwinTransformer internally with different key names).

        Mapping highlights
        ------------------
        Encoder (HF SwinTransformer → timm SwinTransformer):
          encoder.encoder.layers.N.blocks.M.attention.self.{query,key,value}
            → encoder.model.layers.N.blocks.M.attn.qkv  (Q/K/V concatenated)
          encoder.encoder.layers.N.blocks.M.attention.output.dense
            → encoder.model.layers.N.blocks.M.attn.proj
          encoder.encoder.layers.N.blocks.M.layernorm_{before,after}
            → encoder.model.layers.N.blocks.M.norm{1,2}
          encoder.encoder.layers.N.blocks.M.intermediate.dense
            → encoder.model.layers.N.blocks.M.mlp.fc1
          encoder.encoder.layers.N.blocks.M.output.dense
            → encoder.model.layers.N.blocks.M.mlp.fc2
          encoder.encoder.layers.N.downsample.*
            → encoder.model.layers.N.downsample.*
          encoder.embeddings.norm.*
            → encoder.model.norm.*
          encoder.embeddings.patch_embeddings.projection.*
            → encoder.model.patch_embed.proj.*

        Decoder (prefix only):
          decoder.model.decoder.*  → decoder.model.model.decoder.*
          decoder.lm_head.*        → decoder.model.lm_head.*
        """
        import torch

        new = {}

        # --- collect per-block Q/K/V for concatenation ---
        qkv_weight_buf: dict = {}   # key → {q,k,v}
        qkv_bias_buf:   dict = {}

        for k, v in ckpt.items():
            # ---- ENCODER ----
            # patch embed
            if k.startswith("encoder.embeddings.patch_embeddings.projection."):
                suffix = k[len("encoder.embeddings.patch_embeddings.projection."):]
                new[f"encoder.model.patch_embed.proj.{suffix}"] = v
                continue

            # patch-embedding layer norm (size = embed_dim, e.g. 128)
            if k.startswith("encoder.embeddings.norm."):
                suffix = k[len("encoder.embeddings.norm."):]
                new[f"encoder.model.patch_embed.norm.{suffix}"] = v
                continue

            # encoder blocks: layernorm
            if ".layernorm_before." in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".layernorm_before.", ".norm1."))
                new[new_k] = v
                continue
            if ".layernorm_after." in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".layernorm_after.", ".norm2."))
                new[new_k] = v
                continue

            # encoder blocks: MLP
            if ".intermediate.dense." in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".intermediate.dense.", ".mlp.fc1."))
                new[new_k] = v
                continue
            if ".output.dense." in k and "attention" not in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".output.dense.", ".mlp.fc2."))
                new[new_k] = v
                continue

            # encoder blocks: attention proj
            if ".attention.output.dense." in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".attention.output.dense.", ".attn.proj."))
                new[new_k] = v
                continue

            # encoder blocks: Q/K/V → buffer for merge
            if ".attention.self.query." in k:
                base = k.replace("encoder.encoder.layers.", "encoder.model.layers.")
                base = re.sub(r"\.attention\.self\.query\.(weight|bias)$", "", base)
                suffix = "weight" if k.endswith(".weight") else "bias"
                buf = qkv_weight_buf if suffix == "weight" else qkv_bias_buf
                buf.setdefault(base, {}); buf[base]["q"] = v
                continue
            if ".attention.self.key." in k:
                base = k.replace("encoder.encoder.layers.", "encoder.model.layers.")
                base = re.sub(r"\.attention\.self\.key\.(weight|bias)$", "", base)
                suffix = "weight" if k.endswith(".weight") else "bias"
                buf = qkv_weight_buf if suffix == "weight" else qkv_bias_buf
                buf.setdefault(base, {}); buf[base]["k"] = v
                continue
            if ".attention.self.value." in k:
                base = k.replace("encoder.encoder.layers.", "encoder.model.layers.")
                base = re.sub(r"\.attention\.self\.value\.(weight|bias)$", "", base)
                suffix = "weight" if k.endswith(".weight") else "bias"
                buf = qkv_weight_buf if suffix == "weight" else qkv_bias_buf
                buf.setdefault(base, {}); buf[base]["v"] = v
                continue

            # encoder blocks: relative position bias / index
            if ".attention.self.relative_position_bias_table" in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".attention.self.relative_position_bias_table",
                             ".attn.relative_position_bias_table"))
                new[new_k] = v
                continue
            if ".attention.self.relative_position_index" in k:
                new_k = (k
                    .replace("encoder.encoder.layers.", "encoder.model.layers.")
                    .replace(".attention.self.relative_position_index",
                             ".attn.relative_position_index"))
                new[new_k] = v
                continue

            # encoder downsample layers
            if k.startswith("encoder.encoder.layers."):
                new_k = k.replace("encoder.encoder.layers.", "encoder.model.layers.")
                new[new_k] = v
                continue

            # ---- DECODER ----
            if k.startswith("decoder.model.decoder."):
                new_k = k.replace("decoder.model.decoder.",
                                  "decoder.model.model.decoder.")
                new[new_k] = v
                continue
            if k.startswith("decoder.lm_head."):
                new_k = k.replace("decoder.lm_head.", "decoder.model.lm_head.")
                new[new_k] = v
                continue

            # pass-through anything else
            new[k] = v

        # --- merge Q/K/V into combined attn.qkv ---
        for base, d in qkv_weight_buf.items():
            if "q" in d and "k" in d and "v" in d:
                new[f"{base}.attn.qkv.weight"] = torch.cat(
                    [d["q"], d["k"], d["v"]], dim=0
                )
        for base, d in qkv_bias_buf.items():
            if "q" in d and "k" in d and "v" in d:
                new[f"{base}.attn.qkv.bias"] = torch.cat(
                    [d["q"], d["k"], d["v"]], dim=0
                )

        return new

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from nougat import NougatModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[VLM SCAN] Loading Nougat ({self.MODEL_ID}) on {self._device}..."
        )
        t0 = time.time()
        # Build model skeleton from a correct NougatConfig.
        # NougatModel.from_pretrained() with HF-format checkpoints uses the
        # wrong NougatConfig defaults (decoder_layer=10 instead of 4 for -small).
        # We read decoder_layer from the checkpoint config directly and create
        # the model manually so it has the right architecture from the start.
        from nougat.model import NougatConfig
        import json as _json
        cfg_path = Path(self.MODEL_ID) / "config.json"
        with open(cfg_path) as _f:
            raw_cfg = _json.load(_f)
        _dec_layers = raw_cfg.get("decoder", {}).get("decoder_layers", 4)
        nougat_cfg = NougatConfig(
            decoder_layer=_dec_layers,
            name_or_path=self.MODEL_ID,  # needed for tokenizer lookup
        )
        self._model = NougatModel(nougat_cfg).to(self._device)

        # Convert HF-format checkpoint → nougat-native keys and reload weights.
        ckpt_path = Path(self.MODEL_ID) / "model.safetensors"
        if ckpt_path.exists():
            from safetensors.torch import load_file as _load_sf
            raw_ckpt = _load_sf(str(ckpt_path))
        else:
            ckpt_path = Path(self.MODEL_ID) / "pytorch_model.bin"
            raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        converted = self._convert_hf_checkpoint(raw_ckpt)

        # Drop shape-mismatching keys except positional embeddings which we
        # handle explicitly via nougat's own interpolation helper.
        model_state = self._model.state_dict()
        pos_emb_key = "decoder.model.model.decoder.embed_positions.weight"
        filtered = {}
        pos_emb_ckpt = None
        for k, v in converted.items():
            if k == pos_emb_key:
                pos_emb_ckpt = v   # handle separately
            elif k not in model_state or model_state[k].shape == v.shape:
                filtered[k] = v

        missing, unexpected = self._model.load_state_dict(filtered, strict=False)

        # Interpolate positional embedding if sizes differ
        if pos_emb_ckpt is not None:
            tgt_len = model_state[pos_emb_key].shape[0]
            if pos_emb_ckpt.shape[0] != tgt_len:
                resized = self._model.decoder.resize_bart_abs_pos_emb(
                    pos_emb_ckpt, tgt_len
                )
                self._model.decoder.model.model.decoder.embed_positions.weight = (
                    torch.nn.Parameter(resized.to(self._device))
                )
            else:
                self._model.decoder.model.model.decoder.embed_positions.weight = (
                    torch.nn.Parameter(pos_emb_ckpt.to(self._device))
                )

        logger.info(
            f"[VLM SCAN] Checkpoint loaded — missing: {len(missing)}, "
            f"unexpected: {len(unexpected)}"
        )

        self._model.eval()

        # Patch for transformers >= 4.38: prepare_inputs_for_generation is
        # bound as an instance method on decoder.model, so patch the instance.
        _orig = self._model.decoder.prepare_inputs_for_inference
        def _patched(input_ids, past_key_values=None, attention_mask=None,
                     use_cache=None, encoder_outputs=None, **_kw):
            return _orig(input_ids, past_key_values=past_key_values,
                         attention_mask=attention_mask, use_cache=use_cache,
                         encoder_outputs=encoder_outputs)
        self._model.decoder.model.prepare_inputs_for_generation = _patched

        logger.info(f"[VLM SCAN] Nougat ready in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Page selection (skip well-covered, keep math-rich)
    # ------------------------------------------------------------------

    def _select_pages(
        self, pdf_path: str, doc_id: str, existing: List[Dict]
    ) -> List[int]:
        """Return page numbers worth scanning (math-rich but poorly covered)."""
        import fitz  # PyMuPDF

        # Count good (non-trivial, non-OCR) existing formulas per page for this doc.
        # OCR-fallback formulas are lower quality than Nougat — pages that only have
        # OCR coverage should still be scanned by Nougat.
        good_per_page: Dict[int, int] = {}
        for fc in existing:
            if (
                fc.get("doc_id") == doc_id
                and not fc.get("is_trivial", False)
                and fc.get("extraction_source", "") != "ocr_fallback"
            ):
                p = fc.get("page_number", -1)
                good_per_page[p] = good_per_page.get(p, 0) + 1

        MATH_CHARS = set("=^/∫∑∂≤≥∝≡∼≈±×÷")
        selected = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            # Skip pages already well-covered by the regex/docling pipeline
            if good_per_page.get(page_num, 0) >= 3:
                continue
            text = doc[page_num].get_text()
            # Image-only pages (scanned PDFs) have no text layer — these are
            # exactly what Nougat is designed for, so always include them.
            if len(text) < 30:
                selected.append(page_num)
                continue
            math_density = sum(text.count(c) for c in MATH_CHARS) / max(len(text), 1)
            if math_density >= 0.002 or any(
                kw in text.lower()
                for kw in (
                    "equation",
                    "formula",
                    "potential",
                    "hamiltonian",
                    "\\begin{equation}",
                )
            ):
                selected.append(page_num)
        doc.close()
        return selected

    # ------------------------------------------------------------------
    # Page → PIL image
    # ------------------------------------------------------------------

    def _page_to_pil(self, pdf_path: str, page_num: int):
        """Render a PDF page to a PIL Image at 2× zoom."""
        import io
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        doc.close()
        return Image.open(io.BytesIO(pix.tobytes("png")))

    # ------------------------------------------------------------------
    # Nougat inference
    # ------------------------------------------------------------------

    def _run_nougat(self, pil_image) -> str:
        """Run Nougat on a PIL image; return the .mmd string."""
        import torch

        with torch.no_grad():
            out = self._model.inference(image=pil_image, early_stopping=True)
        return out["predictions"][0] if out.get("predictions") else ""

    # ------------------------------------------------------------------
    # .mmd formula parsing
    # ------------------------------------------------------------------

    def _parse_mmd(self, mmd: str) -> List[tuple]:
        """Extract (latex, context_text) pairs from a Nougat .mmd page output."""
        results: List[tuple] = []
        CONTEXT_WINDOW = 300  # chars before/after delimiter

        for pat in (self._DISPLAY_RE, self._DOLLAR_RE, self._INLINE_RE):
            for m in pat.finditer(mmd):
                latex = m.group(1).strip()

                # Extract surrounding prose
                pre_start = max(0, m.start() - CONTEXT_WINDOW)
                post_end = min(len(mmd), m.end() + CONTEXT_WINDOW)
                pre_text = mmd[pre_start:m.start()]
                post_text = mmd[m.end():post_end]

                # Strip nested formula delimiters from context
                pre_clean = re.sub(
                    r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$", "", pre_text, flags=re.DOTALL
                )
                post_clean = re.sub(
                    r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$", "", post_text, flags=re.DOTALL
                )

                context = re.sub(r"\s+", " ", (pre_clean + " " + post_clean)).strip()
                context = context[:400]  # cap at 400 chars total

                results.append((latex, context))
        return results

    # ------------------------------------------------------------------
    # Main scan entry point
    # ------------------------------------------------------------------

    def scan_pdf(
        self, pdf_path: str, doc_id: str, existing: List[Dict]
    ) -> Dict:
        """
        Scan a single PDF with Nougat.

        Parameters
        ----------
        pdf_path : str
            Absolute path to the PDF file.
        doc_id : str
            Document identifier (filename without extension).
        existing : list[dict]
            Current formula chunks for *all* documents (not just this one).

        Returns
        -------
        dict with keys:
            merged        — new combined formula list (all docs)
            added         — number of new formulas added
            replaced      — number of old formula slots replaced
            pages_scanned — how many pages Nougat processed
            elapsed_s     — wall-clock seconds
        """
        self._ensure_model()
        pages = self._select_pages(pdf_path, doc_id, existing)
        if not pages:
            logger.info(f"[VLM SCAN] {doc_id}: no pages to scan (all well-covered)")
            return {
                "merged": existing,
                "added": 0,
                "replaced": 0,
                "pages_scanned": 0,
                "elapsed_s": 0.0,
            }

        t0 = time.time()
        raw_chunks: List[Dict] = []

        for page_num in pages:
            logger.debug(f"[VLM SCAN] {doc_id} page {page_num} …")
            try:
                img = self._page_to_pil(pdf_path, page_num)
                mmd = self._run_nougat(img)
                mmd = _fix_nougat_greek(mmd)   # fix Greek misreadings at document level
                formulas = self._parse_mmd(mmd)
                for idx, (latex, context_text) in enumerate(formulas):
                    cleaned = _normalize_nougat_latex(latex)
                    if cleaned:
                        raw_chunks.append(
                            _make_formula_chunk(cleaned, doc_id, page_num, idx, context_text)
                        )
            except Exception as exc:
                logger.warning(
                    f"[VLM SCAN] {doc_id} page {page_num} failed: {exc}"
                )

        # Embed all new chunks in one batch
        if raw_chunks:
            texts = [fc["embedding_text"] for fc in raw_chunks]
            try:
                embs = self.embedder.embed_texts(texts)
                for fc, emb in zip(raw_chunks, embs):
                    fc["embedding"] = (
                        emb.tolist() if hasattr(emb, "tolist") else emb
                    )
            except Exception as exc:
                logger.warning(f"[VLM SCAN] Embedding failed: {exc}")
                raw_chunks = []   # discard un-embedded chunks

        # Deduplicate within the new batch, then merge into the full index
        raw_chunks = _remove_subsets(raw_chunks)

        from core.formula.deduplicator import _merge_nougat_formulas

        merged, added, replaced = _merge_nougat_formulas(existing, raw_chunks, doc_id)
        elapsed = time.time() - t0

        return {
            "merged": merged,
            "added": added,
            "replaced": replaced,
            "pages_scanned": len(pages),
            "elapsed_s": elapsed,
        }
