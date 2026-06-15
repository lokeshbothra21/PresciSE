"""Re-embed every stored chunk with the current PRESCISE_EMBED_MODEL.

Run this after changing the embedding model (e.g. SPECTER 768-dim ->
models/gemini-embedding-001 3072-dim): query embeddings from the new model are
incompatible with vectors stored under the old one, so the whole corpus must be
re-embedded. This rewrites the embedding BLOBs in the DB in place; FAISS + BM25
rebuild from the DB on the next API startup, so there are no index files to touch.

Usage (from repo root):
    python -m scripts.reembed              # re-embed all chunks
    python -m scripts.reembed --dry-run    # just report how many would change
    python -m scripts.reembed --batch 32   # tune embedding batch size

Honors PRESCISE_EMBED_MODEL / PRESCISE_DATABASE_URL / GEMINI_API_KEY from the env.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def _embed_input(chunk_type: str, data: dict) -> str:
    """Reconstruct the text that was embedded for a chunk (matches indexing)."""
    if chunk_type == "formula":
        return (
            data.get("embedding_text")
            or "The equation is defined as:\n"
            f"{data.get('normalized_formula', data.get('formula_text', ''))}\n"
            f"{data.get('context_text', '')}"
        )
    return data.get("text", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size.")
    ap.add_argument("--dry-run", action="store_true", help="Report counts only; no writes.")
    args = ap.parse_args()

    from core.persistence import db
    from core.embeddings.embedder import Embedder

    db.init_db()
    rows = db.all_chunks_for_reembed()
    model = os.getenv("PRESCISE_EMBED_MODEL", "models/gemini-embedding-001")
    print(f"{len(rows)} chunk(s) to re-embed with model={model}")
    if args.dry_run or not rows:
        return

    embedder = Embedder()
    updated = 0
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        texts = [_embed_input(ct, dj) for (_cid, ct, dj) in batch]
        embs = embedder.embed_texts(texts)
        mapping = {cid: emb for (cid, _ct, _dj), emb in zip(batch, embs)}
        updated += db.update_chunk_embeddings(mapping)
        print(f"  re-embedded {min(i + args.batch, len(rows))}/{len(rows)}")

    print(f"Done. Updated {updated} chunk embedding(s). Restart the API to rebuild the index.")


if __name__ == "__main__":
    main()
