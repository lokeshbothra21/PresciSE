"""Structure-aware chunking for PresciSE.

Replaces the old raw character-slicing chunker (``text[start:start+max_chars]``),
which split mid-word and mid-sentence. This packs whole SENTENCES up to a size
target with sentence-level overlap, so a chunk never cuts through the sentence
that answers a question. It also guards against false sentence breaks on decimals
("0.5") and common scientific abbreviations ("Fig.", "et al."), and propagates
richer metadata (doc title + section type + pages) onto each chunk.

Public API is unchanged: make_chunks_from_doc(doc, max_chars=900, overlap=150).
"""

import re
from typing import Any, Dict, List, Union

from core.ingestion.document_schema import Document
from core.nlp.tokenizer import tokenize

# Abbreviations that end in "." but do NOT end a sentence (lowercased, no dot).
_ABBREV = {
    "fig", "figs", "eq", "eqs", "ref", "refs", "al", "etc", "e.g", "i.e",
    "vs", "cf", "approx", "no", "vol", "pp", "ca", "dr", "prof", "mr", "ms",
    "st", "tbl", "sec", "ch", "ed", "eds", "repr", "min", "max", "avg",
}

# Split points: sentence-ending punctuation followed by whitespace. Note a
# decimal like "0.5" has no whitespace after the dot, so it is never split here.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting paragraph breaks, decimals and
    abbreviations (merge a fragment back when the previous 'sentence' ended in a
    known abbreviation or a single-letter initial)."""
    sentences: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        frags = _SENT_BOUNDARY.split(para)
        for frag in frags:
            frag = frag.strip()
            if not frag:
                continue
            if sentences:
                prev = sentences[-1]
                m = re.findall(r"[A-Za-z.]+$", prev)
                tail = (m[0].rstrip(".").lower() if m else "")
                if tail in _ABBREV or len(tail) == 1:
                    sentences[-1] = prev + " " + frag
                    continue
            sentences.append(frag)
    return sentences


def _hard_split(sentence: str, max_chars: int) -> List[str]:
    """Fallback for a single sentence longer than max_chars: split on whitespace
    into <=max_chars pieces (rare; avoids unbounded chunks)."""
    pieces: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for word in sentence.split():
        if cur and cur_len + len(word) + 1 > max_chars:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(word)
        cur_len += len(word) + 1
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _pack_sentences(sentences: List[str], max_chars: int, overlap: int) -> List[str]:
    """Pack sentences into <=max_chars chunks, carrying trailing sentences up to
    ~overlap chars into the next chunk for context continuity."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.extend(_hard_split(s, max_chars))
            continue
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            # Build the overlap: trailing sentences of the chunk just emitted.
            carry: List[str] = []
            carry_len = 0
            for prev in reversed(cur):
                if carry and carry_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            cur = carry
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_chunks_from_doc(
    doc: Union[Document, Dict[str, Any]],
    max_chars: int = 900,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Create retrieval chunks from a normalized PresciSE Document (Pydantic or dict).

    Strategy:
      - iterate sections,
      - split section text into sentences (decimal/abbreviation-safe),
      - pack whole sentences into <=max_chars chunks with sentence-level overlap,
      - attach metadata (doc title, section_type, pages).

    Args:
        doc: Document to chunk (Pydantic model or dict).
        max_chars: Target maximum characters per chunk (default 900).
        overlap: Approximate character overlap carried between consecutive chunks.
    """
    chunks: List[Dict[str, Any]] = []

    if isinstance(doc, Document):
        doc_id = doc.document_id
        doc_title = doc.title or ""
        sections = doc.sections
    else:
        doc_id = doc["document_id"]
        doc_title = doc.get("title") or ""
        sections = doc.get("sections", [])

    for i, section in enumerate(sections):
        if hasattr(section, "section_type"):
            section_type = section.section_type
            text = (section.content or "").strip()
            pages = section.page_numbers
        else:
            section_type = section.get("section_type", "unknown")
            text = (section.get("content") or "").strip()
            pages = section.get("page_numbers", [])

        if not text:
            continue

        sentences = _split_sentences(text)
        if not sentences:
            continue

        for part, piece in enumerate(_pack_sentences(sentences, max_chars, overlap)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "chunk_id": f"{doc_id}_{section_type}_{i}_{part}",
                "doc_id": doc_id,
                "text": piece,
                "tokens": tokenize(piece),
                "metadata": {
                    "section_type": section_type,
                    "pages": pages,
                    "doc_title": doc_title,
                },
            })

    return chunks
