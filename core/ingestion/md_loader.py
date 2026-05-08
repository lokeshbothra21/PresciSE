"""
Markdown document loader for PresciSE.

Reads a .md file and converts it into a Document compatible with
make_chunks_from_doc(). No Docling, no ML — pure text processing.

Sections are split at Markdown headings (# and ##). Content under
each heading becomes one Section. Files with no headings are treated
as a single section named "body".
"""

from __future__ import annotations

import re
from pathlib import Path

from core.ingestion.document_schema import Document, Section, Figure


def load_md_as_document(md_path: str) -> Document:
    """
    Parse a Markdown file into a PresciSE Document.

    Parameters
    ----------
    md_path : str
        Absolute or relative path to the .md file.

    Returns
    -------
    Document
        A Document with one Section per top-level heading block.
        page_numbers are all [0] (no physical pages in Markdown).
    """
    path = Path(md_path)
    doc_id = path.stem
    raw = path.read_text(encoding="utf-8", errors="replace")

    sections = _split_by_headings(raw)

    return Document(
        document_id=doc_id,
        title=_extract_title(raw, doc_id),
        sections=sections,
        figures=[],
        metadata={"source": str(path), "format": "markdown"},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[Section]:
    """
    Split markdown text into sections at # and ## headings.
    Returns a list of Section objects.
    """
    # Find all heading positions
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        # No headings — treat entire file as one body section
        content = _clean_md(text)
        if content:
            return [Section(section_type="body", content=content, page_numbers=[0])]
        return []

    sections: list[Section] = []

    # Text before the first heading (preamble)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        cleaned = _clean_md(preamble)
        if cleaned:
            sections.append(
                Section(section_type="preamble", content=cleaned, page_numbers=[0])
            )

    for i, m in enumerate(matches):
        heading_text = m.group(2).strip()
        section_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        body = text[section_start:section_end].strip()
        content = heading_text + "\n\n" + _clean_md(body) if body else heading_text
        content = content.strip()

        if content:
            sections.append(
                Section(
                    section_type="section",
                    content=content,
                    page_numbers=[0],
                )
            )

    return sections


def _clean_md(text: str) -> str:
    """
    Light Markdown cleanup — remove code fences, strip link syntax,
    collapse blank lines. Preserves all prose and inline code.
    """
    # Remove fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove image syntax ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Unwrap links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(text: str, fallback: str) -> str:
    """Return the first H1 heading as title, or fallback to filename stem."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback
