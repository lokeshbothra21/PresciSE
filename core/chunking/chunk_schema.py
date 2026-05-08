from typing import Dict, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Canonical chunk representation used across PresciSE.

    A Chunk is the atomic unit for retrieval:
    - BM25 uses: tokens
    - FAISS uses: embedding
    - Agents/LLMs use: text + metadata
    """
    chunk_id: str = Field(..., description="Unique ID for this chunk")
    doc_id: str = Field(..., description="Unique ID for the source document")

    text: str = Field(..., description="Chunk text content")
    tokens: list[str] = Field(default_factory=list, description="Tokenized representation for lexical retrieval")

    metadata: Dict = Field(default_factory=dict, description="Additional metadata e.g. page, section, filename")

    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding of the chunk text for semantic retrieval (FAISS)"
    )
