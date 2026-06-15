"""Embedding model wrapper for PresciSE.

Default: hosted Gemini embeddings (``models/gemini-embedding-001``) — strong
retrieval quality and the same provider as the answer LLM. Falls back to a local
SentenceTransformer (e.g. ``allenai/specter``) when PRESCISE_EMBED_MODEL names a
local model, or when Gemini is selected but no API key is present (keeps local
dev and offline tests working).

Select the model with PRESCISE_EMBED_MODEL. Convention:
  * embed_text()  -> a single QUERY string
  * embed_texts() -> a batch of DOCUMENT chunks
"""

import os
import threading
import warnings
from typing import List, Optional

from loguru import logger

# Suppress transformers generation warnings (not relevant for embeddings)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.generation.configuration_utils"
)

_DEFAULT_MODEL = "models/gemini-embedding-001"
_LOCAL_FALLBACK = "allenai/specter"


def _l2_normalize(vec) -> List[float]:
    """Unit-normalize a vector so FAISS/pgvector inner-product == cosine.

    Gemini embeddings are only guaranteed pre-normalized at the full 3072 dim;
    normalizing here keeps the index correct across any output dimension.
    """
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


class Embedder:
    """Query/document embedder with hosted-Gemini or local-model backends."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("PRESCISE_EMBED_MODEL", _DEFAULT_MODEL)
        self._is_gemini = "gemini" in self.model_name.lower()
        self._model = None
        # encode()/embed are not safe to call concurrently on one model; serialise
        # load + inference behind a single lock (each request runs in a worker).
        self._lock = threading.Lock()

        if self._is_gemini and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            logger.warning(
                f"{self.model_name} selected but no GEMINI_API_KEY/GOOGLE_API_KEY — "
                f"falling back to local {_LOCAL_FALLBACK}."
            )
            self._is_gemini = False
            self.model_name = _LOCAL_FALLBACK

        if self._is_gemini:
            self.device = "api"
            logger.info(f"Embedder: hosted Gemini embeddings ({self.model_name})")
        else:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Embedder: local model {self.model_name} on {self.device}")

    def _load(self):
        """Lazy-load the backend (Gemini client or local SentenceTransformer)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if self._is_gemini:
                        from langchain_google_genai import GoogleGenerativeAIEmbeddings

                        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                        logger.info(f"Loading Gemini embeddings: {self.model_name}")
                        self._model = GoogleGenerativeAIEmbeddings(
                            model=self.model_name, google_api_key=api_key
                        )
                    else:
                        from sentence_transformers import SentenceTransformer

                        logger.info(f"Loading {self.model_name} on {self.device}...")
                        self._model = SentenceTransformer(self.model_name, device=self.device)
                        logger.info(
                            "Model loaded (embedding dim: "
                            f"{self._model.get_sentence_embedding_dimension()})"
                        )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single QUERY string."""
        self._load()
        if self._is_gemini:
            with self._lock:
                vec = self._model.embed_query(text)
            return _l2_normalize(vec)
        with self._lock:
            vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of DOCUMENT chunks."""
        self._load()
        if self._is_gemini:
            with self._lock:
                vecs = self._model.embed_documents(list(texts))
            return [_l2_normalize(v) for v in vecs]
        with self._lock:
            vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
