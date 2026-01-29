from typing import List, Optional
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper for embedding models used in PresciSE.

    Uses SentenceTransformers locally (client-friendly).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _load(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single piece of text.
        """
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple texts efficiently.
        """
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
