from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
import warnings

# Suppress transformers generation warnings (not relevant for embeddings)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")


class Embedder:
    """
    Wrapper for embedding models used in PresciSE.

    Uses SPECTER for scientific document embeddings with GPU acceleration.
    """

    def __init__(self, model_name: str = "allenai/specter"):
        """
        Initialize embedder with SPECTER.
        
        Args:
            model_name: Model identifier (default: SPECTER for scientific papers)
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model: Optional[SentenceTransformer] = None
        
        if self.device == 'cuda':
            print(f"  🎮 GPU detected! Using {torch.cuda.get_device_name(0)}")
        else:
            print("  💻 No GPU detected, using CPU")

    def _load(self):
        """Lazy load the model to save memory."""
        if self._model is None:
            print(f"  📥 Loading {self.model_name} on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"  ✅ Model loaded (embedding dim: {self._model.get_sentence_embedding_dimension()})")

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single piece of text.
        
        Args:
            text: Input text
            
        Returns:
            768-dimensional embedding vector (SPECTER)
        """
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple texts efficiently with batch processing.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of 768-dimensional embedding vectors
        """
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]