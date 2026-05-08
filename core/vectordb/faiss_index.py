from typing import List
import numpy as np
import faiss
from loguru import logger


class FAISSIndex:
    """
    Simple FAISS wrapper for semantic search.

    We use cosine similarity by storing normalized embeddings
    and using Inner Product similarity (IndexFlatIP).
    """

    def __init__(self, embedding_dim: int, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.on_gpu = False
        self._gpu_res = None

        cpu_index = faiss.IndexFlatIP(embedding_dim)

        # Optional GPU acceleration when faiss-gpu is installed.
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                self._gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
                self.on_gpu = True
                logger.info("FAISS index initialized on GPU")
            except Exception as e:
                logger.warning(f"FAISS GPU init failed, falling back to CPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index

    def add(self, embeddings: List[List[float]]):
        vecs = np.array(embeddings, dtype="float32")
        if vecs.ndim != 2 or vecs.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings with dim={self.embedding_dim}, got {vecs.shape}")
        self.index.add(vecs)

    def search(self, query_embedding: List[float], top_k: int = 5):
        q = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(q, top_k)
        return scores[0].tolist(), indices[0].tolist()

    def get_persistable_index(self):
        """
        Return a CPU index suitable for faiss.write_index.
        """
        if self.on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    def load_from_cpu_index(self, cpu_index):
        """
        Load a CPU index, optionally moving it to GPU.
        """
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                self._gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
                self.on_gpu = True
                logger.info("FAISS index moved to GPU after load")
                return
            except Exception as e:
                logger.warning(f"FAISS GPU load failed, using CPU index: {e}")
        self.index = cpu_index
        self.on_gpu = False
