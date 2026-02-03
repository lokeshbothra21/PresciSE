"""
Persistence module for PresciSE.

Handles:
- Index persistence (BM25, FAISS)
- Document tracking and change detection
- Incremental updates
"""

from .index_manager import IndexManager
from .document_tracker import DocumentTracker
from .utils import get_file_hash, ensure_dir

__all__ = ["IndexManager", "DocumentTracker", "get_file_hash", "ensure_dir"]
