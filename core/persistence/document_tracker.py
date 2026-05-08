"""
Document registry tracker.

Maintains a JSON registry of all indexed documents with:
- File hash (SHA256) for reliable change detection
- Modification time
- Number of chunks
- List of chunk IDs (for deletion)
"""

import json
import os
from typing import Dict, List, Set
from pathlib import Path
from loguru import logger

from .utils import get_file_hash, get_file_mtime, ensure_dir, file_exists


class DocumentTracker:
    """
    Tracks which documents have been processed and indexed.
    
    Registry format:
    {
        "doc1.pdf": {
            "file_hash": "abc123...",
            "last_modified": 1706600000.0,
            "num_chunks": 264,
            "chunk_ids": ["doc1_text_0_0", ...]
        }
    }
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize document tracker.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.registry: Dict[str, dict] = {}
        self.load()
    
    def load(self) -> None:
        """Load registry from disk if it exists."""
        if file_exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} documents")
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted registry file, starting fresh: {e}")
                self.registry = {}
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.registry = {}
        else:
            logger.info("No existing registry found, starting fresh")
            self.registry = {}
    
    def save(self) -> None:
        """Save registry to disk."""
        ensure_dir(os.path.dirname(self.registry_path))
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Saved registry with {len(self.registry)} documents")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise
    
    def add_document(self, pdf_name: str, pdf_path: str, chunk_ids: List[str]) -> None:
        """
        Add or update a document in the registry.
        
        Args:
            pdf_name: Document filename (e.g., "doc1.pdf")
            pdf_path: Full path to PDF file
            chunk_ids: List of chunk IDs created from this document
        """
        try:
            file_hash = get_file_hash(pdf_path)
            mtime = get_file_mtime(pdf_path)
            
            self.registry[pdf_name] = {
                "file_hash": file_hash,
                "last_modified": mtime,
                "num_chunks": len(chunk_ids),
                "chunk_ids": chunk_ids
            }
            logger.debug(f"Added {pdf_name} to registry ({len(chunk_ids)} chunks)")
        except Exception as e:
            logger.error(f"Error adding document {pdf_name}: {e}")
            raise
    
    def remove_document(self, pdf_name: str) -> List[str]:
        """
        Remove a document from registry.
        
        Args:
            pdf_name: Document filename to remove
            
        Returns:
            List of chunk IDs that were associated with this document
        """
        if pdf_name in self.registry:
            chunk_ids = self.registry[pdf_name]["chunk_ids"]
            del self.registry[pdf_name]
            logger.info(f"Removed {pdf_name} from registry")
            return chunk_ids
        return []
    
    def has_changed(self, pdf_name: str, pdf_path: str) -> bool:
        """
        Check if a document has changed since last processing.
        
        Uses file hash (reliable) for change detection.
        
        Args:
            pdf_name: Document filename
            pdf_path: Full path to PDF file
            
        Returns:
            True if document has changed or is new
        """
        if pdf_name not in self.registry:
            return True  # New document
        
        try:
            current_hash = get_file_hash(pdf_path)
            stored_hash = self.registry[pdf_name]["file_hash"]
            return current_hash != stored_hash
        except Exception as e:
            logger.warning(f"Error checking {pdf_name}, treating as changed: {e}")
            return True  # Conservative: treat errors as changes
    
    def get_chunk_ids(self, pdf_name: str) -> List[str]:
        """
        Get chunk IDs for a document.
        
        Args:
            pdf_name: Document filename
            
        Returns:
            List of chunk IDs, or empty list if document not found
        """
        return self.registry.get(pdf_name, {}).get("chunk_ids", [])
    
    def get_all_documents(self) -> Set[str]:
        """
        Get set of all tracked document names.
        
        Returns:
            Set of document filenames
        """
        return set(self.registry.keys())
