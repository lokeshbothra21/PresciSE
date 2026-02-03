"""
Index Manager - Orchestrates persistent indexes with incremental updates.

Handles:
- Loading existing indexes from disk
- Detecting document changes (new, modified, deleted)
- Incremental index updates
- Saving indexes to disk

Priority: Reliability > Speed
"""

import os
import pickle
from glob import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from core.ingestion.pdf_loader import load_pdf_as_document
from core.chunking.pdf_chunker import make_chunks_from_doc
from core.embeddings.embedder import Embedder
from core.retrieval.hybrid_retriever import HybridRetriever

from .document_tracker import DocumentTracker
from .utils import ensure_dir, file_exists


class IndexManager:
    """
    Manages persistent indexes with incremental updates.
    
    Directory structure:
        data/index/
        ├── chunks.pkl              # All chunks with embeddings
        ├── document_registry.json  # Tracks processed documents
        ├── bm25_index.pkl         # BM25 index
        └── faiss_index.bin        # FAISS index
    """
    
    def __init__(self, pdf_dir: str = "data/pdfs", index_dir: str = "data/index"):
        """
        Initialize Index Manager.
        
        Args:
            pdf_dir: Directory containing PDF files
            index_dir: Directory for storing indexes
        """
        self.pdf_dir = pdf_dir
        self.index_dir = index_dir
        self.registry_path = os.path.join(index_dir, "document_registry.json")
        self.chunks_path = os.path.join(index_dir, "chunks.pkl")
        
        ensure_dir(index_dir)
        self.tracker = DocumentTracker(self.registry_path)
    
    def load_or_build(self) -> HybridRetriever:
        """
        Load indexes from disk or build if needed.
        
        Returns:
            HybridRetriever with up-to-date indexes
        """
        logger.info("Checking index status...")
        
        # Scan PDF directory
        current_pdfs = self._get_current_pdfs()
        
        # Detect changes
        new_docs, modified_docs, deleted_docs = self._detect_changes(current_pdfs)
        
        # If no changes, load from disk
        if not new_docs and not modified_docs and not deleted_docs:
            if self._indexes_exist():
                logger.info("✅ Index up-to-date. Loading from disk...")
                return self._load_indexes()
            else:
                logger.info("No indexes found. Building from scratch...")
        
        # Otherwise, perform incremental update
        if new_docs or modified_docs or deleted_docs:
            logger.info(f"📝 Changes detected:")
            logger.info(f"  New: {len(new_docs)}, Modified: {len(modified_docs)}, Deleted: {len(deleted_docs)}")
        
        return self._incremental_update(new_docs, modified_docs, deleted_docs)
    
    def _get_current_pdfs(self) -> Dict[str, str]:
        """
        Get current PDFs in the pdf_dir.
        
        Returns:
            Dict mapping PDF basename to full path
        """
        pdf_pattern = os.path.join(self.pdf_dir, "*.pdf")
        pdf_paths = glob(pdf_pattern)
        
        return {
            os.path.basename(path): path
            for path in pdf_paths
        }
    
    def _detect_changes(self, current_pdfs: Dict[str, str]) -> tuple:
        """
        Detect new, modified, and deleted documents.
        
        Args:
            current_pdfs: Dict of current PDF names to paths
            
        Returns:
            Tuple of (new_docs, modified_docs, deleted_docs)
        """
        tracked_docs = self.tracker.get_all_documents()
        current_names = set(current_pdfs.keys())
        
        new_docs = []
        modified_docs = []
        deleted_docs = []
        
        # Check for new and modified
        for pdf_name, pdf_path in current_pdfs.items():
            if pdf_name not in tracked_docs:
                new_docs.append((pdf_name, pdf_path))
            elif self.tracker.has_changed(pdf_name, pdf_path):
                modified_docs.append((pdf_name, pdf_path))
        
        # Check for deleted
        for pdf_name in tracked_docs:
            if pdf_name not in current_names:
                deleted_docs.append(pdf_name)
        
        return new_docs, modified_docs, deleted_docs
    
    def _indexes_exist(self) -> bool:
        """Check if all index files exist."""
        required_files = [
            self.chunks_path,
            os.path.join(self.index_dir, "bm25_index.pkl"),
            os.path.join(self.index_dir, "faiss_index.bin"),
        ]
        return all(file_exists(f) for f in required_files)
    
    def _load_indexes(self) -> HybridRetriever:
        """
        Load indexes from disk.
        
        Returns:
            HybridRetriever with loaded indexes
        """
        try:
            # Load chunks
            with open(self.chunks_path, "rb") as f:
                chunks = pickle.load(f)
            
            logger.info(f"✅ Loaded {len(chunks)} chunks from disk")
            
            # Create retriever and load indexes
            retriever = HybridRetriever(chunks)
            retriever.load(self.index_dir)
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error loading indexes, rebuilding: {e}")
            # If loading fails, rebuild from scratch
            return self._build_from_scratch()
    
    def _incremental_update(self, new_docs, modified_docs, deleted_docs) -> HybridRetriever:
        """
        Perform incremental index update.
        
        Args:
            new_docs: List of (pdf_name, pdf_path) tuples for new documents
            modified_docs: List of (pdf_name, pdf_path) tuples for modified documents
            deleted_docs: List of pdf_names for deleted documents
            
        Returns:
            Updated HybridRetriever
        """
        # Load existing chunks or start fresh
        if file_exists(self.chunks_path):
            with open(self.chunks_path, "rb") as f:
                all_chunks = pickle.load(f)
            logger.info(f"Loaded {len(all_chunks)} existing chunks")
        else:
            all_chunks = []
            logger.info("Starting with empty chunk collection")
        
        # Handle deletions
        for pdf_name in deleted_docs:
            chunk_ids_to_remove = set(self.tracker.get_chunk_ids(pdf_name))
            all_chunks = [c for c in all_chunks if c["chunk_id"] not in chunk_ids_to_remove]
            self.tracker.remove_document(pdf_name)
            logger.info(f"🗑️  Removed: {pdf_name}")
        
        # Handle modifications (remove old, will add as new)
        for pdf_name, pdf_path in modified_docs:
            chunk_ids_to_remove = set(self.tracker.get_chunk_ids(pdf_name))
            all_chunks = [c for c in all_chunks if c["chunk_id"] not in chunk_ids_to_remove]
            logger.info(f"🔄 Modified: {pdf_name} (removing old chunks)")
        
        # Add modified docs to new_docs list
        docs_to_process = list(new_docs) + list(modified_docs)
        
        # Process new/modified documents
        if docs_to_process:
            embedder = Embedder()
            
            for pdf_name, pdf_path in docs_to_process:
                logger.info(f"📄 Processing: {pdf_name}")
                
                try:
                    # Load and chunk
                    doc = load_pdf_as_document(pdf_path)
                    chunks = make_chunks_from_doc(doc)
                    
                    # Embed
                    texts = [c["text"] for c in chunks]
                    embeddings = embedder.embed_texts(texts)
                    
                    for c, emb in zip(chunks, embeddings):
                        c["embedding"] = emb
                    
                    # Add to corpus
                    all_chunks.extend(chunks)
                    
                    # Update registry
                    chunk_ids = [c["chunk_id"] for c in chunks]
                    self.tracker.add_document(pdf_name, pdf_path, chunk_ids)
                    
                    logger.info(f"  ✅ Created {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing {pdf_name}: {e}")
                    continue
        
        # Rebuild indexes
        logger.info(f"🔨 Rebuilding indexes with {len(all_chunks)} total chunks...")
        retriever = HybridRetriever(all_chunks)
        
        # Save everything
        self._save_indexes(retriever, all_chunks)
        
        return retriever
    
    def _build_from_scratch(self) -> HybridRetriever:
        """
        Build indexes from scratch (all PDFs).
        
        Returns:
            Newly built HybridRetriever
        """
        current_pdfs = self._get_current_pdfs()
        new_docs = [(name, path) for name, path in current_pdfs.items()]
        
        # Clear registry
        self.tracker.registry = {}
        
        return self._incremental_update(new_docs, [], [])
    
    def _save_indexes(self, retriever: HybridRetriever, chunks: List[Dict[str, Any]]) -> None:
        """
        Save indexes to disk.
        
        Args:
            retriever: HybridRetriever to save
            chunks: List of all chunks
        """
        try:
            # Save chunks
            with open(self.chunks_path, "wb") as f:
                pickle.dump(chunks, f)
            
            # Save indexes
            retriever.save(self.index_dir)
            
            # Save registry
            self.tracker.save()
            
            logger.info(f"💾 Saved indexes to {self.index_dir}/")
            
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
            raise
