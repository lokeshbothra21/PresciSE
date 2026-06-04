"""
Index Manager - Orchestrates persistent indexes with incremental updates.

Handles:
- Loading existing indexes from disk
- Detecting document changes (new, modified, deleted)
- Incremental index updates (text + formula)
- Saving indexes to disk
- Formula extraction and separate formula indexing

Priority: Reliability > Speed
"""

import os
import pickle
import time
from glob import glob
from typing import List, Dict, Any, Optional
from loguru import logger

from core.ingestion.pdf_loader import load_pdf_as_document
from core.ingestion.md_loader import load_md_as_document
from core.chunking.pdf_chunker import make_chunks_from_doc
from core.formula.page_text_extractor import extract_formulas_from_pdf_pages
from core.formula.diagnostics import build_formula_diagnostics, save_formula_diagnostics
from core.embeddings.embedder import Embedder
from core.retrieval.hybrid_retriever import HybridRetriever

from .document_tracker import DocumentTracker
from .utils import ensure_dir, file_exists


class IndexManager:
    """
    Manages persistent indexes with incremental updates.
    
    Directory structure:
        data/index/
        ├── chunks.pkl              # Text chunks with embeddings
        ├── formula_chunks.pkl      # Formula chunks with embeddings
        ├── document_registry.json  # Tracks processed documents
        ├── bm25_index.pkl         # BM25 index (text)
        ├── faiss_index.bin        # FAISS index (text)
        └── formula_faiss_index.bin # FAISS index (formulas)
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
        self.formula_chunks_path = os.path.join(index_dir, "formula_chunks.pkl")
        self.formula_quality_mode = os.getenv("PRESCISE_FORMULA_QUALITY_MODE", "balanced")
        
        ensure_dir(index_dir)
        self.tracker = DocumentTracker(self.registry_path)
    
    def load_or_build(self) -> HybridRetriever:
        """
        Load indexes from disk or build if needed.
        
        Validates index integrity before loading to prevent crashes from
        stale or corrupted index data.
        
        Returns:
            HybridRetriever with up-to-date indexes
        """
        logger.info("Checking index status...")
        
        # Scan PDF directory
        current_pdfs = self._get_current_pdfs()
        
        # VALIDATE BEFORE LOAD: Check if indexes exist and are valid
        if self._indexes_exist():
            # Check if the saved chunks can be loaded and are non-empty
            is_valid, validation_msg = self._validate_indexes(current_pdfs)
            if not is_valid:
                logger.warning(f"[WARN] Index validation failed: {validation_msg}")
                logger.info("[REBUILD] Rebuilding indexes from scratch...")
                return self._build_from_scratch()
        
        # Detect changes
        new_docs, modified_docs, deleted_docs = self._detect_changes(current_pdfs)
        
        # If no changes, load from disk
        if not new_docs and not modified_docs and not deleted_docs:
            if self._indexes_exist():
                if not self._formula_index_exists():
                    logger.info("Text index up-to-date but formula index missing — rebuilding formulas only...")
                    return self._rebuild_formula_index_only()
                logger.info("✅ Index up-to-date. Loading from disk...")
                return self._load_indexes()
            else:
                logger.info("No indexes found. Building from scratch...")
        
        # Otherwise, perform incremental update
        if new_docs or modified_docs or deleted_docs:
            logger.info(f"Changes detected:")
            logger.info(f"  New: {len(new_docs)}, Modified: {len(modified_docs)}, Deleted: {len(deleted_docs)}")
        
        return self._incremental_update(new_docs, modified_docs, deleted_docs)
    
    def _validate_indexes(self, current_pdfs: Dict[str, str]) -> tuple:
        """
        Validate that saved indexes are consistent with current state.
        
        Checks:
        1. Chunks file exists and is loadable
        2. Chunks are non-empty
        3. Registry documents match current PDF files
        
        Args:
            current_pdfs: Dict of current PDF names to paths
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            # Check 1: Can we load chunks?
            with open(self.chunks_path, "rb") as f:
                chunks = pickle.load(f)
            
            # Check 2: Are chunks non-empty?
            if not chunks or len(chunks) == 0:
                return False, "Saved chunks file is empty"
            
            # Check 3: Do all chunks have required fields?
            required_fields = ["chunk_id", "tokens", "text"]
            for i, chunk in enumerate(chunks[:5]):  # Check first 5 chunks
                for field in required_fields:
                    if field not in chunk:
                        return False, f"Chunk {i} missing required field: {field}"
            
            # Check 4: Does registry match current PDFs?
            tracked_docs = self.tracker.get_all_documents()
            current_pdf_names = set(current_pdfs.keys())
            
            # If there are tracked docs that no longer exist, index is stale
            missing_pdfs = tracked_docs - current_pdf_names
            if missing_pdfs:
                return False, f"Registry contains PDFs no longer in folder: {missing_pdfs}"
            
            # All checks passed
            return True, "Index is valid"
            
        except FileNotFoundError:
            return False, "Chunks file not found"
        except Exception as e:
            return False, f"Error validating index: {e}"
    
    def _get_current_pdfs(self) -> Dict[str, str]:
        """
        Get current documents (PDFs and Markdown files) in the pdf_dir.

        Returns:
            Dict mapping document basename to full path
        """
        paths = glob(os.path.join(self.pdf_dir, "*.pdf"))
        paths += glob(os.path.join(self.pdf_dir, "*.md"))

        return {
            os.path.basename(path): path
            for path in paths
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
        """Check if all text index files exist."""
        required_files = [
            self.chunks_path,
            os.path.join(self.index_dir, "bm25_index.pkl"),
            os.path.join(self.index_dir, "faiss_index.bin"),
        ]
        return all(file_exists(f) for f in required_files)

    def _formula_index_exists(self) -> bool:
        """Check if the formula index files exist."""
        return (
            file_exists(self.formula_chunks_path)
            and file_exists(os.path.join(self.index_dir, "formula_faiss_index.bin"))
        )
    
    def _rebuild_formula_index_only(self) -> HybridRetriever:
        """
        Re-extract and re-embed formula chunks for every PDF without touching text indexes.

        Called when text indexes are intact but formula_chunks.pkl or
        formula_faiss_index.bin is missing (e.g. after changing the extractor).
        Saves new formula_chunks.pkl and formula_faiss_index.bin to disk.
        """
        logger.info("Rebuilding formula index only (text indexes kept intact)...")

        # Load existing text chunks
        with open(self.chunks_path, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} text chunks for reference")

        # Determine which PDFs are indexed (by doc_id from text chunks)
        indexed_doc_ids = {c["doc_id"] for c in all_chunks}
        current_pdfs = self._get_current_pdfs()

        all_formula_chunks = []
        embedder = Embedder()

        for pdf_name, pdf_path in current_pdfs.items():
            doc_id = os.path.splitext(pdf_name)[0]
            if doc_id not in indexed_doc_ids:
                continue

            logger.info(f"  Extracting formulas from: {pdf_name}")
            doc = load_pdf_as_document(pdf_path)
            formula_chunks = extract_formulas_from_pdf_pages(
                pdf_path,
                doc_id=doc_id,
                quality_mode=self.formula_quality_mode,
            )

            if formula_chunks:
                texts = [c["text"] for c in formula_chunks]
                embeddings = embedder.embed_texts(texts)
                for c, emb in zip(formula_chunks, embeddings):
                    c["embedding"] = emb
                all_formula_chunks.extend(formula_chunks)
                logger.info(f"    → {len(formula_chunks)} formula chunks")

        logger.info(f"Formula rebuild complete: {len(all_formula_chunks)} total formula chunks")

        # Persist formula chunks
        with open(self.formula_chunks_path, "wb") as f:
            pickle.dump(all_formula_chunks, f)

        # Build retriever: load text FAISS+BM25 from disk, save formula FAISS
        retriever = HybridRetriever(all_chunks, formula_chunks=all_formula_chunks)
        retriever.load(self.index_dir)
        retriever.save(self.index_dir)

        return retriever

    def _load_indexes(self) -> HybridRetriever:
        """
        Load indexes from disk (text + formula).
        
        Returns:
            HybridRetriever with loaded indexes and formula chunks
        """
        try:
            # Load text chunks (with .bak fallback if the main file is corrupt)
            chunks = self._load_pickle_with_backup(self.chunks_path)

            logger.info(f"✅ Loaded {len(chunks)} text chunks from disk")

            # Load formula chunks
            formula_chunks = []
            if file_exists(self.formula_chunks_path):
                formula_chunks = self._load_pickle_with_backup(self.formula_chunks_path)
                logger.info(f"🔬 Loaded {len(formula_chunks)} formula chunks from disk")
            
            # Create retriever and load indexes
            retriever = HybridRetriever(chunks, formula_chunks=formula_chunks)
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
            logger.info(f"Loaded {len(all_chunks)} existing text chunks")
        else:
            all_chunks = []
            logger.info("Starting with empty text chunk collection")
        
        # Load existing formula chunks or start fresh
        if file_exists(self.formula_chunks_path):
            with open(self.formula_chunks_path, "rb") as f:
                all_formula_chunks = pickle.load(f)
            logger.info(f"Loaded {len(all_formula_chunks)} existing formula chunks")
        else:
            all_formula_chunks = []
        
        # Handle deletions
        for pdf_name in deleted_docs:
            chunk_ids_to_remove = set(self.tracker.get_chunk_ids(pdf_name))
            all_chunks = [c for c in all_chunks if c["chunk_id"] not in chunk_ids_to_remove]
            # Also remove formula chunks from deleted docs
            doc_id = os.path.splitext(pdf_name)[0]
            all_formula_chunks = [c for c in all_formula_chunks if c["doc_id"] != doc_id]
            self.tracker.remove_document(pdf_name)
            logger.info(f"[DELETED] Removed: {pdf_name}")
        
        # Handle modifications (remove old, will add as new)
        for pdf_name, pdf_path in modified_docs:
            chunk_ids_to_remove = set(self.tracker.get_chunk_ids(pdf_name))
            all_chunks = [c for c in all_chunks if c["chunk_id"] not in chunk_ids_to_remove]
            doc_id = os.path.splitext(pdf_name)[0]
            all_formula_chunks = [c for c in all_formula_chunks if c["doc_id"] != doc_id]
            logger.info(f"[MODIFIED] {pdf_name} (removing old chunks)")
        
        # Add modified docs to new_docs list
        docs_to_process = list(new_docs) + list(modified_docs)
        
        # Process new/modified documents
        stage_metrics: Dict[str, Any] = {
            "formula_counts_by_stage": {"standard": 0},
            "doc_warnings": [],
        }

        embedder = None
        if docs_to_process:
            embedder = Embedder()
            
            for pdf_name, pdf_path in docs_to_process:
                logger.info(f"Processing: {pdf_name}")
                
                try:
                    # Load document — branch on file type
                    is_markdown = pdf_name.lower().endswith(".md")
                    if is_markdown:
                        doc = load_md_as_document(pdf_path)
                    else:
                        doc = load_pdf_as_document(pdf_path)

                    # --- Text chunks ---
                    chunks = make_chunks_from_doc(doc)
                    texts = [c["text"] for c in chunks]
                    embeddings = embedder.embed_texts(texts)
                    for c, emb in zip(chunks, embeddings):
                        c["embedding"] = emb
                    all_chunks.extend(chunks)

                    # --- Formula chunks (PDF-only; skip for Markdown) ---
                    doc_id = os.path.splitext(pdf_name)[0]
                    if not is_markdown:
                        formula_chunks = extract_formulas_from_pdf_pages(
                            pdf_path,
                            doc_id=doc_id,
                            quality_mode=self.formula_quality_mode,
                        )
                        stage_metrics["formula_counts_by_stage"]["standard"] += len(formula_chunks)

                        cue_hits = self._count_equation_cues(doc)
                        if cue_hits >= 2 and not formula_chunks:
                            stage_metrics["doc_warnings"].append(
                                {
                                    "pdf_name": pdf_name,
                                    "doc_id": doc_id,
                                    "warning": "High equation cue density but no formulas retained",
                                    "cue_hits": cue_hits,
                                }
                            )
                            logger.warning(
                                f"  Formula warning: high cue density (hits={cue_hits}) but no formulas retained"
                            )

                        if formula_chunks:
                            formula_texts = [
                                fc.get("embedding_text")
                                or f"The equation is defined as:\n{fc.get('normalized_formula', fc.get('formula_text', ''))}\n{fc.get('context_text', '')}\nKeywords: equation model law theorem potential relation"
                                for fc in formula_chunks
                            ]
                            formula_embeddings = embedder.embed_texts(formula_texts)
                            for fc, emb in zip(formula_chunks, formula_embeddings):
                                fc["embedding"] = emb
                            all_formula_chunks.extend(formula_chunks)
                            logger.info(f"  Extracted {len(formula_chunks)} formulas")
                    
                    # Update registry
                    chunk_ids = [c["chunk_id"] for c in chunks]
                    self.tracker.add_document(pdf_name, pdf_path, chunk_ids)
                    
                    logger.info(f"  Created {len(chunks)} text chunks")
                    
                except Exception as e:
                    logger.error(f"  [ERROR] Processing {pdf_name}: {e}")
                    continue
        
        # Check that we have chunks to index
        if not all_chunks:
            raise ValueError(
                f"No chunks were generated from PDFs in {self.pdf_dir}. "
                "Please check that PDFs exist and are readable."
            )
        
        # Global subset-aware dedup across all formula chunks before indexing
        all_formula_chunks = self._global_dedup_formulas(all_formula_chunks)

        # Rebuild indexes
        logger.info(f"[BUILD] Rebuilding indexes with {len(all_chunks)} text chunks + {len(all_formula_chunks)} formula chunks...")
        retriever = HybridRetriever(all_chunks, formula_chunks=all_formula_chunks)

        # Save formula quality diagnostics for debugging retrieval failures.
        try:
            diag = build_formula_diagnostics(all_formula_chunks, stage_metrics=stage_metrics)
            diag_path = save_formula_diagnostics(diag, self.index_dir)
            logger.info(f"[DIAG] Formula diagnostics saved: {diag_path}")
        except Exception as diag_error:
            logger.warning(f"Could not write formula diagnostics: {diag_error}")
        
        # Save everything
        self._save_indexes(retriever, all_chunks, all_formula_chunks)

        # Launch Nougat branch in background after SPECTER has finished all docs
        # (avoids VRAM contention: SPECTER ~1.2GB + Nougat ~1.4GB).
        if os.getenv("PRESCISE_ENABLE_VLM_BACKGROUND_SCAN", "0") == "1" and embedder is not None:
            import threading
            current_pdfs = self._get_current_pdfs()
            nougat_thread = threading.Thread(
                target=self._run_nougat_branch,
                args=(current_pdfs, all_formula_chunks, embedder, retriever),
                daemon=False,
                name="nougat-branch",
            )
            nougat_thread.start()
            logger.info("[NOUGAT BRANCH] Background Nougat scan started after index build.")

        return retriever

    @staticmethod
    def _normed(fc: Dict[str, Any]) -> str:
        """Compact normalised key for subset comparison (strip whitespace and LaTeX noise)."""
        import re
        raw = (
            fc.get("normalized_formula")
            or fc.get("latex_formula")
            or fc.get("formula_text")
            or ""
        )
        return re.sub(r"[\s\\{}]", "", raw).lower()

    def _merge_formula_chunks(
        self,
        primary_chunks: List[Dict[str, Any]],
        fallback_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge formula chunks from primary and fallback extraction using subset-aware dedup.

        Rules (same page + doc_id):
        - If a fallback formula contains a primary formula as a *proper substring*,
          the primary is absorbed (replaced by the richer fallback).
        - If a fallback formula is itself a proper substring of any existing formula,
          it is skipped (we already have the fuller version).
        - Otherwise the fallback is added normally.
        """
        merged = list(primary_chunks)
        merged_normed = [self._normed(fc) for fc in merged]

        for fc_f in fallback_chunks:
            fn = self._normed(fc_f)
            if not fn:
                continue
            f_page = fc_f.get("page_number", -1)
            f_doc = fc_f.get("doc_id", "")

            # Indices of primaries that are proper substrings of this fallback (same page+doc)
            absorbed = [
                i
                for i, (mn, mc) in enumerate(zip(merged_normed, merged))
                if mn
                and mn in fn
                and mn != fn
                and mc.get("page_number", -1) == f_page
                and mc.get("doc_id", "") == f_doc
            ]
            # Is this fallback a proper substring of any existing formula?
            subsumed = any(fn in mn and fn != mn for mn in merged_normed)

            if subsumed:
                continue
            if absorbed:
                # Replace absorbed primaries with the richer fallback (highest index first)
                for i in sorted(absorbed, reverse=True):
                    merged.pop(i)
                    merged_normed.pop(i)
            merged.append(fc_f)
            merged_normed.append(fn)

        return merged

    def _global_dedup_formulas(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove any formula chunk whose normalised form is a proper substring of
        another chunk's normalised form anywhere in the collection.
        Keeps the most complete representation of each formula.
        """
        normed = [self._normed(fc) for fc in chunks]
        keep = []
        for i, fc in enumerate(chunks):
            if not normed[i]:
                keep.append(fc)
                continue
            dominated = any(
                j != i
                and normed[i]
                and normed[i] in normed[j]
                and normed[i] != normed[j]
                for j in range(len(chunks))
            )
            if not dominated:
                keep.append(fc)
        logger.info(f"Global formula dedup: {len(chunks)} → {len(keep)} chunks")
        return keep

    def _count_equation_cues(self, doc: Any) -> int:
        if isinstance(doc, dict):
            sections = doc.get("sections", [])
            section_texts = [(s.get("content", "") if isinstance(s, dict) else "") for s in sections]
        else:
            sections = getattr(doc, "sections", []) or []
            section_texts = []
            for s in sections:
                if hasattr(s, "content"):
                    section_texts.append(s.content or "")
                elif isinstance(s, dict):
                    section_texts.append(s.get("content", ""))
        corpus_text = " ".join(section_texts).lower()
        cue_patterns = [
            "equation",
            "potential",
            "hamiltonian",
            "partition function",
            "lennard",
            "jones",
            "where",
            "defined as",
            "theorem",
            "model",
            "eq.",
        ]
        return sum(1 for cue in cue_patterns if cue in corpus_text)

    def _run_nougat_branch(
        self,
        current_pdfs: Dict[str, str],
        primary_formula_chunks: List[Dict[str, Any]],
        embedder,
        retriever: HybridRetriever,
    ) -> None:
        """
        Background worker: scan all PDFs with Nougat, merge with primary formula
        chunks, then finalize and persist the formula index.

        Runs AFTER SPECTER has finished all docs to avoid VRAM contention.
        NougatFormulaScanner.needs_scan() is used to skip already-scanned PDFs.
        """
        from core.formula.nougat_scanner import NougatFormulaScanner

        logger.info("[NOUGAT BRANCH] Starting Nougat formula scan across all PDFs...")
        scanner = NougatFormulaScanner(embedder, self.index_dir)

        # Mutable working copy — updated doc-by-doc as Nougat finishes each PDF
        current_formula_chunks = list(primary_formula_chunks)
        total_added = 0
        total_t0 = time.time()

        for pdf_name, pdf_path in sorted(current_pdfs.items()):
            doc_id = self._derive_doc_id(pdf_path)
            if not scanner.needs_scan(doc_id):
                logger.debug(f"[NOUGAT BRANCH] {doc_id}: already scanned, skipping.")
                continue

            existing_doc = [fc for fc in current_formula_chunks if fc.get("doc_id") == doc_id]
            others = [fc for fc in current_formula_chunks if fc.get("doc_id") != doc_id]

            try:
                result = scanner.scan_pdf(str(pdf_path), doc_id, existing_doc)
                current_formula_chunks = others + result["merged"]
                scanner.mark_scanned(
                    doc_id,
                    result["pages_scanned"],
                    result["added"],
                    result["replaced"],
                    result["elapsed_s"],
                )
                total_added += result["added"]
                logger.info(
                    f"[NOUGAT BRANCH] {doc_id}: "
                    f"{result['pages_scanned']} pages | "
                    f"{result['elapsed_s']:.1f}s | "
                    f"formulas: {len(existing_doc)} → {len(result['merged'])} "
                    f"(+{result['added']} new, {result['replaced']} replaced)"
                )
            except Exception as exc:
                logger.error(f"[NOUGAT BRANCH] {doc_id}: scan failed: {exc}")

        # Save intermediate pending file for diagnostics / crash recovery
        pending_path = os.path.join(self.index_dir, "formula_chunks_nougat_pending.pkl")
        with open(pending_path, "wb") as f:
            pickle.dump(current_formula_chunks, f)
        logger.info(
            f"[NOUGAT BRANCH] Pending file saved: {len(current_formula_chunks)} formula chunks "
            f"({time.time() - total_t0:.1f}s total)"
        )

        # Finalize: global dedup + hot-swap retriever + persist
        self._finalize_formula_index(primary_formula_chunks, current_formula_chunks, retriever)
        logger.info(
            f"[NOUGAT BRANCH] Done. Formulas added by Nougat: {total_added}. "
            f"Final index size: {len(retriever.formula_chunks)}"
        )

    def _finalize_formula_index(
        self,
        primary_chunks: List[Dict[str, Any]],
        nougat_chunks: List[Dict[str, Any]],
        retriever: HybridRetriever,
    ) -> None:
        """
        Run global dedup on nougat-merged formula chunks, hot-swap the in-memory
        retriever, and persist the final formula index to disk.
        """
        final_chunks = self._global_dedup_formulas(nougat_chunks)
        logger.info(
            f"[NOUGAT BRANCH] Formula merge: {len(primary_chunks)} primary + "
            f"nougat → {len(final_chunks)} final"
        )
        retriever.update_formula_index(final_chunks)
        self.save_formula_chunks(final_chunks, retriever=retriever)
        logger.info(f"[NOUGAT BRANCH] Final formula index saved ({len(final_chunks)} chunks)")

    def _build_from_scratch(self) -> HybridRetriever:
        """
        Build indexes from scratch (all PDFs).
        
        Returns:
            Newly built HybridRetriever
        """
        current_pdfs = self._get_current_pdfs()
        new_docs = [(name, path) for name, path in current_pdfs.items()]
        
        # Clear registry and persisted index artifacts so rebuild is truly fresh.
        self.tracker.registry = {}
        self._clear_index_artifacts()
        
        return self._incremental_update(new_docs, [], [])

    def _clear_index_artifacts(self) -> None:
        """
        Remove persisted index/chunk artifacts before full rebuild.
        """
        artifacts = [
            self.chunks_path,
            self.formula_chunks_path,
            os.path.join(self.index_dir, "bm25_index.pkl"),
            os.path.join(self.index_dir, "faiss_index.bin"),
            os.path.join(self.index_dir, "formula_faiss_index.bin"),
        ]
        for path in artifacts:
            try:
                if file_exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Could not remove artifact {path}: {e}")
    
    def _derive_doc_id(self, pdf_path) -> str:
        """Return the doc_id for a PDF path (filename without extension)."""
        return os.path.splitext(os.path.basename(str(pdf_path)))[0]

    def save_formula_chunks(
        self,
        formula_chunks: List[Dict[str, Any]],
        retriever: HybridRetriever = None,
    ) -> None:
        """
        Persist formula chunks to disk and (optionally) save the FAISS index.

        Called by the Nougat background scanner after each PDF is processed so
        that the improved formula index survives server restarts.

        Parameters
        ----------
        formula_chunks : list[dict]
            The full updated formula chunk list (all documents).
        retriever : HybridRetriever, optional
            If provided, its FAISS index is also saved to disk.
        """
        import pickle

        with open(self.formula_chunks_path, "wb") as f:
            pickle.dump(formula_chunks, f)
        logger.info(
            f"[INDEX MGR] Saved {len(formula_chunks)} formula chunks → "
            f"{self.formula_chunks_path}"
        )

        if retriever is not None:
            try:
                retriever.save(self.index_dir)
                logger.info(
                    f"[INDEX MGR] Formula FAISS index saved → {self.index_dir}"
                )
            except Exception as exc:
                logger.warning(f"[INDEX MGR] Could not save formula FAISS index: {exc}")

    def load_only(self) -> HybridRetriever:
        """
        Startup path: rebuild the in-memory FAISS + BM25 indexes from the DB
        chunk store (the source of truth). No folder scan, no pickle — ingestion
        happens via upload. A fresh DB yields an empty retriever that uploads
        will populate.
        """
        from core.persistence import db

        text_chunks, formula_chunks = db.load_chunks()
        logger.info(
            f"Loaded {len(text_chunks)} text + {len(formula_chunks)} formula chunks "
            f"from DB (rebuilding FAISS + BM25 in memory)..."
        )
        return HybridRetriever(text_chunks, formula_chunks=formula_chunks)

    def index_uploaded_pdf(
        self,
        pdf_path: str,
        doc_id: str,
        owner_user_id: str,
        embedder: "Embedder",
        retriever: "HybridRetriever",
    ) -> int:
        """
        Parse, chunk, embed a single uploaded PDF; tag every chunk with
        ``doc_id`` + ``owner_user_id``; hot-add to the live retriever (no
        restart) and persist. Returns the number of text chunks added.

        Used by the upload endpoint's background indexer. Raises on failure so
        the caller can mark the document FAILED.
        """
        doc = load_pdf_as_document(pdf_path)

        # --- Text chunks ---
        chunks = make_chunks_from_doc(doc)
        if not chunks:
            raise ValueError("No text chunks were produced from the PDF.")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count {len(embeddings)} != chunk count {len(chunks)}"
            )
        # Tag EVERY chunk (loop over chunks, not zip) so a count mismatch can
        # never leave a chunk untagged → owner-less → world-visible ("__shared__").
        for i, c in enumerate(chunks):
            c["embedding"] = embeddings[i]
            c["doc_id"] = doc_id
            c["owner_user_id"] = owner_user_id
            c["chunk_id"] = f"{doc_id}_{i}"  # doc_id is a uuid → globally unique

        # --- Formula chunks (best-effort; never fail the upload over formulas) ---
        new_formula_chunks: List[Dict[str, Any]] = []
        try:
            formula_chunks = extract_formulas_from_pdf_pages(
                pdf_path, doc_id=doc_id, quality_mode=self.formula_quality_mode
            )
            if formula_chunks:
                formula_texts = [
                    fc.get("embedding_text")
                    or f"The equation is defined as:\n{fc.get('normalized_formula', fc.get('formula_text', ''))}\n{fc.get('context_text', '')}"
                    for fc in formula_chunks
                ]
                formula_embeddings = embedder.embed_texts(formula_texts)
                if len(formula_embeddings) != len(formula_chunks):
                    raise ValueError("formula embedding/chunk count mismatch")
                for j, fc in enumerate(formula_chunks):
                    fc["embedding"] = formula_embeddings[j]
                    fc["owner_user_id"] = owner_user_id
                new_formula_chunks = formula_chunks
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[UPLOAD] Formula extraction failed for {doc_id}: {exc}")

        # --- Hot-add to live indexes (atomic inside the retriever's locks) ---
        retriever.add_text_chunks(chunks)
        if new_formula_chunks:
            retriever.add_formula_chunks(new_formula_chunks)

        # --- Persist to the DB (the source of truth; survives restart) ---
        from core.persistence import db

        db.add_chunks(chunks, chunk_type="text")
        if new_formula_chunks:
            db.add_chunks(new_formula_chunks, chunk_type="formula")
        logger.info(f"[UPLOAD] Indexed {doc_id}: +{len(chunks)} chunks, +{len(new_formula_chunks)} formulas (DB-backed)")
        return len(chunks)

    def remove_document_from_index(self, doc_id: str, retriever: "HybridRetriever") -> None:
        """Remove a document's text + formula chunks from the live index and the DB."""
        from core.persistence import db

        retriever.remove_document_chunks(doc_id)
        all_formula_chunks = [
            fc for fc in retriever.formula_chunks if fc.get("doc_id") != doc_id
        ]
        if len(all_formula_chunks) != len(retriever.formula_chunks):
            retriever.update_formula_index(all_formula_chunks)
        db.delete_chunks_for_doc(doc_id)
        logger.info(f"[UPLOAD] Removed {doc_id} from index + DB")

    @staticmethod
    def _atomic_pickle(obj: Any, path: str) -> None:
        """
        Crash-safe pickle write: dump to a temp file (fsynced), keep the current
        file as a one-deep ``.bak``, then atomically rename into place. A crash
        mid-write leaves the previous good file (or its .bak) intact rather than
        a truncated/corrupt pickle. Important now that uploads persist live,
        mid-serving.
        """
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            try:
                os.replace(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)

    @staticmethod
    def _load_pickle_with_backup(path: str) -> Any:
        """Load a pickle, falling back to its ``.bak`` if the main file is corrupt."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            bak = path + ".bak"
            if os.path.exists(bak):
                logger.warning(f"{path} unreadable ({e}); falling back to backup {bak}")
                with open(bak, "rb") as f:
                    return pickle.load(f)
            raise

    def _save_indexes(self, retriever: HybridRetriever, chunks: List[Dict[str, Any]], formula_chunks: List[Dict[str, Any]] = None) -> None:
        """
        Save indexes to disk (text + formula). Chunk pickles are written
        atomically with a one-deep backup; the BM25/FAISS binaries are
        rebuildable from the chunks, so they use the retriever's own save.
        """
        try:
            # Save text chunks (atomic + backup)
            self._atomic_pickle(chunks, self.chunks_path)

            # Save formula chunks (atomic + backup)
            if formula_chunks is not None:
                self._atomic_pickle(formula_chunks, self.formula_chunks_path)
                logger.info(f"Saved {len(formula_chunks)} formula chunks")

            # Save indexes (BM25/FAISS — rebuildable from chunks)
            retriever.save(self.index_dir)

            # Save registry
            self.tracker.save()

            logger.info(f"[SAVED] Indexes saved to {self.index_dir}/")

        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
            raise
