# PresciSE Runtime Architecture Map (Source of Truth)

This map reflects the **actual current code path** in the repository.
It is intended to be the implementation-grounded reference for developers.

## 1. Main Entry Point

- CLI demo path: `scripts/pdf_demo.py`
- Core call sequence:
1. `IndexManager.load_or_build()` from `core/persistence/index_manager.py`
2. Query preprocess: `tokenize()` + `Embedder.embed_text()`
3. Retrieval: `HybridRetriever.retrieve_with_router(...)`
4. Answering: `ScientificAnswerAgent.answer(...)`

## 2. End-to-End Data Flow

1. PDFs are discovered in `data/pdfs/`
2. Each PDF is parsed by Docling via `core/ingestion/pdf_loader.py`
3. Parsed docs are chunked by `core/chunking/pdf_chunker.py`
4. Chunk metadata/tokens are generated via `core/nlp/*`
5. Text chunks are embedded by `core/embeddings/embedder.py` (default `allenai/specter`)
6. Formula chunks are extracted by `core/formula/formula_chunker.py` + `core/formula/extractor.py`
7. Formula chunks are embedded and stored separately
8. Indexes are built/loaded:
- BM25 text index
- FAISS text index
- FAISS formula index (optional if formulas exist)
9. At query time, router decides BM25/FAISS weights dynamically
10. Hybrid retrieval merges text and formula results
11. LangGraph agent formats evidence and calls Gemini to produce final cited answer

## 3. Indexing and Persistence

- Orchestrator: `core/persistence/index_manager.py`
- Registry and change detection: `core/persistence/document_tracker.py`
- Stored artifacts in `data/index/`:
- `chunks.pkl`
- `formula_chunks.pkl`
- `document_registry.json`
- `bm25_index.pkl`
- `faiss_index.bin`
- `formula_faiss_index.bin`

Incremental behavior:
- New PDFs: add chunks and index entries
- Modified PDFs: remove old chunks, reprocess, reindex
- Deleted PDFs: remove associated chunks and registry entries

## 4. Retrieval Stack

- Lexical retrieval: `core/retrieval/bm25_retriever.py`
- Semantic retrieval: `core/vectordb/faiss_retriever.py`
- Fusion and ranking: `core/retrieval/hybrid_retriever.py`
- Dynamic routing: `core/retrieval/search_router.py`

Router details:
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Output: JSON with `intent`, `bm25_weight`, `faiss_weight`, `reason`
- Fallback on failure: balanced defaults from router class

Formula retrieval details:
- Formula chunks have dedicated embeddings and FAISS index
- Formula results are threshold-filtered and merged with text results

## 5. LLM/Agent Stack

- Gemini wrapper: `core/llm/gemini_client.py`
- Prompt template: `core/agent/prompts.py`
- Agent workflow: `core/agent/scientific_answer_agent.py`

LangGraph nodes:
1. `format_evidence`: builds cited evidence blocks
2. `generate_answer`: sends formatted prompt to Gemini

## 6. Key Public Interfaces

- `IndexManager.load_or_build() -> HybridRetriever`
- `HybridRetriever.retrieve_with_router(query_tokens, query_embedding, query_str, top_k) -> {"results": ..., "router_decision": ...}`
- `ScientificAnswerAgent.answer(query, retrieved_chunks) -> str`
- `GeminiClient.generate(prompt) -> str`

## 7. Notes on Configuration Reality

- Runtime defaults currently come mostly from code-level constructor defaults.
- `config/retrieval_config.yaml` is a project config reference file, but current primary path does not centrally bind all retrieval parameters from config at runtime.
- If strict config-driven behavior is needed, add a config loader and wire values into `Embedder`, `HybridRetriever`, and router construction.

## 8. Scripts vs Core Boundary

- `core/` contains reusable business logic and architecture-critical components.
- `scripts/` contains demos, diagnostics, and tests that orchestrate core modules.
- New algorithmic or reusable logic should be added in `core/`, not `scripts/`.
