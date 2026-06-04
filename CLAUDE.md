# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PresciSE** is a Retrieval-Augmented Generation (RAG) system for answering scientific questions from PDFs using hybrid search (BM25 + FAISS) and LLM-powered synthesis via a DD+Expert iterative multi-agent LangGraph loop.

The product flow is **upload-and-ask**: users upload PDFs via the API, they are indexed in the background, and questions are answered grounded in *that user's* documents. Data is **multi-user and per-user-scoped**, persisted in a SQL database (SQLite by default, PostgreSQL-ready). The end goal is integration into the **AURA** platform — the code's auth (`X-API-Key`), identity (`X-User-Id`), `/health`, `X-Request-ID`, and env-driven config are the seams for that.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GEMINI_API_KEY
```

Requires Python 3.10+, 4GB+ RAM, and a Gemini API key.

## Running

```bash
# FastAPI web server (api/ + frontend/) — the primary entry point.
uvicorn api.main:app --reload
# Then open http://localhost:8000 → "Upload / manage documents" → upload a PDF →
# wait for "ready" → ask. Or via API:
#   curl -F "file=@paper.pdf" -H "X-User-Id: me" localhost:8000/api/documents
#   curl -H "X-User-Id: me" localhost:8000/api/documents          # poll status
#   curl -X POST localhost:8000/api/query -H "Content-Type: application/json" \
#        -H "X-User-Id: me" -d '{"query":"..."}'

# Tests (offline — no Gemini, no model downloads)
python -m pytest

# Legacy CLI runner — NOTE: reads the OLD pickle index in data/index/, NOT the
# DB. Diverges from the API path; useful only for poking the legacy corpus.
python -m scripts.ask "Your question here?"
```

**Endpoints:** `/health`; `POST /api/query`; `POST|GET|DELETE /api/documents`; `GET|DELETE /api/history`; `POST /api/agent/query` (agent-to-agent).

Tests live in `tests/` (pytest). Other scripts in `scripts/` are diagnostics/legacy.

## Architecture

**Phase 1A — Indexing (via upload, live):** A user uploads a PDF (`POST /api/documents`). It is saved under `data/uploads/{user_id}/`, a `documents` row is created (`pending`), and a **background thread** indexes it: Docling parse → 900/150 chunking → SPECTER (768-dim) embedding → formula extraction. Every chunk is tagged with `owner_user_id` and **persisted to the DB `chunks` table** (full chunk dict as JSON + embedding as a float32 BLOB). The chunk is then **hot-added to the live in-memory BM25 + FAISS indexes** (`HybridRetriever.add_text_chunks` / `update_formula_index`, guarded by `_text_lock` / `_formula_lock`) so it's queryable **without a restart**. Document status flips to `ready`/`failed`. The old folder-drop-and-restart model is **retired**; on startup `IndexManager.load_only()` rebuilds FAISS + BM25 in memory **from the DB** (`db.load_chunks()`).

**The DB is the single source of truth** for chunks+embeddings; FAISS/BM25 are rebuildable in-memory caches. `chunks.pkl`/`.bin` are no longer used by the API path (only the legacy `scripts/ask.py`).

**Phase 1B — Retrieval (online, per query):** The query is embedded and routed through a **rule-based router** (`core/retrieval/search_router.py`, regex/heuristics — *not* an LLM; the old Qwen2.5 model was removed) that weights BM25 vs. FAISS by intent. Retrieval is **scoped per user**: callers pass `allowed_owners = {user_id, "__shared__"}`, threaded through `retrieve*()` and the agent, so a user only sees their own documents plus any shared (owner-less) chunks. `DDExpertAgent` runs its own per-subquestion retrieval via `retriever.retrieve_with_formulas(..., allowed_owners=...)`.

**Phase 2 — Answer Generation (DD+Expert LangGraph loop):** `DDExpertAgent` (`core/agent/dd_expert_agent.py`) runs an iterative `StateGraph`:

```
dd_plan → retrieve → expert_answer → dd_evaluate
                          ↑                  │ (not satisfied & iter < 4)
                          └──────────────────┘
                                             │ (satisfied or iter ≥ 4)
                                             ▼
                                         synthesize → END
```

- **dd_plan**: DD reads corpus context, generates 3–6 subquestions, or returns an out-of-scope signal.
- **retrieve**: Embeds + retrieves top-k chunks per subquestion.
- **expert_answer**: Single batched LLM call — Expert answers all subquestions with citations.
- **dd_evaluate**: DD judges whether evidence is sufficient; if not, refines 1–3 subquestions and iterates.
- **synthesize**: DD writes the final grounded answer from all accumulated Q&A pairs.

Max LLM call budget: ~11–14 calls (plan + N×(expert+evaluate) + formula round + synthesize). **Graceful degradation:** node failures are caught — `dd_plan` falls back to a single-subquestion plan, `synthesize` assembles the gathered findings, and `run()` has a top-level guard so a transient LLM/embedder outage returns a clean message instead of a 500.

**LLM client** (`core/llm/gemini_client.py`): rate-limited (`PRESCISE_RATE_LIMIT`, default 60/min — was a hardcoded, *uncalled* 10), retried with backoff (`tenacity`), and **raises `LLMError`** on failure (it no longer silently returns an error string). Per-request stats (`begin_request()`/`request_stats()`) feed the `[REQUEST STATS]` log line (llm_calls, llm_time, total_time).

## Code Structure

```
core/          # All business logic — organized by concern
  ingestion/   # Docling PDF parsing (standard/OCR/VLM modes)
  chunking/    # Text chunking strategies
  embeddings/  # SPECTER embedder wrapper
  formula/     # Formula extraction, chunking, indexing, normalisation
    extractor.py          # Regex-based formula extractor (6 strategies)
    formula_chunker.py    # Docling-path formula chunker
    page_text_extractor.py # PyMuPDF-path formula chunker
    normalizer.py         # ASCII-math normaliser + ascii_math_to_latex()
    formula_schema.py     # FormulaChunk dataclass (includes latex_formula field)
  nlp/         # Tokenization, keywords, NER, query classification
  retrieval/   # Hybrid retriever (per-user scoping, live hot-add), BM25, FAISS, rule-based router
  vectordb/    # FAISS index building and search
  persistence/ # index_manager.py (DB-backed chunk store, upload indexing) + db.py (SQLAlchemy)
  settings.py  # Central env-driven config (paths, CORS, limits, environment)
  llm/         # LangChain Gemini client (rate limit, retry, LLMError, per-request stats)
  agent/       # LangGraph agents, prompts, context builder
    dd_expert_agent.py  # DDExpertAgent — primary answer agent
    dd_prompts.py       # DD+Expert prompt templates
    context_builder.py  # Corpus description for DD planning
    scientific_answer_agent.py  # Legacy single-pass agent (kept for reference)
    prompts.py          # Prompt templates (shared)
    enhancement_prompts.py  # Exploratory/comparative answer prompts
  utils/       # Output formatting (render_formula_answer)

api/           # FastAPI backend: /health, /api/query, /api/documents (upload), /api/history, /api/agent/query
frontend/      # Throwaway test UI (KaTeX; Docs panel for upload). Removed at AURA integration.
tests/         # pytest suite (offline): db scoping, chunk roundtrip, retriever scoping, auth, gemini
scripts/       # Diagnostics + legacy CLI (scripts/ask.py uses the OLD pickle index, not the DB)
config/        # app_config.yaml, retrieval_config.yaml (descriptive; not loaded at runtime)
data/          # gitignored
  uploads/{user_id}/  # uploaded PDFs (the live ingestion source)
  prescise.db         # SQLite: users, sessions, messages, documents, chunks(+embeddings)
  pdfs/, index/       # LEGACY (used only by scripts/ask.py; API ignores these)
```

## Key Design Rules

- **All business logic belongs in `core/`**. Scripts in `scripts/` are strictly for orchestration, demos, and diagnostics.
- The entry point for index orchestration is `core/persistence/index_manager.py`; the SQLAlchemy models + stores are in `core/persistence/db.py`.
- The primary LangGraph agent entry point is `core/agent/dd_expert_agent.py`.
- `DDExpertAgent` requires `retriever` (HybridRetriever), `embedder` (Embedder), and `llm` (GeminiClient), plus a `context_description`. `run(query, allowed_owners=...)` scopes retrieval per user.
- **The DB is the source of truth** for chunks+embeddings (table `chunks`, embedding as BLOB). FAISS/BM25 are in-memory caches rebuilt from the DB on startup — never persist chunks only to pickle.
- **Per-user isolation:** identity comes from the `X-User-Id` header (default `default_user`); every data store call and retrieval is scoped by it. Auth is a single `X-API-Key` (`require_api_key`): enforced when `PRESCISE_API_KEY` is set, open in `local` env when unset, fail-closed in non-local. Real per-user auth is deferred to AURA (it will supply the verified `X-User-Id`).
- Config is read via `core/settings.py` (env vars), not hardcoded paths.

## Formula Pipeline

### Storage format

Each `FormulaChunk` stores:
- `formula_text` — raw text as extracted from PDF
- `normalized_formula` — cleaned ASCII-math (e.g. `E = mc^2`)
- `latex_formula` — renderable LaTeX (e.g. `E = mc^{2}`) produced by `ascii_math_to_latex()`

`_build_display_text()` emits `[FORMULA]$latex$[/FORMULA]` when `latex_formula` is non-empty, which propagates into evidence blocks and LLM prompts.

### Formula notation in prompts / answers

LLM prompts instruct the model to reproduce formulas as `[FORMULA]$exact LaTeX$[/FORMULA]` on their own line. `render_formula_answer()` in `output_formatter.py` passes `$...$` blocks through unchanged (for KaTeX rendering in the browser) and applies legacy Greek→Unicode conversion only to non-LaTeX blocks.

### Env vars (default off in `scripts/ask.py`)

```
PRESCISE_ENABLE_FORMULA_ENRICHMENT=1   # extract/index mathematical formulas
PRESCISE_ENABLE_CODE_ENRICHMENT=1      # extract/index code blocks
PRESCISE_ENABLE_VLM_FALLBACK=1         # use VLM (vision) fallback for hard PDFs
PRESCISE_FORMULA_QUALITY_MODE=balanced # balanced | fast | accurate
```

## Configuration

Runtime config is **env-driven** via `core/settings.py` (the YAML files in `config/` are descriptive only — not loaded at runtime):

```
GEMINI_API_KEY=...                 # required for real answers; absent → silent mock mode
PRESCISE_API_KEY=...               # service auth; if set, X-API-Key is enforced everywhere
PRESCISE_ENV=local|production      # non-local + no API key → endpoints fail closed
PRESCISE_DATABASE_URL=sqlite:///data/prescise.db   # or postgresql+psycopg://… (AURA)
PRESCISE_DATA_DIR=data             # base dir for uploads/index
PRESCISE_CORS_ORIGINS=http://localhost:8000,...    # comma-separated; '*' allows all
PRESCISE_RATE_LIMIT=60             # Gemini req/min (raise on paid tier)
PRESCISE_MAX_UPLOAD_MB=50
PRESCISE_MAX_QUERY_LEN=512
PRESCISE_ENABLE_VLM_BACKGROUND_SCAN=0|1  # Nougat scan; currently no-ops (transformers 5.x incompat)
```

Retrieval tuning (BM25/FAISS weights, top-k, formula thresholds) currently lives in `HybridRetriever` defaults, not config.

## Logging

Uses `loguru`. Set log level in `config/app_config.yaml`. See `HOW_TO_CHECK_LOGS.md` for details on reading formula diagnostics and controlling verbosity.
