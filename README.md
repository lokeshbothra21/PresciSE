# PresciSE — Scientific Document RAG

PresciSE is a **Retrieval-Augmented Generation (RAG)** system for answering scientific questions grounded in a user's own PDFs. The product flow is **upload-and-ask**: a user uploads documents, they are indexed in the background, and questions are answered using **hybrid search (BM25 + FAISS)** plus an **iterative multi-agent LLM loop** (DD+Expert on Gemini). Data is **multi-user and per-user-scoped**, persisted in a SQL database.

The longer-term goal is integration into the **AURA** platform; the auth (`X-API-Key`), identity (`X-User-Id`), `/health`, `X-Request-ID`, and env-driven config are the seams for that.

---

## Table of contents
- [Architecture at a glance](#architecture-at-a-glance)
- [How data is processed (ingestion)](#how-data-is-processed-ingestion)
- [How data is stored](#how-data-is-stored)
- [How retrieval works](#how-retrieval-works)
- [How answers are generated (DD+Expert agent)](#how-answers-are-generated-ddexpert-agent)
- [Multi-user model & auth](#multi-user-model--auth)
- [API endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Setup & running](#setup--running)
- [Testing](#testing)
- [Evaluation (RAGAS)](#evaluation-ragas)
- [Project structure](#project-structure)

---

## Architecture at a glance

```
                    ┌──────────────────────── FastAPI (api/main.py) ────────────────────────┐
 upload PDF ──POST /api/documents──►  save to data/uploads/{user}/  + documents row (pending)
                                      └─► background thread: index_uploaded_pdf()
                                            Docling parse → chunk (900/150) → SPECTER embed
                                            → tag owner_user_id → persist to DB (chunks table)
                                            → hot-add to live BM25 + FAISS  → status = ready

 ask ──POST /api/query (X-User-Id)──►  embed query → rule-based router → hybrid retrieve
                                       (BM25 + FAISS, scoped to {user, "__shared__"})
                                       → DD+Expert LangGraph loop (Gemini) → grounded answer
                                       → store turn in DB (sessions/messages)

 startup ──► IndexManager.load_only(): rebuild BM25 + FAISS in memory FROM THE DB
```

**Key principle:** the **SQL database is the single source of truth** for chunks + embeddings (and users/sessions/messages/documents). **FAISS and BM25 are in-memory caches**, rebuilt from the DB on startup. There is no folder-drop-and-restart step — documents enter only via upload.

Two layers:
- **Background (per upload):** parse → chunk → embed → store → hot-add to the live index.
- **Online (per query):** route → retrieve (per-user) → iterative agent → synthesize.

---

## How data is processed (ingestion)

When a PDF is uploaded (`POST /api/documents`), a background thread runs `IndexManager.index_uploaded_pdf()`:

1. **Parse** — `core/ingestion/` uses **Docling** (`DocumentConverter`) to extract structured text (standard / OCR / VLM modes).
2. **Chunk** — `core/chunking/pdf_chunker.py` splits section text into **~900-character chunks with 150-character overlap**, tokenizing each for BM25.
3. **Embed** — `core/embeddings/embedder.py` wraps **SPECTER** (`allenai/specter`, 768-dim) via `sentence-transformers`. All chunk texts are embedded in one batched call (GPU if available, else CPU). The embedder is lock-guarded for thread-safe concurrent use.
4. **Tag** — every chunk gets `owner_user_id` (the uploader) and a globally-unique `chunk_id` (`{doc_id}_{i}`, `doc_id` being a UUID).
5. **Formula extraction** — `core/formula/` extracts mathematical formulas (regex strategies + normalisation) as `FormulaChunk`s with `formula_text`, `normalized_formula` (ASCII), and `latex_formula` (renderable). These get their own embedded index.
6. **Persist + hot-add** — chunks (text + formula) are written to the DB **and** hot-added to the live in-memory BM25 + FAISS indexes (under locks), so the document is queryable **without a restart**. Status flips `pending → indexing → ready` (or `failed`, with partial chunks rolled back).

On startup, `IndexManager.load_only()` reads all chunks from the DB and rebuilds FAISS + BM25 in memory — so uploaded documents survive restarts.

---

## How data is stored

### SQL database (source of truth)
SQLAlchemy ORM (`core/persistence/db.py`), selected by `PRESCISE_DATABASE_URL`:
- **default:** SQLite at `data/prescise.db`
- **deployment:** PostgreSQL (`postgresql+psycopg://…`) — the same models run unchanged.

| Table | Purpose |
|---|---|
| `users` | user ids |
| `sessions` | chat sessions (`user_id` FK, title, created_at) |
| `messages` | chat turns (`session_id` FK, query, answer, sources JSON, router JSON, ts) |
| `documents` | uploaded docs: `doc_id`, `owner_user_id`, filename, status, n_chunks, error |
| `chunks` | **the chunk store**: `chunk_id`, `doc_id`, `owner_user_id`, `chunk_type` (text/formula), full chunk dict as JSON, and the **embedding as a float32 BLOB** |

### In-memory indexes (rebuildable caches)
- **FAISS** (`IndexFlatIP`, cosine via inner product) — semantic vector search.
- **BM25** (`rank_bm25`) — lexical search over chunk tokens.
- A **separate** formula FAISS+BM25 pair for formula chunks.

These are rebuilt from the DB on startup and hot-updated on upload/delete — never the source of truth. (Legacy `chunks.pkl`/`*.bin` exist only for the legacy `scripts/ask.py` CLI; the API ignores them.)

### Uploaded files
Raw PDFs are saved under `data/uploads/{user_id}/{doc_id}.pdf`. The `data/` directory is gitignored.

---

## How retrieval works

Per query, scoped to `allowed_owners = {user_id, "__shared__"}` (chunks with no owner are treated as shared/global):

1. **Embed + tokenize** the query.
2. **Rule-based router** (`core/retrieval/search_router.py`) — regex/heuristics (NOT an LLM) classify intent (formula lookup, definition, methodology, comparison, …) and choose **BM25 vs FAISS weights**.
3. **Hybrid retrieval** (`core/retrieval/hybrid_retriever.py`):
   - BM25 and FAISS each run; scores normalised to `[0,1]`; combined `score = bm25_weight·bm25 + faiss_weight·faiss`.
   - **Per-user scoping:** results filtered to the user's chunks (+ shared); candidate pool widened when scoping so relevant chunks aren't starved.
   - Weights passed **per-call** (no shared-state mutation → safe under concurrency).
   - Reads snapshot the `(bm25, faiss, chunks)` triple under a lock so a concurrent hot-swap can't be seen half-applied.
4. **Formula retrieval (SR2)** — parallel pass over the formula index, with **same-page co-retrieval** so full equation systems surface together.
5. Merged into a balanced evidence set for the agent.

**Why FAISS + BM25 and not pgvector/Chroma?** The DB holds embeddings as BLOBs and stays the system of record; FAISS/BM25 are rebuilt from it. This keeps hybrid lexical+semantic + formula co-retrieval in one place, with relational per-user filtering in SQL, and no extra vector-DB service to run.

---

## How answers are generated (DD+Expert agent)

`core/agent/dd_expert_agent.py` runs an iterative LangGraph `StateGraph`:

```
dd_plan → retrieve → expert_answer → dd_evaluate
              ↑                            │ (not satisfied & iter < max)
              └────────────────────────────┘
                                           │ (satisfied or iter ≥ max)
                                           ▼
                        formula_plan → retrieve → expert_answer → dd_evaluate → synthesize → END
```

- **dd_plan** — the Director of Decomposition (DD) decomposes the question into sub-questions (or flags out-of-scope).
- **retrieve** — batched-embeds + retrieves top-k chunks per sub-question (scoped by `allowed_owners`).
- **expert_answer** — one batched LLM call answers all sub-questions with citations.
- **dd_evaluate** — DD judges sufficiency and refines sub-questions, looping to a max.
- **formula_plan** — a dedicated formula-retrieval round.
- **synthesize** — DD writes the final grounded answer; formulas emit as `[FORMULA]$latex$[/FORMULA]` for KaTeX.

Budget: ~11–14 Gemini calls/query. **Graceful degradation:** node failures are caught — planning falls back to a single sub-question, synthesis assembles gathered findings, and a top-level guard returns a clean message instead of a 500 on transient LLM/embedder outages.

**LLM client** (`core/llm/gemini_client.py`): Gemini via LangChain, **rate-limited** (`PRESCISE_RATE_LIMIT`, default 60/min), **retried** with backoff (`tenacity`), raises `LLMError` on failure. No key → **mock mode** (surfaced in `/health`).

---

## Multi-user model & auth

- **Identity:** the `X-User-Id` header (default `default_user`); every store call and retrieval is scoped by it. Validated against a strict charset (it's used in filesystem paths).
- **Service auth:** a single `X-API-Key` (`require_api_key`):
  - `PRESCISE_API_KEY` set → enforced everywhere.
  - unset + `PRESCISE_ENV=local` → open (local dev / bundled test UI).
  - unset + non-local → **fail closed**.
- **Note:** `X-User-Id` is currently *trusted*, not authenticated — verifying the user is deferred to AURA, which will forward a verified `X-User-Id`. Today's isolation is logical (correct partitioning) — safe behind the service key / AURA, not as a raw public multi-tenant boundary.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness/readiness (model, DB, index, mock-mode) — unauthenticated |
| POST | `/api/documents` | upload a PDF (multipart) → background index → `{doc_id, status}` |
| GET | `/api/documents` | list the user's documents + indexing status |
| GET | `/api/documents/{id}` | one document's status |
| DELETE | `/api/documents/{id}` | remove a document (chunks + file + row) |
| POST | `/api/query` | ask a question, scoped to the user's docs |
| GET/DELETE | `/api/history[/{id}]` | list / read / delete chat sessions |
| POST | `/api/agent/query` | stateless agent-to-agent endpoint (AURA integration seam) |

All requests echo `X-Request-ID`; per-request `[REQUEST STATS]` logs report `llm_calls`, `llm_time`, `total_time`.

---

## Configuration

Runtime config is **env-driven** via `core/settings.py` (the YAML in `config/` is descriptive only):

```
GEMINI_API_KEY=...                 # required for real answers; absent → mock mode
PRESCISE_API_KEY=...               # service auth; if set, X-API-Key enforced
PRESCISE_ENV=local|production      # non-local + no key → endpoints fail closed
PRESCISE_DATABASE_URL=sqlite:///data/prescise.db   # or postgresql+psycopg://…
PRESCISE_DATA_DIR=data             # base dir for uploads/index
PRESCISE_CORS_ORIGINS=http://localhost:8000,...    # comma-separated; '*' = all
PRESCISE_RATE_LIMIT=60             # Gemini req/min
PRESCISE_MAX_UPLOAD_MB=50
PRESCISE_MAX_QUERY_LEN=512
PRESCISE_ENABLE_VLM_BACKGROUND_SCAN=0|1            # Nougat scan (off; transformers 5.x incompat)
```

See `.env.example` for the full list.

---

## Setup & running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # pinned versions
cp .env.example .env                   # add GEMINI_API_KEY

uvicorn api.main:app --reload          # http://localhost:8000
```

Then: UI → **Upload / manage documents** → upload a PDF → wait for **ready** → ask. Or via API:

```bash
curl -F "file=@paper.pdf" -H "X-User-Id: me" localhost:8000/api/documents
curl -H "X-User-Id: me" localhost:8000/api/documents            # poll until "ready"
curl -X POST localhost:8000/api/query -H "Content-Type: application/json" \
     -H "X-User-Id: me" -d '{"query":"..."}'
```

Requires Python 3.10+, ~4GB+ RAM, and a Gemini API key for real answers.

---

## Testing

Offline pytest suite (no Gemini, no model downloads) — DB scoping, chunk-store round-trip, retriever owner-scoping + hot-add/remove, auth gate, Gemini stats, and an upload→index→isolation→delete integration test:

```bash
python -m pytest
```

---

## Evaluation (RAGAS)

RAG quality is measured with [RAGAS](https://docs.ragas.io) in `evals/`. RAGAS needs an older langchain ecosystem that conflicts with the app's, so scoring runs in an **isolated venv**, decoupled via a predictions file:

```bash
python3 -m venv .venv-ragas && .venv-ragas/bin/pip install -r evals/requirements-ragas.txt

# STEP 1 (main venv): run questions through the agent → predictions file
python evals/generate_predictions.py --testset evals/testset.battery.json --user evaluser
# STEP 2 (eval venv): score with Gemini as judge
.venv-ragas/bin/python evals/score_ragas.py --predictions evals/predictions.jsonl
```

Metrics: faithfulness, answer_relevancy, context_precision (no ground-truth needed); context_recall, answer_correctness (need ground-truth). See `evals/README.md` for the test-set authoring guide.

---

## Project structure

```
core/
  ingestion/    # Docling PDF parsing
  chunking/     # 900/150 text chunking
  embeddings/   # SPECTER embedder (thread-safe)
  formula/      # formula extraction, normalisation, schema
  nlp/          # tokenizer, NER, keyword, query classifier
  retrieval/    # hybrid retriever (per-user scoping, live hot-add), BM25, FAISS, rule-based router
  vectordb/     # FAISS index build/search
  persistence/  # index_manager.py (DB-backed chunk store, upload indexing) + db.py (SQLAlchemy)
  settings.py   # env-driven config
  llm/          # Gemini client (rate limit, retry, stats, LLMError)
  agent/        # DDExpertAgent (LangGraph) + prompts
  utils/        # output formatting

api/            # FastAPI backend (endpoints above)
frontend/       # throwaway test UI (Docs panel for upload; KaTeX) — removed at AURA integration
tests/          # pytest suite (offline)
evals/          # RAGAS harness (isolated venv) + test sets
config/         # descriptive YAML (not loaded at runtime)
data/           # gitignored: prescise.db, uploads/{user}/, (legacy pdfs/, index/)
```

> See `CLAUDE.md` for contributor guidance, design rules, and deeper notes on the formula pipeline.
