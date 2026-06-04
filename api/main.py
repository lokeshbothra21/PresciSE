"""
PresciSE Web API — FastAPI backend.

Endpoints:
  POST /api/query             Run a query through the ML pipeline
  GET  /api/history           List all chat sessions
  GET  /api/history/{id}      Load a specific session
  DELETE /api/history/{id}    Delete a session

Startup: loads all ML singletons once (IndexManager, Embedder, GeminiClient, Agent).
All ML calls are synchronous; they run in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import settings
from core.persistence import db

# ---------------------------------------------------------------------------
# ML singletons (populated during lifespan startup)
# ---------------------------------------------------------------------------
_retriever = None
_llm = None
_agent = None
_embedder = None

# Nougat background scan state — updated by _run_nougat_scan()
_nougat_scan_running: bool = False
_nougat_scan_complete: bool = False

MAX_QUERY_LEN = settings.max_query_len()

# Chunks with no owner are visible to everyone; a query allows {user, shared}.
SHARED_OWNER = "__shared__"

# Uploaded PDFs are stored per user (under the configured data dir).
UPLOAD_DIR = settings.upload_dir()
MAX_UPLOAD_BYTES = settings.max_upload_bytes()
PDF_DIR = settings.pdf_dir()
INDEX_DIR = settings.index_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML components once at startup."""
    global _retriever, _llm, _agent, _embedder

    from core.persistence.index_manager import IndexManager
    from core.llm.gemini_client import GeminiClient
    from core.agent.dd_expert_agent import DDExpertAgent
    from core.agent.context_builder import build_context_description
    from core.embeddings.embedder import Embedder

    db.init_db()  # create tables if missing
    # A crash during a previous run can leave a document stuck in pending/indexing.
    n_stuck = db.reconcile_interrupted_documents()
    if n_stuck:
        from loguru import logger as _log
        _log.warning(f"[STARTUP] Marked {n_stuck} interrupted document(s) as failed.")

    _embedder = Embedder()
    # Load-only startup: the folder-drop-and-restart model is retired; documents
    # now enter via POST /api/documents. This loads any persisted index (incl.
    # previously-uploaded docs) without scanning data/pdfs or rebuilding.
    _retriever = IndexManager(PDF_DIR, INDEX_DIR).load_only()
    _llm = GeminiClient()
    context_desc = build_context_description(getattr(_retriever, "chunks", []))
    _agent = DDExpertAgent(_retriever, _embedder, _llm, context_desc)

    # Nougat background scanner — OFF by default. It scans the legacy data/pdfs/
    # folder (not uploads), persists to pickle (not the DB source of truth), and
    # nougat-ocr is incompatible with transformers 5.x — so it's disabled unless
    # explicitly enabled. Opt in with PRESCISE_ENABLE_VLM_BACKGROUND_SCAN=1.
    if os.getenv("PRESCISE_ENABLE_VLM_BACKGROUND_SCAN", "0") == "1":
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_nougat_scan, _retriever, _embedder)

    yield  # application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PresciSE", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Correlation id: echo an incoming X-Request-ID or mint one, and return it
    on the response. AURA propagates X-Request-ID, so this threads tracing
    through cleanly once integrated."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class SourceItem(BaseModel):
    doc_id: str
    pages: list[int]
    section_type: str
    score: float
    text: str
    is_formula: bool


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    router_decision: dict[str, Any]
    session_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chunk_to_source(item: dict) -> SourceItem:
    chunk = item["chunk"]
    meta = chunk.get("metadata") or {}

    pages_raw = meta.get("pages") or []
    if not pages_raw:
        pn = chunk.get("page_number")
        pages_raw = [pn] if pn is not None else []
    pages = [int(p) for p in pages_raw if p is not None]

    return SourceItem(
        doc_id=chunk.get("doc_id", "unknown"),
        pages=pages,
        section_type=chunk.get("section_type", meta.get("section_type", "text")),
        score=round(float(item.get("score", 0.0)), 4),
        text=chunk.get("text", "")[:300],
        is_formula=chunk.get("chunk_type") == "formula",
    )


def _run_query(query: str, allowed_owners: set | None = None, request_id: str = "-") -> tuple[str, list[dict], dict]:
    """Synchronous ML pipeline — called inside thread pool.

    allowed_owners scopes retrieval to a user's documents (+ shared). None =
    no scoping (returns everything — used only when user-scoping is disabled).
    """
    from core.nlp.tokenizer import tokenize
    from loguru import logger

    t0 = time.time()
    _llm.begin_request()  # reset per-request LLM stats for this thread

    # Retrieve a representative set of chunks for router_decision metadata.
    # Best-effort: this is only for the response's sources/router metadata, so a
    # hiccup here must not fail the whole request — the agent does its own
    # retrieval internally.
    router_decision: dict = {}
    representative_chunks: list[dict] = []
    try:
        tokens = tokenize(query)
        embedding = _embedder.embed_text(query)
        main_result = _retriever.retrieve_with_router(
            query_tokens=tokens,
            query_embedding=embedding,
            query_str=query,
            top_k=12,
            allowed_owners=allowed_owners,
        )
        router_decision = main_result["router_decision"]
        representative_chunks = list(main_result["results"])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[QUERY] representative retrieval failed (non-fatal): {exc}")

    # Run DD+Expert agent (handles its own retrieval internally).
    # Return the raw answer with [FORMULA]$...$[/FORMULA] blocks intact —
    # render_formula_answer() is for terminal only; the frontend uses KaTeX.
    result = _agent.run(query, allowed_owners=allowed_owners)
    answer = result["answer"]

    # Per-request summary: how many LLM calls and how long it all took.
    stats = _llm.request_stats()
    total_s = round(time.time() - t0, 2)
    logger.info(
        f"[REQUEST STATS] request_id={request_id} | llm_calls={stats['llm_calls']} | "
        f"llm_time={stats['llm_time_s']}s | total_time={total_s}s "
        f"(non-llm={round(total_s - stats['llm_time_s'], 2)}s) | "
        f"query={query[:60]!r}"
    )

    return answer, representative_chunks, router_decision


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
# A single API-key gate protects every data endpoint. Behaviour:
#   * PRESCISE_API_KEY set            -> the key is enforced on every request
#                                        (callers send  X-API-Key: <key>).
#   * key unset + environment local   -> open (so the bundled test frontend
#                                        works during local development).
#   * key unset + environment != local-> fail closed (refuse all requests) so
#                                        a deployment can never run unprotected.
# Environment is read from PRESCISE_ENV, falling back to app_config.yaml's
# app.environment, defaulting to "local".
# ---------------------------------------------------------------------------
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    expected = os.getenv("PRESCISE_API_KEY", "").strip()
    if not expected:
        if settings.environment() != "local":
            raise HTTPException(
                status_code=500,
                detail="Server auth misconfigured: PRESCISE_API_KEY must be set outside the local environment.",
            )
        return  # local dev with no key configured: endpoints are open
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# Per-user identity. Today it comes from an X-User-Id header (defaulting to a
# single shared user for local testing); when AURA integrates, AURA forwards the
# authenticated user id here. All chat history is scoped by this value.
_USER_ID_HEADER = APIKeyHeader(name="X-User-Id", auto_error=False)


_USER_ID_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


def current_user_id(user_id: str | None = Security(_USER_ID_HEADER)) -> str:
    uid = (user_id or "").strip()
    if not uid:
        return "default_user"
    # user_id is used in filesystem paths (data/uploads/{user_id}/…), so it must
    # be a strict, traversal-safe token. Reject anything with '/', '..', etc.
    if not _USER_ID_RE.match(uid):
        raise HTTPException(status_code=400, detail="Invalid X-User-Id.")
    return uid


# ---------------------------------------------------------------------------
# Nougat background scanner
# ---------------------------------------------------------------------------

def _run_nougat_scan(retriever, embedder) -> None:
    """
    Background worker — scans each unscanned PDF with Nougat once.

    Runs in a thread-pool executor so it does not block the asyncio event loop.
    After each PDF is scanned the formula index is hot-swapped via
    HybridRetriever.update_formula_index() and persisted to disk.
    """
    global _nougat_scan_running, _nougat_scan_complete
    _nougat_scan_running = True

    from loguru import logger
    from core.formula.nougat_scanner import NougatFormulaScanner
    from core.persistence.index_manager import IndexManager

    scanner = NougatFormulaScanner(embedder, INDEX_DIR)
    mgr = IndexManager(PDF_DIR, INDEX_DIR)

    pdf_dir = Path(PDF_DIR)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.info("[VLM SCAN] No PDFs found — skipping Nougat scan.")
        _nougat_scan_running = False
        _nougat_scan_complete = True
        return

    total_added = 0
    total_t0 = time.time()

    for pdf_path in pdfs:
        doc_id = mgr._derive_doc_id(pdf_path)
        if not scanner.needs_scan(doc_id):
            logger.debug(f"[VLM SCAN] {doc_id}: already scanned, skipping.")
            continue

        prev_count = sum(
            1 for fc in retriever.formula_chunks
            if fc.get("doc_id") == doc_id
        )

        # Split existing chunks into this-doc and others for scan_pdf
        existing_doc = [
            fc for fc in retriever.formula_chunks
            if fc.get("doc_id") == doc_id
        ]
        others = [
            fc for fc in retriever.formula_chunks
            if fc.get("doc_id") != doc_id
        ]

        try:
            result = scanner.scan_pdf(str(pdf_path), doc_id, existing_doc)
        except Exception as exc:
            logger.error(f"[VLM SCAN] {doc_id}: scan failed: {exc}")
            continue

        # Reconstruct the full formula list and hot-swap
        new_all = others + result["merged"]
        retriever.update_formula_index(new_all)
        mgr.save_formula_chunks(new_all, retriever=retriever)

        scanner.mark_scanned(
            doc_id,
            result["pages_scanned"],
            result["added"],
            result["replaced"],
            result["elapsed_s"],
        )
        total_added += result["added"]

        logger.info(
            f"[VLM SCAN] {doc_id}: "
            f"{result['pages_scanned']} pages | "
            f"{result['elapsed_s']:.1f}s | "
            f"formulas: {prev_count} → {len(result['merged'])} "
            f"(+{result['added']} new, {result['replaced']} replaced)"
        )

    logger.info(
        f"[VLM SCAN] ✅ All docs scanned. "
        f"Total time: {time.time() - total_t0:.1f}s. "
        f"Total formulas added: {total_added}. "
        f"Index size: {len(retriever.formula_chunks)}"
    )

    _nougat_scan_running = False
    _nougat_scan_complete = True


# ---------------------------------------------------------------------------
# Document upload — background indexing jobs (run in the thread pool)
# ---------------------------------------------------------------------------
def _index_document_job(pdf_path: str, doc_id: str, user_id: str) -> None:
    """Parse/chunk/embed an uploaded PDF and hot-add it to the live index."""
    from loguru import logger
    from core.persistence.index_manager import IndexManager

    db.set_document_status(doc_id, db.DOC_INDEXING)
    try:
        mgr = IndexManager(PDF_DIR, INDEX_DIR)
        n = mgr.index_uploaded_pdf(pdf_path, doc_id, user_id, _embedder, _retriever)
        db.set_document_status(doc_id, db.DOC_READY, n_chunks=n)
        logger.info(f"[UPLOAD] {doc_id} ready ({n} chunks) for user {user_id}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[UPLOAD] Indexing failed for {doc_id}: {exc}")
        # Roll back any chunks already added (index + DB) so a 'failed' doc can't
        # leave orphan chunks live/queryable.
        try:
            IndexManager(PDF_DIR, INDEX_DIR).remove_document_from_index(doc_id, _retriever)
        except Exception:  # noqa: BLE001
            pass
        db.set_document_status(doc_id, db.DOC_FAILED, error=str(exc))


def _remove_document_job(doc_id: str) -> None:
    from core.persistence.index_manager import IndexManager
    IndexManager(PDF_DIR, INDEX_DIR).remove_document_from_index(doc_id, _retriever)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness/readiness probe (unauthenticated). Reports model/index/DB state."""
    try:
        db_chunks = db.count_chunks()
        db_ok = True
    except Exception:
        db_chunks, db_ok = None, False
    retr = _retriever
    return {
        "status": "ok",
        "environment": settings.environment(),
        "embedder_initialized": _embedder is not None,
        "llm_mock_mode": bool(getattr(_llm, "is_mock", False)),
        "db_ok": db_ok,
        "db_chunks": db_chunks,
        "index_text_chunks": len(getattr(retr, "chunks", []) or []),
        "index_formula_chunks": len(getattr(retr, "formula_chunks", []) or []),
    }


@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query_endpoint(req: QueryRequest, request: Request, user_id: str = Depends(current_user_id)):
    if len(req.query) > MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {MAX_QUERY_LEN} chars).",
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    request_id = getattr(request.state, "request_id", "-")

    allowed_owners = {user_id, SHARED_OWNER}
    loop = asyncio.get_event_loop()
    try:
        answer, raw_results, router_decision = await loop.run_in_executor(
            None, _run_query, req.query, allowed_owners, request_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [_chunk_to_source(item) for item in raw_results]

    try:
        db.add_message(
            user_id=user_id,
            session_id=session_id,
            query=req.query,
            answer=answer,
            sources=[s.model_dump() for s in sources],
            router=router_decision,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Session belongs to another user.")

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        router_decision=router_decision,
        session_id=session_id,
        timestamp=timestamp,
    )


@app.get("/api/history", dependencies=[Depends(require_api_key)])
async def list_history(user_id: str = Depends(current_user_id)):
    """Return summary list of the current user's sessions, newest first."""
    return db.list_sessions(user_id)


@app.get("/api/history/{session_id}", dependencies=[Depends(require_api_key)])
async def get_session(session_id: str, user_id: str = Depends(current_user_id)):
    """Return all messages for one of the current user's sessions."""
    messages = db.get_session_messages(user_id, session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return messages


@app.delete("/api/history/{session_id}", status_code=204, dependencies=[Depends(require_api_key)])
async def delete_session(session_id: str, user_id: str = Depends(current_user_id)):
    """Delete one of the current user's sessions."""
    if not db.delete_session(user_id, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


# ---------------------------------------------------------------------------
# Document upload / management (the core "upload your docs and ask" flow)
# ---------------------------------------------------------------------------
@app.post("/api/documents", dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(current_user_id),
):
    """Upload a PDF; it is indexed in the background and becomes queryable when
    its status reaches 'ready' (poll GET /api/documents/{doc_id})."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = uuid.uuid4().hex
    user_dir = UPLOAD_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / f"{doc_id}.pdf"

    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    db.create_document(user_id, doc_id, file.filename)

    # Index in the background so the request returns immediately.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _index_document_job, str(dest), doc_id, user_id)

    return {"doc_id": doc_id, "filename": file.filename, "status": db.DOC_PENDING}


@app.get("/api/documents", dependencies=[Depends(require_api_key)])
async def list_documents(user_id: str = Depends(current_user_id)):
    """List the current user's documents and their indexing status."""
    return db.list_documents(user_id)


@app.get("/api/documents/{doc_id}", dependencies=[Depends(require_api_key)])
async def get_document(doc_id: str, user_id: str = Depends(current_user_id)):
    """Status/progress for one of the user's documents (for the 'indexing…' UI)."""
    doc = db.get_document(user_id, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/api/documents/{doc_id}", status_code=204, dependencies=[Depends(require_api_key)])
async def delete_document(doc_id: str, user_id: str = Depends(current_user_id)):
    """Delete a document: remove its chunks from the index, its file, and its row."""
    doc = db.get_document(user_id, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _remove_document_job, doc_id)
    except Exception as exc:  # noqa: BLE001
        # Index removal failing must not block deleting the row + file.
        from loguru import logger
        logger.error(f"[DELETE] index removal failed for {doc_id}: {exc}")

    db.delete_document(user_id, doc_id)
    (UPLOAD_DIR / user_id / f"{doc_id}.pdf").unlink(missing_ok=True)


@app.get("/api/vlm_scan_status", dependencies=[Depends(require_api_key)])
async def vlm_scan_status():
    """Return the Nougat VLM scan status, including running/complete flags."""
    p = Path("data/index/vlm_scan_status.json")
    docs: dict = {}
    if p.exists():
        try:
            docs = json.loads(p.read_text())
        except Exception:
            pass
    return {
        "status": "ok",
        "is_running": _nougat_scan_running,
        "is_complete": _nougat_scan_complete,
        "docs": docs,
    }


# ---------------------------------------------------------------------------
# Agent-to-Agent API  (/api/agent/query)
# ---------------------------------------------------------------------------
# Authentication: shared with all data endpoints via require_api_key (see the
# Authentication section above). The key is enforced whenever PRESCISE_API_KEY
# is set, and is mandatory outside the local environment.
#
# Callers pass the key as:   X-API-Key: <your-key>
#
# Response format is stable and designed for machine consumption:
#   - answer         plain text with optional [FORMULA]$LaTeX$[/FORMULA] blocks
#   - sources        deduplicated, sorted by relevance — the actual chunks the
#                    DDExpertAgent retrieved across all its internal iterations
#   - request_id     echoed from the request (or a fresh UUID if omitted)
#   - processing_time_s  wall-clock seconds for the full pipeline
# ---------------------------------------------------------------------------

class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512, description="The scientific question to answer.")
    request_id: str | None = Field(None, description="Optional caller-supplied ID echoed back in the response.")


class AgentSource(BaseModel):
    doc_id: str = Field(description="Document identifier (PDF filename without extension).")
    pages: list[int] = Field(description="Page numbers within that document (0-indexed).")
    snippet: str = Field(description="Short excerpt from the source chunk (≤200 chars).")
    is_formula: bool = Field(description="True if this source is a mathematical formula chunk.")
    relevance_score: float = Field(description="Combined BM25+FAISS score; higher is more relevant.")


class AgentQueryResponse(BaseModel):
    answer: str = Field(description="Final synthesised answer. May contain [FORMULA]$LaTeX$[/FORMULA] blocks.")
    sources: list[AgentSource] = Field(description="Deduplicated sources used by the agent, sorted by relevance.")
    request_id: str = Field(description="Echoed from the request, or a fresh UUID.")
    processing_time_s: float = Field(description="Wall-clock time for the full pipeline.")


def _extract_agent_sources(all_retrieved_chunks: list[dict]) -> list[AgentSource]:
    """
    Deduplicate by (doc_id, first_page) and return up to 10 sources sorted by
    descending relevance score.  Only the highest-scoring occurrence of each
    (doc_id, page) pair is kept.
    """
    best: dict[tuple, dict] = {}   # (doc_id, page) → best item

    for item in all_retrieved_chunks:
        chunk = item.get("chunk", {})
        doc_id = chunk.get("doc_id", "unknown")
        meta = chunk.get("metadata") or {}
        pages = meta.get("pages") or []
        if not pages:
            pn = chunk.get("page_number")
            pages = [pn] if pn is not None else [0]
        pages = [int(p) for p in pages if p is not None]

        key = (doc_id, pages[0] if pages else 0)
        score = float(item.get("score", 0.0))
        if key not in best or score > float(best[key].get("score", 0.0)):
            best[key] = {**item, "_pages": pages}

    sources: list[AgentSource] = []
    for item in sorted(best.values(), key=lambda x: -float(x.get("score", 0.0))):
        chunk = item.get("chunk", {})
        pages = item["_pages"]
        is_formula = chunk.get("chunk_type") == "formula"

        if is_formula:
            latex = chunk.get("latex_formula", "")
            snippet = latex[:200] if latex else chunk.get("text", "")[:200]
        else:
            snippet = chunk.get("text", "")[:200]

        sources.append(AgentSource(
            doc_id=chunk.get("doc_id", "unknown"),
            pages=pages,
            snippet=snippet.strip(),
            is_formula=is_formula,
            relevance_score=round(float(item.get("score", 0.0)), 4),
        ))

    return sources[:10]


@app.post(
    "/api/agent/query",
    response_model=AgentQueryResponse,
    summary="Agent-to-agent query endpoint",
    description=(
        "Submit a scientific query and receive the synthesised answer plus the "
        "actual sources the agent retrieved. Authenticate with X-API-Key header "
        "(set PRESCISE_API_KEY in .env; omit the env var to disable auth for local dev)."
    ),
)
async def agent_query(
    req: AgentQueryRequest,
    _: None = Depends(require_api_key),
):
    from loguru import logger

    request_id = req.request_id or str(uuid.uuid4())
    t0 = time.time()

    def _run() -> tuple[dict, dict]:
        # begin_request + read stats must run in the same worker thread.
        _llm.begin_request()
        res = _agent.run(req.query)
        return res, _llm.request_stats()

    loop = asyncio.get_event_loop()
    try:
        result, stats = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_s = round(time.time() - t0, 2)
    logger.info(
        f"[REQUEST STATS] request_id={request_id} | llm_calls={stats['llm_calls']} | "
        f"llm_time={stats['llm_time_s']}s | total_time={total_s}s | query={req.query[:60]!r}"
    )

    sources = _extract_agent_sources(result.get("all_retrieved_chunks", []))

    return AgentQueryResponse(
        answer=result["answer"],
        sources=sources,
        request_id=request_id,
        processing_time_s=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Static files — MUST be mounted LAST (catches everything else)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
