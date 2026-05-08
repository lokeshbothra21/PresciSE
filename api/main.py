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
import time
import uuid

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

CHAT_HISTORY_DIR = Path("data/chat_history")
MAX_QUERY_LEN = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML components once at startup."""
    global _retriever, _llm, _agent, _embedder

    from core.persistence.index_manager import IndexManager
    from core.llm.gemini_client import GeminiClient
    from core.agent.dd_expert_agent import DDExpertAgent
    from core.agent.context_builder import build_context_description
    from core.embeddings.embedder import Embedder

    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    _embedder = Embedder()
    _retriever = IndexManager("data/pdfs", "data/index").load_or_build()
    _llm = GeminiClient()
    context_desc = build_context_description(getattr(_retriever, "chunks", []))
    _agent = DDExpertAgent(_retriever, _embedder, _llm, context_desc)

    # Launch Nougat background scanner (non-blocking) unless disabled.
    # run_in_executor already submits the function to the thread pool and
    # returns an asyncio.Future — no wrapping needed.
    if os.getenv("PRESCISE_ENABLE_VLM_BACKGROUND_SCAN", "1") == "1":
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_nougat_scan, _retriever, _embedder)

    yield  # application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PresciSE", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def _session_file(session_id: str) -> Path | None:
    """Find the JSONL file for a session_id (searches all date prefixes)."""
    for f in CHAT_HISTORY_DIR.glob("*.jsonl"):
        if session_id in f.name:
            return f
    return None


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


def _run_query(query: str) -> tuple[str, list[dict], dict]:
    """Synchronous ML pipeline — called inside thread pool."""
    from core.nlp.tokenizer import tokenize

    # Retrieve a representative set of chunks for router_decision metadata
    tokens = tokenize(query)
    embedding = _embedder.embed_text(query)
    main_result = _retriever.retrieve_with_router(
        query_tokens=tokens,
        query_embedding=embedding,
        query_str=query,
        top_k=12,
    )
    router_decision = main_result["router_decision"]
    # Use the main result chunks as the source list for the API response
    representative_chunks: list[dict] = list(main_result["results"])

    # Run DD+Expert agent (handles its own retrieval internally).
    # Return the raw answer with [FORMULA]$...$[/FORMULA] blocks intact —
    # render_formula_answer() is for terminal only; the frontend uses KaTeX.
    result = _agent.run(query)
    answer = result["answer"]

    return answer, representative_chunks, router_decision


def _append_to_session(session_id: str, record: dict) -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_path = CHAT_HISTORY_DIR / f"{date_str}_{session_id}.jsonl"
    with open(session_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    scanner = NougatFormulaScanner(embedder, "data/index")
    mgr = IndexManager("data/pdfs", "data/index")

    pdf_dir = Path("data/pdfs")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.info("[VLM SCAN] No PDFs found — skipping Nougat scan.")
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
# Routes
# ---------------------------------------------------------------------------
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if len(req.query) > MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {MAX_QUERY_LEN} chars).",
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    loop = asyncio.get_event_loop()
    try:
        answer, raw_results, router_decision = await loop.run_in_executor(
            None, _run_query, req.query
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [_chunk_to_source(item) for item in raw_results]

    record = {
        "query": req.query,
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "router_decision": router_decision,
        "session_id": session_id,
        "timestamp": timestamp,
    }
    _append_to_session(session_id, record)

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        router_decision=router_decision,
        session_id=session_id,
        timestamp=timestamp,
    )


@app.get("/api/history")
async def list_history():
    """Return summary list of all sessions, newest first."""
    sessions: list[dict] = []
    for f in sorted(CHAT_HISTORY_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            first = json.loads(lines[0])
            # Extract session_id from filename: {date}_{session_id}.jsonl
            stem = f.stem  # e.g. "2026-02-19_abc123"
            sid = stem[len("2026-02-19_"):] if "_" in stem else stem
            # More robust: split on first underscore after date
            parts = stem.split("_", 1)
            date_part = parts[0] if len(parts) > 1 else ""
            sid = parts[1] if len(parts) > 1 else stem
            sessions.append({
                "session_id": sid,
                "date": date_part,
                "title": first.get("query", "")[:80],
                "message_count": len(lines),
            })
        except Exception:
            continue
    return sessions


@app.get("/api/history/{session_id}")
async def get_session(session_id: str):
    """Return all messages for a session."""
    f = _session_file(session_id)
    if f is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/history/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session file."""
    f = _session_file(session_id)
    if f is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    f.unlink(missing_ok=True)


@app.get("/api/vlm_scan_status")
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
# Authentication: set PRESCISE_API_KEY in .env.  If the variable is empty or
# absent, the endpoint is open (useful for local development).  Protect it in
# production by setting a strong random value.
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

_AGENT_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_api_key(api_key: str | None = Security(_AGENT_API_KEY_HEADER)) -> None:
    expected = os.getenv("PRESCISE_API_KEY", "")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


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
    _: None = Depends(_check_api_key),
):
    request_id = req.request_id or str(uuid.uuid4())
    t0 = time.time()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _agent.run(req.query))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
