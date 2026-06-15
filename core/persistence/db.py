"""
SQLAlchemy persistence for PresciSE.

Synchronous SQLAlchemy 2.0 ORM. The engine is selected by PRESCISE_DATABASE_URL:
  - default ``sqlite:///data/prescise.db``  (local / dev)
  - set a ``postgresql+psycopg://…`` URL in deployment — the same models run
    unchanged (this matches AURA's stack, so the eventual migration is a
    connection-string change, not a rewrite).

All chat-history access is scoped by ``user_id`` so one user can never read or
delete another user's sessions. ``user_id`` is supplied by the API layer (today
from an ``X-User-Id`` header / default; later from AURA's authenticated JWT).

Document/chunk tables (the system-of-record for the retrieval index) are added
in a later step — this module currently covers users, sessions and messages.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    LargeBinary,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
    update,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[Any] = mapped_column(JSON, default=list)
    router_json: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# Indexing-status values for Document.status
DOC_PENDING = "pending"
DOC_INDEXING = "indexing"
DOC_READY = "ready"
DOC_FAILED = "failed"


class Document(Base):
    """An uploaded document, owned by a user. The actual chunks/vectors live in
    the FAISS+BM25 index (tagged with owner_user_id); this row is the ownership
    + lifecycle record used by the upload/list/delete endpoints."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(128), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default=DOC_PENDING)
    n_chunks: Mapped[int] = mapped_column(default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Chunk(Base):
    """The system-of-record for an indexed chunk (text or formula). The full
    chunk dict (minus the embedding) is stored as JSON; the embedding is stored
    as a compact float32 BLOB. FAISS + BM25 are rebuilt from these rows on
    startup, so this table — not any pickle — is the source of truth."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), index=True)
    owner_user_id: Mapped[str] = mapped_column(String(128), index=True, default="")
    chunk_type: Mapped[str] = mapped_column(String(16), default="text")  # text | formula
    data_json: Mapped[Any] = mapped_column(JSON)        # chunk dict WITHOUT embedding
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # float32 bytes


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------
_DB_URL = os.getenv("PRESCISE_DATABASE_URL", "sqlite:///data/prescise.db")


def _make_engine(url: str):
    if url.startswith("sqlite"):
        if url.startswith("sqlite:///"):
            db_path = url.replace("sqlite:///", "", 1)
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url, connect_args={"check_same_thread": False}, future=True
        )

        # WAL improves concurrent read/write on SQLite (FastAPI serves requests
        # from multiple threads); enforce FKs so user-scoping is honoured.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine
    return create_engine(url, future=True, pool_pre_ping=True)


_engine = _make_engine(_DB_URL)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commit on success, rollback on error, always close."""
    s = _SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _ensure_user(s: Session, user_id: str) -> None:
    if s.get(User, user_id) is None:
        s.add(User(id=user_id))
        s.flush()


# ---------------------------------------------------------------------------
# Chat-history store (all functions are user-scoped)
# ---------------------------------------------------------------------------
def add_message(
    user_id: str,
    session_id: str,
    query: str,
    answer: str,
    sources: list[dict],
    router: dict,
) -> None:
    """Append one Q&A turn to a session, creating the session/user as needed."""
    with session_scope() as s:
        _ensure_user(s, user_id)
        sess = s.get(ChatSession, session_id)
        if sess is None:
            s.add(ChatSession(id=session_id, user_id=user_id, title=query[:200]))
            s.flush()
        elif sess.user_id != user_id:
            raise PermissionError("Session belongs to another user.")
        s.add(
            Message(
                session_id=session_id,
                user_id=user_id,
                query=query,
                answer=answer,
                sources_json=sources,
                router_json=router,
            )
        )


def list_sessions(user_id: str) -> list[dict]:
    """All of a user's sessions, newest first, with message counts."""
    with session_scope() as s:
        rows = s.execute(
            select(
                ChatSession.id,
                ChatSession.title,
                ChatSession.created_at,
                func.count(Message.id),
            )
            .join(Message, Message.session_id == ChatSession.id, isouter=True)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id, ChatSession.title, ChatSession.created_at)
            .order_by(ChatSession.created_at.desc())
        ).all()
        return [
            {
                "session_id": r[0],
                "title": r[1],
                "date": r[2].strftime("%Y-%m-%d") if r[2] else "",
                "message_count": r[3],
            }
            for r in rows
        ]


def get_session_messages(user_id: str, session_id: str) -> list[dict] | None:
    """All messages for a session, or None if it doesn't exist / isn't the user's."""
    with session_scope() as s:
        sess = s.get(ChatSession, session_id)
        if sess is None or sess.user_id != user_id:
            return None
        msgs = (
            s.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id)
            )
            .scalars()
            .all()
        )
        return [
            {
                "query": m.query,
                "answer": m.answer,
                "sources": m.sources_json or [],
                "router_decision": m.router_json or {},
                "session_id": session_id,
                "timestamp": m.created_at.isoformat() if m.created_at else "",
            }
            for m in msgs
        ]


def delete_session(user_id: str, session_id: str) -> bool:
    """Delete a session + its messages. Returns False if not found / not owned."""
    with session_scope() as s:
        sess = s.get(ChatSession, session_id)
        if sess is None or sess.user_id != user_id:
            return False
        s.execute(delete(Message).where(Message.session_id == session_id))
        s.delete(sess)
        return True


# ---------------------------------------------------------------------------
# Document store (ownership + indexing lifecycle; all user-scoped)
# ---------------------------------------------------------------------------
def _doc_to_dict(d: "Document") -> dict:
    return {
        "doc_id": d.doc_id,
        "filename": d.filename,
        "status": d.status,
        "n_chunks": d.n_chunks,
        "error": d.error,
        "created_at": d.created_at.isoformat() if d.created_at else "",
        "updated_at": d.updated_at.isoformat() if d.updated_at else "",
    }


def create_document(user_id: str, doc_id: str, filename: str, sha256: str = "") -> dict:
    with session_scope() as s:
        _ensure_user(s, user_id)
        doc = Document(
            doc_id=doc_id,
            owner_user_id=user_id,
            filename=filename,
            sha256=sha256,
            status=DOC_PENDING,
        )
        s.add(doc)
        s.flush()
        return _doc_to_dict(doc)


def set_document_status(
    doc_id: str,
    status: str,
    n_chunks: int | None = None,
    error: str | None = None,
) -> None:
    """Update a document's indexing status (called by the background indexer)."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return
        doc.status = status
        if n_chunks is not None:
            doc.n_chunks = n_chunks
        if error is not None:
            doc.error = error[:2000]


def reconcile_interrupted_documents() -> int:
    """Mark any documents stuck in pending/indexing as failed. Called at startup
    so a crash mid-index doesn't leave a document spinning forever."""
    with session_scope() as s:
        result = s.execute(
            update(Document)
            .where(Document.status.in_([DOC_PENDING, DOC_INDEXING]))
            .values(status=DOC_FAILED, error="Interrupted (server restarted during indexing).")
        )
        return result.rowcount or 0


def list_documents(user_id: str) -> list[dict]:
    with session_scope() as s:
        docs = (
            s.execute(
                select(Document)
                .where(Document.owner_user_id == user_id)
                .order_by(Document.created_at.desc())
            )
            .scalars()
            .all()
        )
        return [_doc_to_dict(d) for d in docs]


def get_document(user_id: str, doc_id: str) -> dict | None:
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None or doc.owner_user_id != user_id:
            return None
        return _doc_to_dict(doc)


def delete_document(user_id: str, doc_id: str) -> bool:
    """Delete the document row. Returns False if not found / not owned. (Callers
    are responsible for removing the doc's chunks from the index + its file.)"""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None or doc.owner_user_id != user_id:
            return False
        s.delete(doc)
        return True


# ---------------------------------------------------------------------------
# Chunk store (source of truth for the retrieval index; embeddings as BLOBs)
# ---------------------------------------------------------------------------
def _encode_embedding(emb) -> bytes:
    if emb is None:
        return b""
    return np.asarray(emb, dtype=np.float32).tobytes()


def _decode_embedding(buf: bytes):
    if not buf:
        return None
    return np.frombuffer(buf, dtype=np.float32).tolist()


def add_chunks(chunk_dicts: list[dict], chunk_type: str = "text") -> int:
    """Upsert chunks (text or formula). The embedding is split out into the BLOB
    column; everything else is stored as JSON. Returns the number written."""
    n = 0
    with session_scope() as s:
        for c in chunk_dicts:
            data = {k: v for k, v in c.items() if k != "embedding"}
            s.merge(
                Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c.get("doc_id", ""),
                    owner_user_id=c.get("owner_user_id") or "",
                    chunk_type=chunk_type,
                    data_json=data,
                    embedding=_encode_embedding(c.get("embedding")),
                )
            )
            n += 1
    return n


def load_chunks() -> tuple[list[dict], list[dict]]:
    """Return (text_chunks, formula_chunks) as full dicts with embeddings,
    reconstructed from the DB. Used at startup to (re)build FAISS + BM25."""
    text_chunks: list[dict] = []
    formula_chunks: list[dict] = []
    with session_scope() as s:
        rows = s.execute(select(Chunk)).scalars().all()
        for r in rows:
            d = dict(r.data_json or {})
            emb = _decode_embedding(r.embedding)
            if emb is not None:
                d["embedding"] = emb
            # Re-inject chunk_type so downstream formula detection (is_formula,
            # evidence formatting) keeps working after a reload.
            if r.chunk_type == "formula":
                d.setdefault("chunk_type", "formula")
                formula_chunks.append(d)
            else:
                text_chunks.append(d)
    return text_chunks, formula_chunks


def delete_chunks_for_doc(doc_id: str) -> None:
    with session_scope() as s:
        s.execute(delete(Chunk).where(Chunk.doc_id == doc_id))


def all_chunks_for_reembed() -> list[tuple[str, str, dict]]:
    """Return (chunk_id, chunk_type, data_json) for every chunk (no embeddings).

    Used by the re-embed migration (scripts/reembed.py) when the embedding model
    changes — only the text inputs are needed to recompute vectors.
    """
    with session_scope() as s:
        rows = s.execute(
            select(Chunk.chunk_id, Chunk.chunk_type, Chunk.data_json)
        ).all()
        return [(r[0], r[1], dict(r[2] or {})) for r in rows]


def update_chunk_embeddings(items: dict) -> int:
    """Overwrite stored embeddings by chunk_id. ``items`` maps chunk_id -> vector.

    The new vectors may have a different dimension than the old ones (e.g. after
    SPECTER 768 -> gemini-embedding-001 3072); FAISS/BM25 rebuild from these rows
    on the next startup, adopting the new dimension. Returns the number updated.
    """
    n = 0
    with session_scope() as s:
        for chunk_id, emb in items.items():
            row = s.get(Chunk, chunk_id)
            if row is not None:
                row.embedding = _encode_embedding(emb)
                n += 1
    return n


def count_chunks() -> int:
    with session_scope() as s:
        return s.execute(select(func.count(Chunk.chunk_id))).scalar() or 0


def count_chunks_for_owner(owner_user_id: str) -> int:
    with session_scope() as s:
        return (
            s.execute(
                select(func.count(Chunk.chunk_id)).where(Chunk.owner_user_id == owner_user_id)
            ).scalar()
            or 0
        )
