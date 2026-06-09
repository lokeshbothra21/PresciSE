"""
Read-only inspector for the PresciSE SQLite/Postgres database.

Usage (from repo root):
    python scripts/inspect_db.py                       # overview: counts + docs + chunks-by-owner
    python scripts/inspect_db.py docs [user]           # list documents (optionally for a user)
    python scripts/inspect_db.py sessions <user>       # list a user's chat sessions
    python scripts/inspect_db.py messages <session_id> # show a session's Q&A turns
    python scripts/inspect_db.py chunks <doc_id> [N]   # show N chunks of a doc (default 5)

Honors PRESCISE_DATABASE_URL (defaults to sqlite:///data/prescise.db).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func, select  # noqa: E402

from core.persistence import db  # noqa: E402


def overview():
    with db.session_scope() as s:
        for name, m in [("users", db.User), ("sessions", db.ChatSession),
                        ("messages", db.Message), ("documents", db.Document),
                        ("chunks", db.Chunk)]:
            n = s.execute(select(func.count()).select_from(m)).scalar()
            print(f"{name:10s}: {n}")
        print("\nchunks by owner:")
        for owner, c in s.execute(
            select(db.Chunk.owner_user_id, func.count()).group_by(db.Chunk.owner_user_id)
        ).all():
            print(f"  {owner}: {c}")


def docs(user=None):
    with db.session_scope() as s:
        q = select(db.Document)
        if user:
            q = q.where(db.Document.owner_user_id == user)
        for d in s.execute(q).scalars().all():
            print(f"{d.doc_id}  owner={d.owner_user_id}  status={d.status}  "
                  f"n_chunks={d.n_chunks}  file={d.filename}")


def sessions(user):
    for srow in db.list_sessions(user):
        print(f"{srow['session_id']}  msgs={srow['message_count']}  "
              f"date={srow['date']}  title={srow['title'][:60]}")


def messages(session_id):
    with db.session_scope() as s:
        sess = s.get(db.ChatSession, session_id)
        if not sess:
            print("session not found"); return
        msgs = db.get_session_messages(sess.user_id, session_id) or []
        for i, m in enumerate(msgs, 1):
            print(f"\n[{i}] Q: {m['query']}")
            print(f"    A: {m['answer'][:300]}")
            print(f"    sources: {len(m['sources'])}")


def chunks(doc_id, n=5):
    with db.session_scope() as s:
        rows = s.execute(
            select(db.Chunk).where(db.Chunk.doc_id == doc_id).limit(int(n))
        ).scalars().all()
        for c in rows:
            text = (c.data_json or {}).get("text", "")[:200]
            print(f"\n{c.chunk_id}  type={c.chunk_type}  owner={c.owner_user_id}  "
                  f"emb={len(c.embedding)//4}d")
            print(f"  {text}")


if __name__ == "__main__":
    db.init_db()
    args = sys.argv[1:]
    cmd = args[0] if args else "overview"
    if cmd == "overview":
        overview()
    elif cmd == "docs":
        docs(args[1] if len(args) > 1 else None)
    elif cmd == "sessions" and len(args) > 1:
        sessions(args[1])
    elif cmd == "messages" and len(args) > 1:
        messages(args[1])
    elif cmd == "chunks" and len(args) > 1:
        chunks(args[1], args[2] if len(args) > 2 else 5)
    else:
        print(__doc__)
