"""DB layer: per-user isolation of chat history, document lifecycle, and the
chunk store (embeddings round-trip through the BLOB column)."""

import pytest

from core.persistence import db


def test_chat_history_is_user_scoped():
    db.add_message("alice", "s1", "q1", "a1", [{"doc_id": "d", "pages": [1]}], {"intent": "def"})
    db.add_message("alice", "s1", "q2", "a2", [], {})
    db.add_message("bob", "s2", "qb", "ab", [], {})

    alice_sessions = db.list_sessions("alice")
    assert [s["session_id"] for s in alice_sessions] == ["s1"]
    assert alice_sessions[0]["message_count"] == 2
    assert [s["session_id"] for s in db.list_sessions("bob")] == ["s2"]

    # Cross-user reads/deletes are blocked.
    assert db.get_session_messages("bob", "s1") is None
    assert db.delete_session("bob", "s1") is False
    assert db.delete_session("alice", "s1") is True
    assert db.list_sessions("alice") == []


def test_message_for_foreign_session_is_rejected():
    db.add_message("alice", "s1", "q", "a", [], {})
    with pytest.raises(PermissionError):
        db.add_message("bob", "s1", "q", "a", [], {})


def test_document_lifecycle_and_scoping():
    db.create_document("alice", "doc1", "paper.pdf")
    db.set_document_status("doc1", db.DOC_READY, n_chunks=5)

    docs = db.list_documents("alice")
    assert len(docs) == 1 and docs[0]["status"] == "ready" and docs[0]["n_chunks"] == 5

    assert db.list_documents("bob") == []
    assert db.get_document("bob", "doc1") is None      # isolation
    assert db.delete_document("bob", "doc1") is False  # isolation
    assert db.delete_document("alice", "doc1") is True
    assert db.list_documents("alice") == []


def test_chunk_store_roundtrip_preserves_embedding():
    chunk = {
        "chunk_id": "c1",
        "doc_id": "doc1",
        "owner_user_id": "alice",
        "text": "hello world",
        "tokens": ["hello", "world"],
        "embedding": [0.5, 0.25, -0.125],  # float32-exact values
        "metadata": {"pages": [1]},
    }
    formula = {"chunk_id": "f1", "doc_id": "doc1", "owner_user_id": "alice",
               "formula_text": "E=mc^2", "embedding": [1.0, 0.0]}
    assert db.add_chunks([chunk], chunk_type="text") == 1
    assert db.add_chunks([formula], chunk_type="formula") == 1

    text_chunks, formula_chunks = db.load_chunks()
    assert len(text_chunks) == 1 and len(formula_chunks) == 1
    assert text_chunks[0]["embedding"] == pytest.approx([0.5, 0.25, -0.125])
    assert text_chunks[0]["text"] == "hello world"
    assert text_chunks[0]["tokens"] == ["hello", "world"]
    assert formula_chunks[0]["formula_text"] == "E=mc^2"


def test_delete_chunks_for_doc():
    db.add_chunks([{"chunk_id": "c1", "doc_id": "d1", "owner_user_id": "a", "embedding": [0.1]}])
    db.add_chunks([{"chunk_id": "c2", "doc_id": "d2", "owner_user_id": "a", "embedding": [0.2]}])
    assert db.count_chunks() == 2
    db.delete_chunks_for_doc("d1")
    assert db.count_chunks() == 1
    text_chunks, _ = db.load_chunks()
    assert text_chunks[0]["chunk_id"] == "c2"
