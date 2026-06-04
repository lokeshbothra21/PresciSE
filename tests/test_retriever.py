"""HybridRetriever: per-user owner scoping + live hot-add/remove. Uses tiny
synthetic embeddings (no SPECTER / model download)."""

from core.retrieval.hybrid_retriever import HybridRetriever


def _chunk(cid, owner, doc, toks, emb):
    return {
        "chunk_id": cid, "doc_id": doc, "owner_user_id": owner,
        "text": " ".join(toks), "tokens": toks, "embedding": emb,
        "metadata": {"pages": [1]},
    }


def _ids(results):
    return sorted(i["chunk"]["chunk_id"] for i in results)


def _base():
    return [
        _chunk("s1", None, "shared", ["enzyme", "kinetics"], [1.0, 0.0, 0.0]),
        _chunk("a1", "alice", "docA", ["lennard", "jones", "potential"], [0.0, 1.0, 0.0]),
    ]


def test_owner_scoping_isolates_users():
    r = HybridRetriever(_base(), formula_chunks=[])
    qt, qe = ["lennard", "jones"], [0.0, 1.0, 0.0]

    # No scope → everything.
    assert _ids(r.retrieve(qt, qe, top_k=10)) == ["a1", "s1"]
    # alice → her doc + shared.
    assert _ids(r.retrieve(qt, qe, top_k=10, allowed_owners={"alice", "__shared__"})) == ["a1", "s1"]
    # bob → only shared (NOT alice's a1).
    assert _ids(r.retrieve(qt, qe, top_k=10, allowed_owners={"bob", "__shared__"})) == ["s1"]


def test_hot_add_and_remove_live():
    r = HybridRetriever(_base(), formula_chunks=[])
    qt, qe = ["lennard", "jones"], [0.0, 1.0, 0.0]

    r.add_text_chunks([_chunk("b1", "bob", "docB", ["lennard", "jones", "potential"], [0.0, 1.0, 0.0])])
    assert _ids(r.retrieve(qt, qe, top_k=10, allowed_owners={"bob", "__shared__"})) == ["b1", "s1"]
    # alice still isolated from bob's new doc.
    assert "b1" not in _ids(r.retrieve(qt, qe, top_k=10, allowed_owners={"alice", "__shared__"}))

    r.remove_document_chunks("docB")
    assert _ids(r.retrieve(qt, qe, top_k=10, allowed_owners={"bob", "__shared__"})) == ["s1"]


def test_empty_retriever_returns_nothing():
    r = HybridRetriever([], formula_chunks=[])
    assert r.retrieve(["anything"], [0.0, 1.0, 0.0], top_k=5) == []
