"""End-to-end upload-and-ask integration test through the real FastAPI app.

Exercises the headline flow: upload PDF → background index → status ready →
per-user isolation → delete. Also checks /health and X-User-Id validation.

Slower than the other tests (boots the app + loads the SPECTER embedder), and
needs pymupdf to synthesize a tiny PDF.
"""

import shutil
import time

import pytest

fitz = pytest.importorskip("fitz")  # pymupdf
from fastapi.testclient import TestClient  # noqa: E402


def _make_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_upload_index_query_isolation_delete(tmp_path):
    import api.main as m

    pdf = tmp_path / "zonk.pdf"
    _make_pdf(pdf, "The Zonktor constant equals 42 in the Plimbo framework. Zonktor is unique.")

    try:
        with TestClient(m.app) as c:
            hA = {"X-User-Id": "alice"}
            hB = {"X-User-Id": "bob"}

            with open(pdf, "rb") as fh:
                r = c.post("/api/documents",
                           files={"file": ("zonk.pdf", fh, "application/pdf")}, headers=hA)
            assert r.status_code == 200, r.text
            doc_id = r.json()["doc_id"]

            # Poll until indexed (no restart needed — live hot-add).
            status = {}
            for _ in range(120):
                status = c.get(f"/api/documents/{doc_id}", headers=hA).json()
                if status["status"] in ("ready", "failed"):
                    break
                time.sleep(0.5)
            assert status["status"] == "ready", status
            assert status["n_chunks"] >= 1

            # Per-user isolation.
            assert c.get("/api/documents", headers=hB).json() == []
            assert c.get(f"/api/documents/{doc_id}", headers=hB).status_code == 404
            assert c.delete(f"/api/documents/{doc_id}", headers=hB).status_code == 404

            # Owner can delete.
            assert c.delete(f"/api/documents/{doc_id}", headers=hA).status_code == 204
            assert c.get("/api/documents", headers=hA).json() == []
    finally:
        shutil.rmtree("data/uploads/alice", ignore_errors=True)


def test_health_and_user_id_validation():
    import api.main as m

    with TestClient(m.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        # Path-traversal-style user id is rejected (400), not used in a path.
        r = c.post("/api/query", json={"query": "hi"}, headers={"X-User-Id": "../etc"})
        assert r.status_code == 400
