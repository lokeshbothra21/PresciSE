"""
Shared pytest setup.

Points the DB at a throwaway temp SQLite file BEFORE any app module is imported
(the engine is created at import time), and gives each test a fresh schema.
These tests are fast and offline — no Gemini, no model downloads.
"""

import os
import pathlib
import tempfile

# Must be set before core.persistence.db is imported anywhere.
_TMP_DB = pathlib.Path(tempfile.gettempdir()) / "prescise_pytest.db"
for _suffix in ("", "-wal", "-shm"):
    _p = pathlib.Path(str(_TMP_DB) + _suffix)
    if _p.exists():
        _p.unlink()
os.environ["PRESCISE_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["PRESCISE_ENV"] = "local"
os.environ["PRESCISE_ENABLE_VLM_BACKGROUND_SCAN"] = "0"
# Force auth open for TestClient tests. Set before api.main's load_dotenv runs;
# load_dotenv won't override an already-set var, so a real .env key can't leak in.
os.environ["PRESCISE_API_KEY"] = ""

import pytest  # noqa: E402
from core.persistence import db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    """Drop + recreate all tables before each test for isolation."""
    db.Base.metadata.drop_all(db._engine)
    db.Base.metadata.create_all(db._engine)
    yield
