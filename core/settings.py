"""
Central runtime settings for PresciSE — all env-driven with sensible defaults.

This is the single place operational config is resolved, so deployment (incl.
the eventual AURA integration) is configured via environment variables rather
than code edits. Search/retrieval *tuning* still lives with the retriever
defaults; this module covers paths, CORS, limits, and environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


# --- Environment -----------------------------------------------------------
def environment() -> str:
    """Resolved deployment environment. PRESCISE_ENV wins; falls back to
    config/app_config.yaml's app.environment; defaults to 'local'."""
    env = os.getenv("PRESCISE_ENV")
    if env:
        return env.strip().lower()
    try:
        import yaml
        cfg = yaml.safe_load(Path("config/app_config.yaml").read_text())
        return str(cfg.get("app", {}).get("environment", "local")).lower()
    except Exception:
        return "local"


# --- Data paths ------------------------------------------------------------
def data_dir() -> Path:
    return Path(_env("PRESCISE_DATA_DIR", "data"))


def pdf_dir() -> str:
    return str(data_dir() / "pdfs")


def index_dir() -> str:
    return str(data_dir() / "index")


def upload_dir() -> Path:
    return data_dir() / "uploads"


# --- Limits ----------------------------------------------------------------
def max_query_len() -> int:
    return int(_env("PRESCISE_MAX_QUERY_LEN", "512"))


def max_upload_bytes() -> int:
    return int(_env("PRESCISE_MAX_UPLOAD_MB", "50")) * 1024 * 1024


# --- CORS ------------------------------------------------------------------
def cors_origins() -> list[str]:
    """Comma-separated PRESCISE_CORS_ORIGINS; defaults to localhost. Use '*'
    to allow all origins (dev only)."""
    raw = _env(
        "PRESCISE_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]
