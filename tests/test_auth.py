"""API-key auth gate + user-id resolution."""

import pytest
from fastapi import HTTPException

from api import main


def _check(api_key):
    return main.require_api_key(api_key)


def test_local_no_key_is_open(monkeypatch):
    monkeypatch.delenv("PRESCISE_API_KEY", raising=False)
    monkeypatch.setenv("PRESCISE_ENV", "local")
    assert _check(None) is None  # open for local dev / test frontend


def test_key_set_is_enforced(monkeypatch):
    monkeypatch.setenv("PRESCISE_API_KEY", "secret")
    monkeypatch.setenv("PRESCISE_ENV", "local")
    with pytest.raises(HTTPException) as e:
        _check("wrong")
    assert e.value.status_code == 401
    assert _check("secret") is None


def test_nonlocal_without_key_fails_closed(monkeypatch):
    monkeypatch.delenv("PRESCISE_API_KEY", raising=False)
    monkeypatch.setenv("PRESCISE_ENV", "production")
    with pytest.raises(HTTPException) as e:
        _check(None)
    assert e.value.status_code == 500


def test_current_user_id_defaults():
    assert main.current_user_id(None) == "default_user"
    assert main.current_user_id("   ") == "default_user"
    assert main.current_user_id("alice") == "alice"
