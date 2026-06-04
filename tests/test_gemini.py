"""GeminiClient: mock mode, per-request stats, configurable rate limit.
No network — runs entirely in mock mode."""

from core.llm import gemini_client


def _mock_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return gemini_client.GeminiClient()


def test_mock_mode_when_no_key(monkeypatch):
    c = _mock_client(monkeypatch)
    assert c.is_mock is True
    assert "MOCK" in c.generate("hello")


def test_request_stats_counts_calls(monkeypatch):
    c = _mock_client(monkeypatch)
    c.begin_request()
    for _ in range(3):
        c.generate("x")
    stats = c.request_stats()
    assert stats["llm_calls"] == 3
    assert "llm_time_s" in stats


def test_rate_limit_is_configurable(monkeypatch):
    monkeypatch.setenv("PRESCISE_RATE_LIMIT", "123")
    c = _mock_client(monkeypatch)
    assert c.rate_limit == 123


def test_rate_limit_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("PRESCISE_RATE_LIMIT", "not-a-number")
    c = _mock_client(monkeypatch)
    assert c.rate_limit == 60
