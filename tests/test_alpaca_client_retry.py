from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, Dict, Optional

from mlstock.data.alpaca.client import AlpacaClient


class _FakeResponse:
    def __init__(self, status_code: int, *, payload: Optional[Dict[str, Any]] = None, headers=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text
        self.ok = 200 <= status_code < 300

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_request_uses_retry_after_header_for_429(monkeypatch) -> None:
    client = AlpacaClient(
        base_url="https://example.test",
        api_key="k",
        api_secret="s",
        max_retries=1,
        backoff_seconds=1.0,
    )
    responses = [
        _FakeResponse(429, headers={"Retry-After": "3"}),
        _FakeResponse(200, payload={"ok": True}),
    ]

    def _request(*args, **kwargs):
        return responses.pop(0)

    sleeps = []
    monkeypatch.setattr(client.session, "request", _request)
    monkeypatch.setattr("mlstock.data.alpaca.client.time.sleep", lambda s: sleeps.append(s))

    result = client.get_calendar("2026-01-01", "2026-01-02")
    assert result["ok"] is True
    assert sleeps and sleeps[0] >= 3.0


def test_retry_after_http_date_is_parsed() -> None:
    client = AlpacaClient(base_url="https://example.test", api_key="k", api_secret="s")
    header_value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=2))
    seconds = client._retry_after_seconds(header_value)
    assert seconds is not None
    assert seconds >= 0.0
