from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlstock.jobs.ingest_bars import _fetch_bars_batch


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def get_bars(self, **kwargs):
        return self._response


def _cfg():
    return SimpleNamespace(
        bars=SimpleNamespace(
            timeframe="1Day",
            feed="iex",
            adjustment="raw",
            asof="-",
        )
    )


def test_fetch_bars_batch_single_symbol_list_response() -> None:
    client = _FakeClient({"bars": [{"t": "2026-01-02T14:30:00Z", "o": 10.0}]})
    collected = _fetch_bars_batch(client, ["AAA"], "2026-01-01", "2026-01-10", _cfg())
    assert len(collected["AAA"]) == 1


def test_fetch_bars_batch_multi_symbol_list_response_uses_record_symbol() -> None:
    client = _FakeClient(
        {
            "bars": [
                {"S": "AAA", "t": "2026-01-02T14:30:00Z", "o": 10.0},
                {"symbol": "BBB", "t": "2026-01-02T14:30:00Z", "o": 20.0},
            ]
        }
    )
    collected = _fetch_bars_batch(client, ["AAA", "BBB"], "2026-01-01", "2026-01-10", _cfg())
    assert len(collected["AAA"]) == 1
    assert len(collected["BBB"]) == 1


def test_fetch_bars_batch_multi_symbol_list_response_without_symbol_raises() -> None:
    client = _FakeClient({"bars": [{"t": "2026-01-02T14:30:00Z", "o": 10.0}]})
    with pytest.raises(RuntimeError):
        _fetch_bars_batch(client, ["AAA", "BBB"], "2026-01-01", "2026-01-10", _cfg())
