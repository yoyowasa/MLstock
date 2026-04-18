from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_SRC = ROOT / "strategies" / "gap_d1_0935" / "src"
for path in [str(ROOT / "src"), str(STRATEGY_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from gap_d1_0935.gap_0935_watchlist_scanner import _aggregate_first5  # noqa: E402
from gap_d1_0935.metadata import classify_suffix_pattern, determine_security_type  # noqa: E402


def test_classify_suffix_pattern_examples() -> None:
    assert classify_suffix_pattern("AACBU") == "unit"
    assert classify_suffix_pattern("AACBR") == "rights"
    assert classify_suffix_pattern("AAM.U") == "unit"
    assert classify_suffix_pattern("AAPL") == "common_like"


def test_determine_security_type_prefers_quote_type() -> None:
    assert determine_security_type("SPY", "ETF", "ARCA") == "etf"
    assert determine_security_type("AAPL", "EQUITY", "NASDAQ") == "common_stock"


def test_aggregate_first5_builds_first_bar_stats() -> None:
    items = [
        {"t": "2026-04-17T13:30:00Z", "o": 10.0, "h": 10.5, "l": 9.9, "c": 10.4, "v": 1000},
        {"t": "2026-04-17T13:31:00Z", "o": 10.4, "h": 10.7, "l": 10.3, "c": 10.6, "v": 800},
        {"t": "2026-04-17T13:34:00Z", "o": 10.6, "h": 10.8, "l": 10.5, "c": 10.7, "v": 700},
    ]
    agg = _aggregate_first5("TEST", items)
    assert agg is not None
    assert agg["open_D"] == 10.0
    assert agg["first5_high"] == 10.8
    assert agg["first5_low"] == 9.9
    assert agg["first5_close"] == 10.7
    assert agg["first5_volume"] == 2500
