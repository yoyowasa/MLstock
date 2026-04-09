from datetime import datetime
from zoneinfo import ZoneInfo

from mlstock.jobs.gap_trader import MinuteBar, _latest_close_price


def test_latest_close_price_uses_last_bar() -> None:
    tz = ZoneInfo("America/New_York")
    bars = [
        MinuteBar(datetime(2026, 4, 8, 15, 28, tzinfo=tz), 10.0, 10.2, 9.9, 10.1, 1000.0),
        MinuteBar(datetime(2026, 4, 8, 15, 29, tzinfo=tz), 10.1, 10.3, 10.0, 10.25, 900.0),
    ]
    assert _latest_close_price(bars) == 10.25


def test_latest_close_price_empty_returns_none() -> None:
    assert _latest_close_price([]) is None
