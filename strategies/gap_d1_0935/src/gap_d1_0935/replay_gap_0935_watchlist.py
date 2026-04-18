from __future__ import annotations

from datetime import date

from .build_gap_d1_watchlist import build_gap_d1_watchlist
from .gap_0935_watchlist_scanner import Gap0935ScanResult, scan_gap_0935_watchlist


def replay_gap_0935_watchlist(trade_date: date, rebuild_watchlist: bool = False) -> Gap0935ScanResult:
    if rebuild_watchlist:
        build_gap_d1_watchlist(trade_date=trade_date)
    return scan_gap_0935_watchlist(trade_date=trade_date)
