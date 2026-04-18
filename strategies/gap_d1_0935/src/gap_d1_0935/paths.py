from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


def strategy_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    path = strategy_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def watchlist_dir() -> Path:
    path = artifacts_dir() / "watchlist"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scans_dir() -> Path:
    path = artifacts_dir() / "scans"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = artifacts_dir() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = artifacts_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return strategy_root() / "config" / "strategy_gap_d1_0935.yaml"


def watchlist_path(trade_date: date) -> Path:
    return watchlist_dir() / f"watchlist_gap_d1_{trade_date:%Y%m%d}.csv"


def scan_path(trade_date: date) -> Path:
    return scans_dir() / f"gap_0935_candidates_{trade_date:%Y%m%d}.csv"


def compare_report_path(trade_date: date) -> Path:
    return reports_dir() / f"gap_0935_compare_{trade_date:%Y%m%d}.csv"


def compare_summary_path() -> Path:
    return reports_dir() / "gap_0935_compare_summary.csv"


def log_path(prefix: str, ts: datetime | None = None) -> Path:
    now = ts or datetime.utcnow()
    return logs_dir() / f"{prefix}_{now:%Y%m%d_%H%M%S_%f}.jsonl"
