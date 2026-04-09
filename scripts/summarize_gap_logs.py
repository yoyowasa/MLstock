from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize gap strategy logs for dry-run/live analysis")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/logs"),
        help="Directory that contains gap_trade_*.jsonl logs",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=5,
        help="Number of most recent gap_trade logs to summarize",
    )
    return parser.parse_args()


def _iter_gap_logs(log_dir: Path) -> List[Path]:
    return sorted(log_dir.glob("gap_trade_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)


def _session_label(path: Path, first_ts: Optional[str]) -> str:
    if first_ts:
        return str(first_ts)
    return path.stem.removeprefix("gap_trade_")


def _summarize_log(path: Path) -> Dict[str, Any]:
    scanner_count = 0
    scanner_symbols: List[str] = []
    options_selected: List[str] = []
    entry_symbols: List[str] = []
    exit_reasons: Counter[str] = Counter()
    first_ts: Optional[str] = None
    gap_trader_summary: Dict[str, Any] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if first_ts is None and isinstance(payload.get("ts_utc"), str):
                first_ts = payload["ts_utc"]
            message = payload.get("message")
            if message == "scanner_complete":
                scanner_count = int(payload.get("count", 0))
                scanner_symbols = [str(item).upper() for item in payload.get("symbols", []) if str(item).strip()]
            elif message == "options_skipped":
                options_selected = [str(item).upper() for item in payload.get("selected", []) if str(item).strip()]
            elif message == "options_filter_complete":
                options_selected = [str(item).upper() for item in payload.get("selected", []) if str(item).strip()]
            elif message == "entry_filled":
                symbol = str(payload.get("symbol", "")).upper()
                if symbol:
                    entry_symbols.append(symbol)
            elif message in {"exit_filled", "force_close_exit"}:
                reason = str(payload.get("reason") or message).strip().lower()
                if reason:
                    exit_reasons[reason] += 1
            elif message == "gap_trader_complete":
                gap_trader_summary = {
                    "entries": int(payload.get("entries", 0)),
                    "exits": int(payload.get("exits", 0)),
                    "closed_trades": int(payload.get("closed_trades", 0)),
                    "realized_pnl_usd": float(payload.get("realized_pnl_usd", 0.0)),
                    "realized_pnl_pct": float(payload.get("realized_pnl_pct", 0.0)),
                    "open_positions": payload.get("open_positions", []),
                }

    return {
        "log": str(path),
        "session": _session_label(path, first_ts),
        "scanner_count": scanner_count,
        "scanner_symbols": scanner_symbols,
        "options_selected": options_selected,
        "entry_symbols": entry_symbols,
        "exit_reasons": dict(exit_reasons),
        **gap_trader_summary,
    }


def main() -> None:
    args = _parse_args()
    logs = _iter_gap_logs(args.log_dir)[: max(args.latest, 1)]
    summaries = [_summarize_log(path) for path in logs]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
