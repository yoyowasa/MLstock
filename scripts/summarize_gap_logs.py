from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        default=None,
        help="Number of most recent gap_trade logs to summarize. Omit to summarize all logs.",
    )
    return parser.parse_args()


def _iter_gap_logs(log_dir: Path) -> List[Path]:
    return sorted(log_dir.glob("gap_trade_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)


def _session_label(path: Path, first_ts: Optional[str]) -> str:
    if first_ts:
        return str(first_ts)
    return path.stem.removeprefix("gap_trade_")


def _mode_tuple(payload: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    return (
        bool(payload.get("scan_only")),
        bool(payload.get("skip_options")),
        bool(payload.get("live")),
    )


def _summarize_log(path: Path) -> Dict[str, Any]:
    scanner_count = 0
    scanner_symbols: List[str] = []
    scanner_diagnostics: Dict[str, Any] = {}
    options_selected: List[str] = []
    entry_symbols: List[str] = []
    exit_reasons: Counter[str] = Counter()
    first_ts: Optional[str] = None
    session_utc: Optional[str] = None
    trade_date: Optional[str] = None
    scan_only: Optional[bool] = None
    skip_options: Optional[bool] = None
    live: Optional[bool] = None
    start_modes: List[Tuple[bool, bool, bool]] = []
    stop_messages: List[str] = []
    completed = False
    gap_trader_summary: Dict[str, Any] = {}
    replay_mode = False

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if first_ts is None and isinstance(payload.get("ts_utc"), str):
                first_ts = payload["ts_utc"]
            if isinstance(payload.get("ts_utc"), str):
                session_utc = payload["ts_utc"]
            message = payload.get("message")
            if message == "start":
                scan_only, skip_options, live = _mode_tuple(payload)
                start_modes.append((scan_only, skip_options, live))
            elif message == "scanner_complete":
                scanner_count = int(payload.get("count", 0))
                scanner_symbols = [str(item).upper() for item in payload.get("symbols", []) if str(item).strip()]
            elif message == "scanner_diagnostics":
                scanner_diagnostics = {
                    "data_source": payload.get("data_source"),
                    "trade_date": payload.get("trade_date"),
                    "universe_count": int(payload.get("universe_count", 0)),
                    "daily_count": int(payload.get("daily_count", 0)),
                    "open_count": int(payload.get("open_count", 0)),
                    "missing_open_count": int(payload.get("missing_open_count", 0)),
                    "liquid_price_count": int(payload.get("liquid_price_count", 0)),
                    "gap_ge_2_count": int(payload.get("gap_ge_2_count", 0)),
                    "raw_candidate_count": int(payload.get("raw_candidate_count", 0)),
                    "candidate_count": int(payload.get("candidate_count", 0)),
                }
                trade_date = payload.get("trade_date")
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
                completed = True
            elif message in {"scan_replay_requested", "scan_replay_complete"}:
                replay_mode = True
            elif message == "complete":
                completed = True
            elif isinstance(message, str) and message.startswith("stop_"):
                stop_messages.append(message)

    mode_collision = len(set(start_modes)) > 1
    if start_modes:
        scan_only, skip_options, live = start_modes[-1]
    final_status = "complete" if completed else (stop_messages[-1] if stop_messages else "")

    return {
        "log": str(path),
        "session": _session_label(path, first_ts),
        "session_utc": session_utc,
        "trade_date": trade_date,
        "scan_only": scan_only,
        "skip_options": skip_options,
        "live": live,
        "start_count": len(start_modes),
        "mode_collision": mode_collision,
        "replay_mode": replay_mode,
        "status": final_status,
        "scanner_count": scanner_count,
        "scanner_symbols": scanner_symbols,
        "scanner_diagnostics": scanner_diagnostics,
        "options_selected": options_selected,
        "entry_symbols": entry_symbols,
        "exit_reasons": dict(exit_reasons),
        **gap_trader_summary,
    }


def main() -> None:
    args = _parse_args()
    logs = _iter_gap_logs(args.log_dir)
    if args.latest is not None:
        logs = logs[: max(args.latest, 1)]
    summaries = [_summarize_log(path) for path in logs]
    summaries.sort(key=lambda item: str(item.get("trade_date") or item.get("session_utc") or item.get("session")))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
