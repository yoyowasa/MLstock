from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

from mlstock.brokers import AlpacaOrderBroker, WebullOrderBroker
from mlstock.config.loader import load_config
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.twelvedata.client import TwelveDataClient
from mlstock.jobs.gap_scanner import scan_gap_candidates
from mlstock.jobs.gap_trader import run_gap_trader, simulate_gap_candidates_session
from mlstock.jobs.options_filter import filter_unusual_options_activity
from mlstock.logging.logger import build_log_path, log_event, setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gap + Options intraday strategy runner")
    parser.add_argument(
        "--gap-config",
        type=Path,
        default=Path("config/gap_config.yaml"),
        help="Path to gap strategy config YAML",
    )
    parser.add_argument("--scan-only", action="store_true", help="Run scanner only and exit")
    parser.add_argument("--skip-options", action="store_true", help="Skip yfinance options filter")
    parser.add_argument("--skip-wait", action="store_true", help="Run stages immediately without clock wait")
    parser.add_argument("--max-loops", type=int, default=None, help="Limit trader loop count for debug")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (override seed universe)")
    parser.add_argument("--live", action="store_true", help="Enable real orders (default is dry-run)")
    parser.add_argument(
        "--replay-log", type=Path, default=None, help="Replay prior scanner log and calculate hypothetical PNL"
    )
    parser.add_argument(
        "--compare-data-sources",
        action="store_true",
        help="Compare today's Alpaca scanner result with Twelve Data on the same symbol set",
    )
    return parser.parse_args()


def _load_gap_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Gap config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("gap_config.yaml must be a mapping")
    return payload


def _wait_until(tz: ZoneInfo, target: datetime) -> None:
    while True:
        now = datetime.now(tz)
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        sleep_for = min(delta, 1.0)
        import time

        time.sleep(sleep_for)


def _stage_datetime(now_local: datetime, hh: int, mm: int, ss: int) -> datetime:
    return now_local.replace(hour=hh, minute=mm, second=ss, microsecond=0)


def _clock_guard(trading_client: AlpacaClient, logger) -> bool:
    try:
        clock = trading_client.get_clock()
    except Exception as exc:
        log_event(logger, "clock_check_failed", error=str(exc))
        return False

    if not isinstance(clock, dict):
        log_event(logger, "clock_check_invalid_payload", payload=str(clock))
        return False

    is_open = bool(clock.get("is_open"))
    log_event(
        logger,
        "clock_status",
        is_open=is_open,
        next_open=clock.get("next_open"),
        next_close=clock.get("next_close"),
        timestamp=clock.get("timestamp"),
    )
    return is_open


def _preflight_iex_bars(cfg, data_client: AlpacaClient, logger) -> None:
    tz = ZoneInfo(cfg.project.timezone)
    now_local = datetime.now(tz)
    start = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_local < start:
        start = now_local.replace(hour=4, minute=0, second=0, microsecond=0)
    try:
        response = data_client.get_bars(
            symbols=["SPY"],
            start=start.astimezone(ZoneInfo("UTC")).isoformat(),
            end=now_local.astimezone(ZoneInfo("UTC")).isoformat(),
            timeframe="1Min",
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
            limit=5,
        )
        bars = response.get("bars", {}) if isinstance(response, dict) else {}
        if isinstance(bars, dict):
            count = len(bars.get("SPY", []))
        elif isinstance(bars, list):
            count = len(bars)
        else:
            count = 0
        log_event(logger, "preflight_iex_bars", feed=cfg.bars.feed, symbol="SPY", bars=count)
    except Exception as exc:
        log_event(logger, "preflight_iex_bars_failed", error=str(exc), feed=cfg.bars.feed)


def _parse_symbols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return sorted(set(symbols)) if symbols else None


def _entry_limit(gap_cfg: Dict[str, Any]) -> int:
    entry = gap_cfg.get("entry", {}) if isinstance(gap_cfg.get("entry"), dict) else {}
    try:
        value = int(entry.get("max_candidates", 3))
    except (TypeError, ValueError):
        value = 3
    return max(value, 1)


def _build_order_broker(trading_client: AlpacaClient):
    broker_name = os.getenv("GAP_ORDER_BROKER", "alpaca").strip().lower()
    if broker_name == "webull":
        base_url = os.getenv("WEBULL_BASE_URL", "https://api.webull.co.jp")
        return broker_name, WebullOrderBroker.from_env(base_url=base_url)
    if broker_name == "alpaca":
        return broker_name, AlpacaOrderBroker(trading_client)
    raise ValueError(f"Unsupported GAP_ORDER_BROKER: {broker_name}")


def _load_candidates_from_log(path: Path) -> tuple[List[Dict[str, Any]], Optional[datetime]]:
    if not path.exists():
        raise FileNotFoundError(f"Replay log not found: {path}")
    scanner_candidates: List[Dict[str, Any]] = []
    scanner_ts: Optional[datetime] = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if payload.get("message") != "scanner_complete":
                continue
            raw_candidates = payload.get("candidates")
            if isinstance(raw_candidates, list):
                scanner_candidates = [item for item in raw_candidates if isinstance(item, dict)]
            ts_utc = payload.get("ts_utc")
            if isinstance(ts_utc, str):
                try:
                    scanner_ts = datetime.fromisoformat(ts_utc)
                except ValueError:
                    scanner_ts = None
    return scanner_candidates, scanner_ts


def _gap_candidate_from_dict(payload: Dict[str, Any]):
    from mlstock.jobs.gap_scanner import GapCandidate

    return GapCandidate(
        symbol=str(payload.get("symbol", "")).strip().upper(),
        gap_pct=float(payload.get("gap_pct", 0.0)),
        prev_close=float(payload.get("prev_close", 0.0)),
        open_price=float(payload.get("open_price", 0.0)),
        avg_volume_30d=float(payload.get("avg_volume_30d", 0.0)),
        first_window_volume=float(payload.get("first_window_volume", 0.0)),
        daily_volume_pace=float(payload.get("daily_volume_pace", 0.0)),
        volume_pace_ratio=float(payload.get("volume_pace_ratio", 0.0)),
        market_cap_m=float(payload.get("market_cap_m", 0.0)),
    )


def _candidate_lookup(candidates: List[Any]) -> Dict[str, Dict[str, Any]]:
    return {item.symbol: item.to_dict() for item in candidates}


def main() -> None:
    load_dotenv(override=False)
    args = _parse_args()
    cfg = load_config()
    gap_cfg = _load_gap_config(args.gap_config)

    log_path = build_log_path(cfg, "gap_trade")
    logger = setup_logger("gap_trade", log_path, cfg.logging.level)
    log_event(
        logger,
        "start",
        gap_config=str(args.gap_config),
        scan_only=args.scan_only,
        skip_options=args.skip_options,
        skip_wait=args.skip_wait,
        live=args.live,
        compare_data_sources=args.compare_data_sources,
    )

    data_client = AlpacaClient.from_env(cfg.alpaca.data_base_url)
    trading_client = AlpacaClient.from_env(cfg.alpaca.trading_base_url)

    tz = ZoneInfo(cfg.project.timezone)
    if args.replay_log is not None:
        raw_candidates, scanner_ts = _load_candidates_from_log(args.replay_log)
        candidates = [_gap_candidate_from_dict(item) for item in raw_candidates if item.get("symbol")]
        session_date = scanner_ts.astimezone(tz).date() if scanner_ts is not None else None
        summary = simulate_gap_candidates_session(
            cfg=cfg,
            gap_cfg=gap_cfg,
            candidates=candidates,
            data_client=data_client,
            logger=logger,
            session_date=session_date,
        )
        log_event(
            logger,
            "scan_replay_requested",
            replay_log=str(args.replay_log),
            session_date=None if session_date is None else session_date.isoformat(),
            candidate_count=len(candidates),
            summary=summary,
        )
        print(
            "scan-replay:"
            f" trades={summary.get('closed_trades')}"
            f" pnl_usd={summary.get('realized_pnl_usd')}"
            f" pnl_pct={summary.get('realized_pnl_pct')}"
        )
        print(f"log: {log_path}")
        return

    broker_name, order_broker = _build_order_broker(trading_client)
    log_event(logger, "order_broker_selected", broker=broker_name, live=args.live)

    now_local = datetime.now(tz)
    scan_at = _stage_datetime(now_local, 9, 30, 5)
    options_at = _stage_datetime(now_local, 9, 30, 30)
    trader_at = _stage_datetime(now_local, 9, 35, 0)

    if not args.skip_wait:
        log_event(logger, "wait_for_scan", target=scan_at.isoformat())
        _wait_until(tz, scan_at)

    if not _clock_guard(trading_client, logger):
        log_event(logger, "stop_market_closed")
        print("market closed: skipped")
        return

    _preflight_iex_bars(cfg, data_client, logger)

    symbols = _parse_symbols(args.symbols)
    candidates = scan_gap_candidates(
        cfg=cfg,
        gap_cfg=gap_cfg,
        data_client=data_client,
        logger=logger,
        as_of=datetime.now(tz),
        symbols=symbols,
        data_source="alpaca",
    )
    log_event(
        logger,
        "scanner_complete",
        count=len(candidates),
        symbols=[item.symbol for item in candidates],
        candidates=[item.to_dict() for item in candidates],
    )
    print(f"scanner: {len(candidates)} candidates")

    if args.compare_data_sources:
        compare_symbols = symbols or [item.symbol for item in candidates]
        if not compare_symbols:
            log_event(logger, "scanner_compare_skipped", reason="no_compare_symbols")
            print("compare: skipped (no symbols)")
        else:
            try:
                td_client = TwelveDataClient.from_env()
            except Exception as exc:
                log_event(logger, "scanner_compare_skipped", reason="twelvedata_key_missing", error=str(exc))
                print("compare: skipped (TWELVEDATA_API_KEY missing)")
            else:
                td_candidates = scan_gap_candidates(
                    cfg=cfg,
                    gap_cfg=gap_cfg,
                    data_client=td_client,
                    logger=logger,
                    as_of=datetime.now(tz),
                    symbols=compare_symbols,
                    data_source="twelvedata",
                )
                alpaca_lookup = _candidate_lookup(candidates)
                td_lookup = _candidate_lookup(td_candidates)
                overlap = sorted(set(alpaca_lookup.keys()) & set(td_lookup.keys()))
                alpaca_only = sorted(set(alpaca_lookup.keys()) - set(td_lookup.keys()))
                td_only = sorted(set(td_lookup.keys()) - set(alpaca_lookup.keys()))
                per_symbol = []
                for symbol in sorted(set(compare_symbols)):
                    per_symbol.append(
                        {
                            "symbol": symbol,
                            "alpaca": alpaca_lookup.get(symbol),
                            "twelvedata": td_lookup.get(symbol),
                        }
                    )
                log_event(
                    logger,
                    "scanner_compare_complete",
                    compare_symbols=sorted(set(compare_symbols)),
                    alpaca_count=len(candidates),
                    twelvedata_count=len(td_candidates),
                    overlap=overlap,
                    alpaca_only=alpaca_only,
                    twelvedata_only=td_only,
                    per_symbol=per_symbol,
                )
                print(
                    "compare:"
                    f" alpaca={len(candidates)}"
                    f" twelvedata={len(td_candidates)}"
                    f" overlap={len(overlap)}"
                )

    if not candidates:
        log_event(logger, "stop_no_candidates")
        return

    if args.scan_only:
        log_event(logger, "stop_scan_only")
        return

    if not args.skip_wait:
        _wait_until(tz, options_at)

    limit = _entry_limit(gap_cfg)
    if args.skip_options:
        selected: List[Any] = list(candidates[:limit])
        log_event(logger, "options_skipped", selected=[item.symbol for item in selected], limit=limit)
    else:
        selected = filter_unusual_options_activity(candidates, gap_cfg=gap_cfg, max_candidates=limit)
        log_event(
            logger,
            "options_filter_complete",
            count=len(selected),
            selected=[item.candidate.symbol for item in selected],
            details=[item.to_dict() for item in selected],
        )
    print(f"options: {len(selected)} selected")

    if not selected:
        log_event(logger, "stop_no_options_selected")
        return

    if not args.skip_wait:
        _wait_until(tz, trader_at)

    summary = run_gap_trader(
        cfg=cfg,
        gap_cfg=gap_cfg,
        candidates=selected,
        data_client=data_client,
        order_broker=order_broker,
        logger=logger,
        dry_run=not args.live,
        max_loops=args.max_loops,
    )
    log_event(logger, "complete", summary=summary, log_path=str(log_path))
    print(f"trader: entries={summary.get('entries')} exits={summary.get('exits')} dry_run={not args.live}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
