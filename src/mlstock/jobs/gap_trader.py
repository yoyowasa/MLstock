from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.brokers.base import BrokerOrderResult, OrderBroker
from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.jobs.gap_scanner import GapCandidate
from mlstock.logging.logger import log_event


@dataclass(frozen=True)
class EntrySettings:
    max_candidates: int
    pullback_from_open_pct: float
    vwap_tolerance_pct: float
    low_volume_ratio: float
    risk_per_trade_usd: float
    dry_run_cash_usd: float
    max_notional_per_trade_usd: float
    min_order_qty: int


@dataclass(frozen=True)
class ExitSettings:
    target_pct: float
    stop_pct: float
    time_cut_hour: int
    force_close_hour: int
    force_close_minute: int


@dataclass(frozen=True)
class TraderSettings:
    entry: EntrySettings
    exit: ExitSettings


@dataclass
class TrackedPosition:
    symbol: str
    qty: int
    entry_price: float
    opened_at: str


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    qty: int
    entry_price: float
    exit_price: float
    exit_reason: str
    entry_notional_usd: float
    realized_pnl_usd: float
    realized_pnl_pct: float
    opened_at: str
    closed_at: str


@dataclass(frozen=True)
class MinuteBar:
    ts_local: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_settings(raw_cfg: Mapping[str, Any]) -> TraderSettings:
    entry_raw = raw_cfg.get("entry", {}) if isinstance(raw_cfg.get("entry"), dict) else {}
    exit_raw = raw_cfg.get("exit", {}) if isinstance(raw_cfg.get("exit"), dict) else {}
    return TraderSettings(
        entry=EntrySettings(
            max_candidates=max(_to_int(entry_raw.get("max_candidates"), 3), 1),
            pullback_from_open_pct=max(_to_float(entry_raw.get("pullback_from_open_pct"), 2.0), 0.0),
            vwap_tolerance_pct=max(_to_float(entry_raw.get("vwap_tolerance_pct"), 1.0), 0.0),
            low_volume_ratio=max(_to_float(entry_raw.get("low_volume_ratio"), 0.5), 0.0),
            risk_per_trade_usd=max(_to_float(entry_raw.get("risk_per_trade_usd"), 50.0), 0.0),
            dry_run_cash_usd=max(_to_float(entry_raw.get("dry_run_cash_usd"), 10000.0), 100.0),
            max_notional_per_trade_usd=max(_to_float(entry_raw.get("max_notional_per_trade_usd"), 0.0), 0.0),
            min_order_qty=max(_to_int(entry_raw.get("min_order_qty"), 1), 1),
        ),
        exit=ExitSettings(
            target_pct=max(_to_float(exit_raw.get("target_pct"), 4.0), 0.0),
            stop_pct=max(_to_float(exit_raw.get("stop_pct"), 2.0), 0.0),
            time_cut_hour=_to_int(exit_raw.get("time_cut_hour"), 14),
            force_close_hour=_to_int(exit_raw.get("force_close_hour"), 15),
            force_close_minute=_to_int(exit_raw.get("force_close_minute"), 30),
        ),
    )


def _emit(logger: Optional[logging.Logger], message: str, **fields: Any) -> None:
    if logger is None:
        return
    log_event(logger, message, **fields)


def _extract_candidate(item: Any) -> GapCandidate:
    if isinstance(item, GapCandidate):
        return item
    candidate = getattr(item, "candidate", None)
    if isinstance(candidate, GapCandidate):
        return candidate
    raise TypeError("candidate の形式が不正です。")


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()


def _extract_bar_symbol(item: Dict[str, Any]) -> Optional[str]:
    symbol = item.get("S") or item.get("symbol") or item.get("s")
    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    return text or None


def _fetch_bars_batch(
    client: AlpacaClient,
    cfg: AppConfig,
    symbols: List[str],
    start_local: datetime,
    end_local: datetime,
) -> Dict[str, List[Dict[str, Any]]]:
    collected: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in symbols}
    page_token: Optional[str] = None
    while True:
        response = client.get_bars(
            symbols=symbols,
            start=_iso_utc(start_local),
            end=_iso_utc(end_local),
            timeframe="1Min",
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
            page_token=page_token,
        )
        bars = response.get("bars") if isinstance(response, dict) else None
        if isinstance(bars, dict):
            for symbol, items in bars.items():
                key = str(symbol).upper()
                if key in collected:
                    collected[key].extend([item for item in items if isinstance(item, dict)])
        elif isinstance(bars, list):
            if len(symbols) == 1:
                symbol = symbols[0]
                collected[symbol].extend([item for item in bars if isinstance(item, dict)])
            else:
                for item in bars:
                    if not isinstance(item, dict):
                        continue
                    symbol = _extract_bar_symbol(item)
                    if symbol and symbol in collected:
                        collected[symbol].append(item)
        page_token = response.get("next_page_token") if isinstance(response, dict) else None
        if not page_token:
            break
    return collected


def _normalize_minute_bars(
    items: List[Dict[str, Any]],
    tz: ZoneInfo,
    session_date: date,
) -> List[MinuteBar]:
    bars: List[MinuteBar] = []
    open_dt = datetime.combine(session_date, dtime(9, 30), tzinfo=tz)
    close_dt = datetime.combine(session_date, dtime(16, 0), tzinfo=tz)
    for item in items:
        ts_raw = item.get("t")
        if ts_raw is None:
            continue
        ts_local = pd.to_datetime(ts_raw, utc=True).tz_convert(tz).to_pydatetime()
        if ts_local < open_dt or ts_local >= close_dt:
            continue
        try:
            bars.append(
                MinuteBar(
                    ts_local=ts_local,
                    open_price=float(item.get("o")),
                    high_price=float(item.get("h")),
                    low_price=float(item.get("l")),
                    close_price=float(item.get("c")),
                    volume=float(item.get("v")),
                )
            )
        except (TypeError, ValueError):
            continue
    bars.sort(key=lambda row: row.ts_local)
    deduped: Dict[datetime, MinuteBar] = {}
    for bar in bars:
        deduped[bar.ts_local] = bar
    return [deduped[key] for key in sorted(deduped.keys())]


def _compute_vwap(bars: Sequence[MinuteBar]) -> Optional[float]:
    pv = 0.0
    volume = 0.0
    for bar in bars:
        pv += bar.close_price * max(bar.volume, 0.0)
        volume += max(bar.volume, 0.0)
    if volume <= 0:
        return None
    return pv / volume


def _latest_close_price(bars: Sequence[MinuteBar]) -> Optional[float]:
    if not bars:
        return None
    return bars[-1].close_price


def _entry_conditions_met(
    bars: Sequence[MinuteBar],
    open_price: float,
    settings: EntrySettings,
) -> tuple[bool, Dict[str, Any]]:
    if not bars:
        return False, {"reason": "bars_missing"}
    current_price = bars[-1].close_price
    pullback_price = open_price * (1.0 - settings.pullback_from_open_pct / 100.0)
    pullback_ok = current_price <= pullback_price

    if len(bars) >= 6:
        last_volume = bars[-1].volume
        prev_five = [bar.volume for bar in bars[-6:-1]]
        avg_prev_five = sum(prev_five) / 5.0
        volume_ok = last_volume <= (avg_prev_five * settings.low_volume_ratio)
    else:
        avg_prev_five = None
        last_volume = bars[-1].volume
        volume_ok = False

    vwap = _compute_vwap(bars)
    if vwap is None:
        vwap_ok = False
    else:
        vwap_floor = vwap * (1.0 - settings.vwap_tolerance_pct / 100.0)
        vwap_ok = current_price >= vwap_floor

    matched = pullback_ok and volume_ok and vwap_ok
    return matched, {
        "current_price": current_price,
        "pullback_price": pullback_price,
        "pullback_ok": pullback_ok,
        "last_volume": last_volume,
        "avg_prev_five_volume": avg_prev_five,
        "volume_ok": volume_ok,
        "vwap": vwap,
        "vwap_ok": vwap_ok,
    }


def _should_take_profit(current_price: float, entry_price: float, target_pct: float) -> bool:
    return current_price >= entry_price * (1.0 + target_pct / 100.0)


def _should_stop_loss(current_price: float, entry_price: float, stop_pct: float) -> bool:
    return current_price <= entry_price * (1.0 - stop_pct / 100.0)


def _calculate_realized_pnl(entry_price: float, exit_price: float, qty: int) -> tuple[float, float, float]:
    entry_notional = max(entry_price, 0.0) * max(qty, 0)
    if entry_notional <= 0:
        return 0.0, 0.0, 0.0
    realized_pnl_usd = (exit_price - entry_price) * qty
    realized_pnl_pct = (realized_pnl_usd / entry_notional) * 100.0
    return realized_pnl_usd, realized_pnl_pct, entry_notional


def _summarize_closed_trades(closed_trades: Sequence[ClosedTrade]) -> Dict[str, Any]:
    realized_pnl_usd = round(sum(item.realized_pnl_usd for item in closed_trades), 6)
    total_entry_notional = sum(item.entry_notional_usd for item in closed_trades)
    realized_pnl_pct = round((realized_pnl_usd / total_entry_notional) * 100.0, 6) if total_entry_notional > 0 else 0.0
    return {
        "closed_trades": len(closed_trades),
        "realized_pnl_usd": realized_pnl_usd,
        "realized_pnl_pct": realized_pnl_pct,
        "closed_entry_notional_usd": round(total_entry_notional, 6),
    }


def _bootstrap_positions(
    order_broker: OrderBroker,
    target_symbols: set[str],
    logger: Optional[logging.Logger],
) -> Dict[str, TrackedPosition]:
    tracked: Dict[str, TrackedPosition] = {}
    try:
        positions = order_broker.list_positions()
    except Exception as exc:
        _emit(logger, "positions_bootstrap_failed", error=str(exc))
        return tracked
    for item in positions:
        symbol = str(item.symbol).strip().upper()
        if symbol not in target_symbols:
            continue
        qty = int(item.qty)
        if qty <= 0:
            continue
        entry_price = float(item.avg_entry_price)
        if entry_price <= 0:
            continue
        tracked[symbol] = TrackedPosition(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            opened_at=datetime.now(ZoneInfo("UTC")).isoformat(),
        )
    if tracked:
        _emit(logger, "positions_bootstrapped", symbols=sorted(tracked.keys()))
    return tracked


def _submit_order(
    order_broker: OrderBroker,
    symbol: str,
    qty: int,
    side: str,
    dry_run: bool,
) -> BrokerOrderResult:
    if dry_run:
        return BrokerOrderResult(
            order_id=f"dryrun-{side}-{symbol}",
            symbol=symbol,
            qty=qty,
            side=side,
            filled_avg_price=0.0,
            raw={"symbol": symbol, "qty": qty, "side": side},
        )
    if side == "buy":
        return order_broker.submit_market_buy(symbol=symbol, qty=qty)
    if side == "sell":
        return order_broker.submit_market_sell(symbol=symbol, qty=qty)
    raise ValueError(f"unsupported side: {side}")


def _sleep_to_next_minute(now_local: datetime) -> None:
    next_tick = (now_local + timedelta(minutes=1)).replace(second=2, microsecond=0)
    sleep_seconds = (next_tick - now_local).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def _available_buying_power(
    order_broker: OrderBroker,
    dry_run: bool,
    dry_run_cash_usd: float,
    logger: Optional[logging.Logger],
) -> float:
    if dry_run:
        return dry_run_cash_usd
    try:
        buying_power = order_broker.get_buying_power()
    except Exception as exc:
        _emit(logger, "account_fetch_failed", error=str(exc), fallback_cash=dry_run_cash_usd)
        return dry_run_cash_usd
    if buying_power > 0:
        return buying_power
    _emit(logger, "account_buying_power_missing", fallback_cash=dry_run_cash_usd)
    return dry_run_cash_usd


def _calculate_entry_qty(
    current_price: float,
    stop_pct: float,
    remaining_budget_usd: float,
    remaining_slots: int,
    entry_settings: EntrySettings,
) -> tuple[int, Dict[str, Any]]:
    if current_price <= 0:
        return 0, {"reason": "invalid_price"}
    if remaining_budget_usd <= 0:
        return 0, {"reason": "budget_empty"}

    per_trade_budget = remaining_budget_usd / float(max(remaining_slots, 1))
    if entry_settings.max_notional_per_trade_usd > 0:
        per_trade_budget = min(per_trade_budget, entry_settings.max_notional_per_trade_usd)

    qty_by_notional = int(per_trade_budget // current_price)
    stop_distance = current_price * (stop_pct / 100.0)
    if entry_settings.risk_per_trade_usd > 0 and stop_distance > 0:
        qty_by_risk = int(entry_settings.risk_per_trade_usd // stop_distance)
    else:
        qty_by_risk = qty_by_notional

    qty = min(qty_by_notional, qty_by_risk)
    if qty < entry_settings.min_order_qty:
        return 0, {
            "reason": "qty_below_min",
            "qty_by_notional": qty_by_notional,
            "qty_by_risk": qty_by_risk,
            "per_trade_budget": per_trade_budget,
            "stop_distance": stop_distance,
        }

    return qty, {
        "qty_by_notional": qty_by_notional,
        "qty_by_risk": qty_by_risk,
        "per_trade_budget": per_trade_budget,
        "stop_distance": stop_distance,
    }


def _close_position(
    tracked_positions: Dict[str, TrackedPosition],
    symbol: str,
    exit_price: float,
    reason: str,
    closed_at: str,
) -> Optional[ClosedTrade]:
    position = tracked_positions.pop(symbol, None)
    if position is None:
        return None
    realized_pnl_usd, realized_pnl_pct, entry_notional = _calculate_realized_pnl(
        entry_price=position.entry_price,
        exit_price=exit_price,
        qty=position.qty,
    )
    return ClosedTrade(
        symbol=symbol,
        qty=position.qty,
        entry_price=position.entry_price,
        exit_price=exit_price,
        exit_reason=reason,
        entry_notional_usd=entry_notional,
        realized_pnl_usd=realized_pnl_usd,
        realized_pnl_pct=realized_pnl_pct,
        opened_at=position.opened_at,
        closed_at=closed_at,
    )


def simulate_gap_candidates_session(
    cfg: AppConfig,
    gap_cfg: Mapping[str, Any],
    candidates: Sequence[Any],
    data_client: AlpacaClient,
    logger: Optional[logging.Logger] = None,
    session_date: Optional[date] = None,
) -> Dict[str, Any]:
    settings = _load_settings(gap_cfg)
    if not candidates:
        summary = {
            "status": "no_candidates",
            "session_date": session_date.isoformat() if session_date else None,
            "closed_trades": 0,
            "realized_pnl_usd": 0.0,
            "realized_pnl_pct": 0.0,
            "trades": [],
        }
        _emit(logger, "scan_replay_complete", **summary)
        return summary

    parsed = [_extract_candidate(item) for item in candidates][: settings.entry.max_candidates]
    if not parsed:
        summary = {
            "status": "no_candidates",
            "session_date": session_date.isoformat() if session_date else None,
            "closed_trades": 0,
            "realized_pnl_usd": 0.0,
            "realized_pnl_pct": 0.0,
            "trades": [],
        }
        _emit(logger, "scan_replay_complete", **summary)
        return summary

    tz = ZoneInfo(cfg.project.timezone)
    if session_date is None:
        session_date = datetime.now(tz).date()
    session_open = datetime.combine(session_date, dtime(9, 30), tzinfo=tz)
    session_close = datetime.combine(session_date, dtime(16, 0), tzinfo=tz)
    time_cut = datetime.combine(session_date, dtime(settings.exit.time_cut_hour, 0), tzinfo=tz)
    force_close = datetime.combine(
        session_date,
        dtime(settings.exit.force_close_hour, settings.exit.force_close_minute),
        tzinfo=tz,
    )
    batch_size = max(1, min(200, int(cfg.bars.batch_size)))
    candidate_by_symbol = {item.symbol: item for item in parsed}
    symbols = sorted(candidate_by_symbol.keys())
    bars_by_symbol: Dict[str, List[MinuteBar]] = {symbol: [] for symbol in symbols}

    for batch in _chunk(symbols, batch_size):
        try:
            batch_raw = _fetch_bars_batch(
                client=data_client,
                cfg=cfg,
                symbols=batch,
                start_local=session_open,
                end_local=session_close,
            )
        except Exception as exc:
            _emit(logger, "scan_replay_bar_fetch_failed", error=str(exc), batch=batch)
            continue
        for symbol in batch:
            bars_by_symbol[symbol] = _normalize_minute_bars(batch_raw.get(symbol, []), tz=tz, session_date=session_date)

    buying_power = settings.entry.dry_run_cash_usd
    reserved_cash = 0.0
    tracked_positions: Dict[str, TrackedPosition] = {}
    entered_symbols: set[str] = set()
    closed_trades: List[ClosedTrade] = []
    timeline: List[datetime] = sorted(
        {
            bar.ts_local
            for symbol in symbols
            for bar in bars_by_symbol.get(symbol, [])
            if session_open <= bar.ts_local < session_close
        }
    )

    for now_local in timeline:
        if now_local >= force_close:
            for symbol, position in list(tracked_positions.items()):
                symbol_bars = [bar for bar in bars_by_symbol.get(symbol, []) if bar.ts_local <= now_local]
                if not symbol_bars:
                    continue
                closed = _close_position(
                    tracked_positions=tracked_positions,
                    symbol=symbol,
                    exit_price=symbol_bars[-1].close_price,
                    reason="force_close",
                    closed_at=now_local.isoformat(),
                )
                if closed is not None:
                    reserved_cash = max(0.0, reserved_cash - closed.entry_notional_usd)
                    closed_trades.append(closed)
            break

        for symbol in symbols:
            symbol_bars = [bar for bar in bars_by_symbol.get(symbol, []) if bar.ts_local <= now_local]
            if not symbol_bars:
                continue
            current_price = symbol_bars[-1].close_price

            position = tracked_positions.get(symbol)
            if position is not None:
                if _should_take_profit(current_price, position.entry_price, settings.exit.target_pct):
                    reason = "target"
                elif _should_stop_loss(current_price, position.entry_price, settings.exit.stop_pct):
                    reason = "stop"
                else:
                    reason = ""
                if reason:
                    closed = _close_position(
                        tracked_positions=tracked_positions,
                        symbol=symbol,
                        exit_price=current_price,
                        reason=reason,
                        closed_at=now_local.isoformat(),
                    )
                    if closed is not None:
                        reserved_cash = max(0.0, reserved_cash - closed.entry_notional_usd)
                        closed_trades.append(closed)
                continue

            if now_local >= time_cut:
                continue
            if symbol in entered_symbols:
                continue
            if len(entered_symbols) >= settings.entry.max_candidates:
                continue

            candidate = candidate_by_symbol[symbol]
            matched, details = _entry_conditions_met(symbol_bars, candidate.open_price, settings.entry)
            if not matched:
                continue

            remaining_slots = max(1, settings.entry.max_candidates - len(entered_symbols))
            remaining_budget = max(0.0, buying_power - reserved_cash)
            qty, _ = _calculate_entry_qty(
                current_price=details["current_price"],
                stop_pct=settings.exit.stop_pct,
                remaining_budget_usd=remaining_budget,
                remaining_slots=remaining_slots,
                entry_settings=settings.entry,
            )
            if qty <= 0:
                continue

            entry_price = details["current_price"]
            reserved_cash += entry_price * qty
            tracked_positions[symbol] = TrackedPosition(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                opened_at=now_local.isoformat(),
            )
            entered_symbols.add(symbol)

    summary = {
        "status": "complete",
        "session_date": session_date.isoformat(),
        "symbols": symbols,
        **_summarize_closed_trades(closed_trades),
        "open_positions": sorted(tracked_positions.keys()),
        "trades": [
            {
                "symbol": item.symbol,
                "qty": item.qty,
                "entry_price": item.entry_price,
                "exit_price": item.exit_price,
                "exit_reason": item.exit_reason,
                "realized_pnl_usd": round(item.realized_pnl_usd, 6),
                "realized_pnl_pct": round(item.realized_pnl_pct, 6),
                "opened_at": item.opened_at,
                "closed_at": item.closed_at,
            }
            for item in closed_trades
        ],
    }
    _emit(logger, "scan_replay_complete", **summary)
    return summary


def run_gap_trader(
    cfg: AppConfig,
    gap_cfg: Mapping[str, Any],
    candidates: Sequence[Any],
    data_client: AlpacaClient,
    order_broker: OrderBroker,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = True,
    max_loops: Optional[int] = None,
) -> Dict[str, Any]:
    settings = _load_settings(gap_cfg)
    if not candidates:
        return {"entries": 0, "exits": 0, "open_positions": [], "status": "no_candidates"}

    parsed = [_extract_candidate(item) for item in candidates]
    parsed = parsed[: settings.entry.max_candidates]
    candidate_by_symbol = {item.symbol: item for item in parsed}
    symbols = sorted(candidate_by_symbol.keys())

    tz = ZoneInfo(cfg.project.timezone)
    now_local = datetime.now(tz)
    session_date = now_local.date()
    session_open = datetime.combine(session_date, dtime(9, 30), tzinfo=tz)
    time_cut = datetime.combine(session_date, dtime(settings.exit.time_cut_hour, 0), tzinfo=tz)
    force_close = datetime.combine(
        session_date,
        dtime(settings.exit.force_close_hour, settings.exit.force_close_minute),
        tzinfo=tz,
    )

    tracked_positions = _bootstrap_positions(order_broker, set(symbols), logger)
    entered_symbols = set(tracked_positions.keys())
    entries = 0
    exits = 0
    exit_reasons: Dict[str, str] = {}
    closed_trades: List[ClosedTrade] = []
    loops = 0
    batch_size = max(1, min(200, int(cfg.bars.batch_size)))
    buying_power = _available_buying_power(
        order_broker=order_broker,
        dry_run=dry_run,
        dry_run_cash_usd=settings.entry.dry_run_cash_usd,
        logger=logger,
    )
    reserved_cash = 0.0

    _emit(
        logger,
        "gap_trader_start",
        symbols=symbols,
        dry_run=dry_run,
        force_close_at=force_close.isoformat(),
        buying_power_usd=buying_power,
        risk_per_trade_usd=settings.entry.risk_per_trade_usd,
    )

    while True:
        now_local = datetime.now(tz)
        if max_loops is not None and loops >= max_loops:
            _emit(logger, "gap_trader_max_loops_reached", loops=loops)
            break

        if now_local >= force_close:
            force_close_bars: Dict[str, List[MinuteBar]] = {symbol: [] for symbol in tracked_positions.keys()}
            tracked_symbols = sorted(tracked_positions.keys())
            for batch in _chunk(tracked_symbols, batch_size):
                try:
                    batch_raw = _fetch_bars_batch(
                        client=data_client,
                        cfg=cfg,
                        symbols=batch,
                        start_local=session_open,
                        end_local=now_local,
                    )
                except Exception as exc:
                    _emit(logger, "force_close_bar_fetch_failed", error=str(exc), batch=batch)
                    continue
                for symbol in batch:
                    force_close_bars[symbol] = _normalize_minute_bars(
                        batch_raw.get(symbol, []), tz=tz, session_date=session_date
                    )
            for symbol, position in list(tracked_positions.items()):
                try:
                    symbol_bars = force_close_bars.get(symbol, [])
                    latest_close = _latest_close_price(symbol_bars)
                    exit_price = position.entry_price if latest_close is None else latest_close
                    order = _submit_order(
                        order_broker=order_broker,
                        symbol=symbol,
                        qty=position.qty,
                        side="sell",
                        dry_run=dry_run,
                    )
                    exits += 1
                    exit_reasons[symbol] = "force_close"
                    closed = _close_position(
                        tracked_positions=tracked_positions,
                        symbol=symbol,
                        exit_price=exit_price,
                        reason="force_close",
                        closed_at=now_local.isoformat(),
                    )
                    _emit(
                        logger,
                        "force_close_exit",
                        symbol=symbol,
                        qty=position.qty,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        realized_pnl_usd=0.0 if closed is None else round(closed.realized_pnl_usd, 6),
                        realized_pnl_pct=0.0 if closed is None else round(closed.realized_pnl_pct, 6),
                        order_id=order.order_id,
                    )
                    if closed is not None:
                        reserved_cash = max(0.0, reserved_cash - closed.entry_notional_usd)
                        closed_trades.append(closed)
                except Exception as exc:
                    _emit(logger, "force_close_failed", symbol=symbol, error=str(exc))
            break

        if now_local >= time_cut and not tracked_positions:
            _emit(logger, "time_cut_reached_no_positions")
            break

        bars_by_symbol: Dict[str, List[MinuteBar]] = {symbol: [] for symbol in symbols}
        for batch in _chunk(symbols, batch_size):
            try:
                batch_raw = _fetch_bars_batch(
                    client=data_client,
                    cfg=cfg,
                    symbols=batch,
                    start_local=session_open,
                    end_local=now_local,
                )
            except Exception as exc:
                _emit(logger, "minute_bar_fetch_failed", error=str(exc), batch=batch)
                continue
            for symbol in batch:
                bars_by_symbol[symbol] = _normalize_minute_bars(
                    batch_raw.get(symbol, []), tz=tz, session_date=session_date
                )

        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])
            if not bars:
                continue
            current_price = bars[-1].close_price

            position = tracked_positions.get(symbol)
            if position is not None:
                if _should_take_profit(current_price, position.entry_price, settings.exit.target_pct):
                    reason = "target"
                elif _should_stop_loss(current_price, position.entry_price, settings.exit.stop_pct):
                    reason = "stop"
                else:
                    reason = ""
                if reason:
                    try:
                        order = _submit_order(
                            order_broker=order_broker,
                            symbol=symbol,
                            qty=position.qty,
                            side="sell",
                            dry_run=dry_run,
                        )
                        exits += 1
                        exit_reasons[symbol] = reason
                        closed = _close_position(
                            tracked_positions=tracked_positions,
                            symbol=symbol,
                            exit_price=current_price,
                            reason=reason,
                            closed_at=now_local.isoformat(),
                        )
                        _emit(
                            logger,
                            "exit_filled",
                            symbol=symbol,
                            reason=reason,
                            qty=position.qty,
                            entry_price=position.entry_price,
                            exit_price=current_price,
                            realized_pnl_usd=0.0 if closed is None else round(closed.realized_pnl_usd, 6),
                            realized_pnl_pct=0.0 if closed is None else round(closed.realized_pnl_pct, 6),
                            order_id=order.order_id,
                        )
                        if closed is not None:
                            reserved_cash = max(0.0, reserved_cash - closed.entry_notional_usd)
                            closed_trades.append(closed)
                    except Exception as exc:
                        _emit(logger, "exit_failed", symbol=symbol, reason=reason, error=str(exc))
                continue

            if now_local >= time_cut:
                continue
            if symbol in entered_symbols:
                continue
            if len(entered_symbols) >= settings.entry.max_candidates:
                continue

            candidate = candidate_by_symbol[symbol]
            matched, details = _entry_conditions_met(bars, candidate.open_price, settings.entry)
            if not matched:
                _emit(logger, "entry_check", symbol=symbol, matched=False, **details)
                continue

            remaining_slots = max(1, settings.entry.max_candidates - len(entered_symbols))
            remaining_budget = max(0.0, buying_power - reserved_cash)
            qty, sizing = _calculate_entry_qty(
                current_price=details["current_price"],
                stop_pct=settings.exit.stop_pct,
                remaining_budget_usd=remaining_budget,
                remaining_slots=remaining_slots,
                entry_settings=settings.entry,
            )
            if qty <= 0:
                _emit(logger, "entry_skip_qty", symbol=symbol, remaining_budget=remaining_budget, **sizing, **details)
                continue

            try:
                order = _submit_order(
                    order_broker=order_broker,
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    dry_run=dry_run,
                )
                entry_price = _to_float(order.filled_avg_price, details["current_price"])
                if entry_price <= 0:
                    entry_price = details["current_price"]
                reserved_cash += entry_price * qty
                tracked_positions[symbol] = TrackedPosition(
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    opened_at=now_local.isoformat(),
                )
                entered_symbols.add(symbol)
                entries += 1
                _emit(
                    logger,
                    "entry_filled",
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    order_id=order.order_id,
                    remaining_budget=max(0.0, buying_power - reserved_cash),
                    **sizing,
                    **details,
                )
            except Exception as exc:
                _emit(logger, "entry_failed", symbol=symbol, qty=qty, error=str(exc), **sizing, **details)

        loops += 1
        if max_loops is not None and loops >= max_loops:
            continue
        _sleep_to_next_minute(now_local)

    pnl_summary = _summarize_closed_trades(closed_trades)
    _emit(
        logger,
        "gap_trader_complete",
        entries=entries,
        exits=exits,
        open_positions=sorted(tracked_positions.keys()),
        exit_reasons=exit_reasons,
        **pnl_summary,
    )
    return {
        "entries": entries,
        "exits": exits,
        "open_positions": sorted(tracked_positions.keys()),
        "exit_reasons": exit_reasons,
        **pnl_summary,
        "status": "complete",
    }
