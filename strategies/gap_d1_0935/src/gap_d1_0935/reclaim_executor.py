from __future__ import annotations

from collections import Counter
import warnings
from dataclasses import dataclass
from datetime import date, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .analysis_phase1 import _analysis_trade_dates, _build_daily_frame, _build_symbol_profiles
from .analysis_regime import _fetch_minute_day_map, _intraday_vwap_series, _valid_minute_rows, _daily_watchlist_rows
from .common import ET, get_previous_trading_day, load_alpaca_client, load_seed_symbols
from .config import StrategyConfig, load_strategy_config
from .gap_0935_watchlist_scanner import _inspect_first5_window
from .logutil import build_strategy_logger, log_event
from .paths import reports_dir


SUPPORTED_BRANCHES = {"reclaim_first5_high", "reclaim_vwap", "continuation_compare"}


@dataclass(frozen=True)
class ReclaimExecutorResult:
    trade_date: date
    log_path: Path
    daily_report_path: Path
    branch_compare_path: Path


@dataclass(frozen=True)
class ReclaimPeriodReplayResult:
    start_date: date
    end_date: date
    compare_path: Path
    daily_report_path: Path
    summary_path: Path
    gate_fail_daily_path: Path
    gate_fail_symbol_path: Path
    trade_days: int


def _liquidity_bucket(avg_dollar_volume_20: float) -> str:
    if avg_dollar_volume_20 >= 20_000_000:
        return "high"
    if avg_dollar_volume_20 >= 5_000_000:
        return "mid"
    if avg_dollar_volume_20 >= 2_000_000:
        return "low"
    return "micro"


def _slippage_bps_for_bucket(cfg: StrategyConfig, bucket: str) -> float:
    mapping = {
        "high": cfg.cost.slippage_bps_high_liquidity,
        "mid": cfg.cost.slippage_bps_mid_liquidity,
        "low": cfg.cost.slippage_bps_low_liquidity,
        "micro": cfg.cost.slippage_bps_micro_liquidity,
    }
    return float(mapping[bucket])


def _entry_row_for_level(minute_rows: List[Dict[str, Any]], level: float) -> Optional[Dict[str, Any]]:
    for row in minute_rows:
        if row["ts"].time() < dtime(9, 36) or row["ts"].time() > dtime(10, 0):
            continue
        if row["high"] > level:
            return row
    return None


def _entry_row_for_vwap(minute_rows_with_vwap: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in minute_rows_with_vwap:
        if row["ts"].time() < dtime(9, 36) or row["ts"].time() > dtime(10, 0):
            continue
        if row["high"] > row["intraday_vwap"]:
            return row
    return None


def _simulate_trade(
    minute_rows: List[Dict[str, Any]],
    entry_row: Optional[Dict[str, Any]],
    entry_price: Optional[float],
    stop_price: Optional[float],
    qty: int,
    slippage_bps_per_side: float,
    fee_bps_round_trip: float,
) -> Dict[str, Any]:
    if entry_row is None or entry_price is None or stop_price is None or entry_price <= stop_price or qty <= 0:
        return {
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "realized_pnl_usd": None,
            "realized_pnl_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "holding_minutes": None,
            "effective_entry_price": None,
            "effective_stop_price": None,
            "target_price": None,
        }
    effective_entry_price = entry_price * (1.0 + slippage_bps_per_side / 10000.0)
    effective_stop_price = stop_price * (1.0 - slippage_bps_per_side / 10000.0)
    risk_per_share = effective_entry_price - effective_stop_price
    if risk_per_share <= 0:
        return {
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "realized_pnl_usd": None,
            "realized_pnl_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "holding_minutes": None,
            "effective_entry_price": effective_entry_price,
            "effective_stop_price": effective_stop_price,
            "target_price": None,
        }
    target_price = entry_price + 2.0 * risk_per_share
    cutoff = dtime(15, 55)
    post_entry = [row for row in minute_rows if row["ts"] >= entry_row["ts"] and row["ts"].time() <= cutoff]
    if not post_entry:
        return {
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "realized_pnl_usd": None,
            "realized_pnl_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "holding_minutes": None,
            "effective_entry_price": effective_entry_price,
            "effective_stop_price": effective_stop_price,
            "target_price": target_price,
        }
    exit_row = post_entry[-1]
    exit_price = exit_row["close"]
    exit_reason = "time_exit"
    for row in post_entry:
        if row["low"] <= stop_price:
            exit_row = row
            exit_price = effective_stop_price
            exit_reason = "stop"
            break
        if row["high"] >= target_price:
            exit_row = row
            exit_price = target_price * (1.0 - slippage_bps_per_side / 10000.0)
            exit_reason = "target"
            break
    if exit_reason == "time_exit":
        exit_price = exit_price * (1.0 - slippage_bps_per_side / 10000.0)
    gross_realized_pnl_usd = (exit_price - effective_entry_price) * qty
    fee_usd = effective_entry_price * qty * (fee_bps_round_trip / 10000.0)
    realized_pnl_usd = gross_realized_pnl_usd - fee_usd
    gross_realized_pnl_pct = (exit_price - effective_entry_price) / effective_entry_price * 100.0
    realized_pnl_pct = gross_realized_pnl_pct - fee_bps_round_trip / 100.0
    mfe_pct = max((row["high"] - effective_entry_price) / effective_entry_price * 100.0 for row in post_entry)
    mae_pct = min((row["low"] - effective_entry_price) / effective_entry_price * 100.0 for row in post_entry)
    holding_minutes = int((exit_row["ts"] - entry_row["ts"]).total_seconds() // 60)
    return {
        "exit_time": exit_row["ts"].isoformat(),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "realized_pnl_usd": realized_pnl_usd,
        "realized_pnl_pct": realized_pnl_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "holding_minutes": holding_minutes,
        "effective_entry_price": effective_entry_price,
        "effective_stop_price": effective_stop_price,
        "target_price": target_price,
    }


def _position_size(cfg: StrategyConfig, entry_price: float, stop_price: float) -> tuple[int, float]:
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return 0, risk_per_share
    qty_by_risk = int(cfg.risk.risk_per_trade_usd // risk_per_share)
    qty_by_notional = int(cfg.risk.max_notional_per_trade_usd // entry_price) if entry_price > 0 else 0
    qty = min(qty_by_risk, qty_by_notional)
    if qty < int(cfg.risk.min_order_qty):
        return 0, risk_per_share
    return qty, risk_per_share


def _candidate_frame_for_trade_date_from_context(
    trade_date: date,
    cfg: StrategyConfig,
    universe: List[str],
    grouped_daily: Dict[str, pd.DataFrame],
    profiles_map: Dict[str, Dict[str, Any]],
    client: Any,
) -> pd.DataFrame:
    previous_date = get_previous_trading_day(client, trade_date)

    selected_symbols: List[str] = []
    selected_ctx: Dict[str, Dict[str, Any]] = {}
    for symbol in universe:
        frame = grouped_daily.get(symbol)
        if frame is None or frame.empty:
            continue
        profile = profiles_map.get(symbol, {})
        context = _daily_watchlist_rows(
            symbol=symbol,
            frame=frame,
            trade_date=trade_date,
            previous_date=previous_date,
            cfg=cfg,
            market_cap=profile.get("market_cap"),
            security_type=str(profile.get("security_type") or "common_stock"),
        )
        if context is None:
            continue
        day_frame = frame[frame["date"] == trade_date]
        if day_frame.empty:
            continue
        selected_symbols.append(symbol)
        selected_ctx[symbol] = {**context, "day_bar": day_frame.iloc[-1].to_dict(), "profile": profile}

    minute_map = _fetch_minute_day_map(selected_symbols, trade_date)
    rows: List[Dict[str, Any]] = []
    for symbol in selected_symbols:
        ctx = selected_ctx[symbol]
        minute_rows = _valid_minute_rows(minute_map.get(symbol, []))
        minute_rows_with_vwap = _intraday_vwap_series(minute_rows)
        agg = _inspect_first5_window(symbol, minute_map.get(symbol, [])).get("aggregate")
        if agg is None:
            continue
        day_bar = ctx["day_bar"]
        open_d = float(day_bar["open"])
        close_prev = float(ctx["close_D-1"])
        regime_label = (
            "regime_b_open_at_or_below_prev_close" if open_d <= close_prev else "regime_a_open_above_prev_close"
        )
        first5_open = agg["first5_open"]
        first5_high = agg["first5_high"]
        first5_low = agg["first5_low"]
        first5_close = agg["first5_close"]
        first5_range = first5_high - first5_low
        first5_range_pos = (first5_close - first5_low) / first5_range if first5_range > 0 else None
        first5_pace = (
            (agg["first5_volume"] * 78.0) / float(ctx["avg_volume_20"]) if float(ctx["avg_volume_20"]) > 0 else None
        )
        vwap = float(agg["vwap"])
        close_vs_vwap = (first5_close / vwap - 1.0) * 100.0 if vwap > 0 else None
        open_gap_vs_prev_close = (open_d - close_prev) / close_prev * 100.0 if close_prev > 0 else None
        first5_oc_ret = (first5_close - first5_open) / first5_open * 100.0 if first5_open > 0 else None
        continuation_gate = bool(
            regime_label == "regime_a_open_above_prev_close"
            and open_gap_vs_prev_close is not None
            and open_gap_vs_prev_close >= cfg.day0.min_gap_today_pct
            and first5_range_pos is not None
            and first5_range_pos >= cfg.day0.min_first5_range_pos
            and first5_oc_ret is not None
            and first5_oc_ret >= cfg.day0.min_first5_oc_ret
            and first5_pace is not None
            and first5_pace >= cfg.day0.min_first5_pace
            and close_vs_vwap is not None
            and close_vs_vwap >= (cfg.day0.min_close_vs_vwap_ratio - 1.0) * 100.0
        )
        reclaim_gate = bool(
            regime_label == "regime_b_open_at_or_below_prev_close"
            and close_vs_vwap is not None
            and close_vs_vwap >= 0.0
            and first5_range_pos is not None
            and first5_range_pos >= 0.50
            and first5_pace is not None
            and first5_pace >= 1.0
        )
        first_vwap_row = _entry_row_for_vwap(minute_rows_with_vwap) if reclaim_gate else None
        first5_high_row = (
            _entry_row_for_level(minute_rows, first5_high) if (reclaim_gate or continuation_gate) else None
        )
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date.isoformat(),
                "regime_label": regime_label,
                "open_D": open_d,
                "close_D-1": close_prev,
                "first5_high": first5_high,
                "first5_low": first5_low,
                "first5_close": first5_close,
                "vwap": vwap,
                "first5_range_pos": first5_range_pos,
                "first5_pace": first5_pace,
                "close_vs_vwap": close_vs_vwap,
                "continuation_gate": continuation_gate,
                "reclaim_gate": reclaim_gate,
                "reclaim_first5_high_trigger_time": (
                    first5_high_row["ts"].isoformat()
                    if (first5_high_row and reclaim_gate and regime_label == "regime_b_open_at_or_below_prev_close")
                    else None
                ),
                "reclaim_vwap_trigger_time": first_vwap_row["ts"].isoformat() if first_vwap_row else None,
                "reclaim_vwap_trigger_price": float(first_vwap_row["intraday_vwap"]) if first_vwap_row else None,
                "continuation_trigger_time": (
                    first5_high_row["ts"].isoformat() if (first5_high_row and continuation_gate) else None
                ),
                "avg_dollar_volume_20": float(ctx["avg_volume_20"]) * float(ctx["close_D-1"]),
                "minute_rows": minute_rows,
            }
        )
    return pd.DataFrame(rows)


def _candidate_frame_for_trade_date(trade_date: date, cfg: StrategyConfig) -> pd.DataFrame:
    _, client = load_alpaca_client()
    universe = load_seed_symbols()
    daily_df = _build_daily_frame(universe, trade_date, trade_date)
    grouped_daily = {
        symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in daily_df.groupby("symbol")
    }
    profiles_df = _build_symbol_profiles(universe, set(universe), reports_dir() / "symbol_profiles_cache.csv")
    profiles_map = {row["symbol"]: row for row in profiles_df.to_dict(orient="records")}
    return _candidate_frame_for_trade_date_from_context(
        trade_date=trade_date,
        cfg=cfg,
        universe=universe,
        grouped_daily=grouped_daily,
        profiles_map=profiles_map,
        client=client,
    )


def _candidate_frame_from_scanner_csv(path: Path, cfg: StrategyConfig) -> pd.DataFrame:
    scan_df = pd.read_csv(path)
    if scan_df.empty:
        return pd.DataFrame()
    scan_df["symbol"] = scan_df["symbol"].astype(str).str.upper()
    trade_date_raw = str(scan_df["trade_date"].dropna().iloc[0])
    trade_date = date.fromisoformat(trade_date_raw)
    symbols = scan_df["symbol"].dropna().astype(str).tolist()
    minute_map = _fetch_minute_day_map(symbols, trade_date)
    rows: List[Dict[str, Any]] = []
    for rec in scan_df.to_dict(orient="records"):
        symbol = str(rec["symbol"]).upper()
        minute_rows = _valid_minute_rows(minute_map.get(symbol, []))
        minute_rows_with_vwap = _intraday_vwap_series(minute_rows)
        open_d = pd.to_numeric(rec.get("open_D"), errors="coerce")
        close_prev = pd.to_numeric(rec.get("close_D-1"), errors="coerce")
        first5_high = pd.to_numeric(rec.get("first5_high"), errors="coerce")
        first5_low = pd.to_numeric(rec.get("first5_low"), errors="coerce")
        first5_close = pd.to_numeric(rec.get("first5_close"), errors="coerce")
        vwap = pd.to_numeric(rec.get("vwap"), errors="coerce")
        first5_range_pos = pd.to_numeric(rec.get("first5_range_pos"), errors="coerce")
        first5_pace = pd.to_numeric(rec.get("first5_pace"), errors="coerce")
        if pd.isna(open_d) or pd.isna(close_prev) or pd.isna(first5_high) or pd.isna(first5_low):
            continue
        regime_label = (
            "regime_b_open_at_or_below_prev_close"
            if float(open_d) <= float(close_prev)
            else "regime_a_open_above_prev_close"
        )
        close_vs_vwap = None
        if pd.notna(first5_close) and pd.notna(vwap) and float(vwap) > 0:
            close_vs_vwap = (float(first5_close) / float(vwap) - 1.0) * 100.0
        first5_high_row = _entry_row_for_level(minute_rows, float(first5_high))
        first_vwap_row = _entry_row_for_vwap(minute_rows_with_vwap)
        reclaim_gate = bool(
            regime_label == "regime_b_open_at_or_below_prev_close"
            and close_vs_vwap is not None
            and close_vs_vwap >= 0.0
            and pd.notna(first5_range_pos)
            and float(first5_range_pos) >= 0.50
            and pd.notna(first5_pace)
            and float(first5_pace) >= 1.0
        )
        continuation_gate = bool(rec.get("pass")) and regime_label == "regime_a_open_above_prev_close"
        avg_dollar_volume_20 = None
        avg_volume_20 = pd.to_numeric(rec.get("avg_volume_20"), errors="coerce")
        if pd.notna(avg_volume_20) and float(avg_volume_20) > 0 and float(close_prev) > 0:
            avg_dollar_volume_20 = float(avg_volume_20) * float(close_prev)
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date.isoformat(),
                "regime_label": regime_label,
                "open_D": float(open_d),
                "close_D-1": float(close_prev),
                "first5_high": float(first5_high),
                "first5_low": float(first5_low),
                "first5_close": float(first5_close) if pd.notna(first5_close) else None,
                "vwap": float(vwap) if pd.notna(vwap) else None,
                "first5_range_pos": float(first5_range_pos) if pd.notna(first5_range_pos) else None,
                "first5_pace": float(first5_pace) if pd.notna(first5_pace) else None,
                "close_vs_vwap": close_vs_vwap,
                "continuation_gate": continuation_gate,
                "reclaim_gate": reclaim_gate,
                "reclaim_first5_high_trigger_time": (
                    first5_high_row["ts"].isoformat()
                    if (first5_high_row and reclaim_gate and regime_label == "regime_b_open_at_or_below_prev_close")
                    else None
                ),
                "reclaim_vwap_trigger_time": first_vwap_row["ts"].isoformat() if first_vwap_row else None,
                "reclaim_vwap_trigger_price": float(first_vwap_row["intraday_vwap"]) if first_vwap_row else None,
                "continuation_trigger_time": (
                    first5_high_row["ts"].isoformat() if (first5_high_row and continuation_gate) else None
                ),
                "avg_dollar_volume_20": avg_dollar_volume_20,
                "minute_rows": minute_rows,
            }
        )
    return pd.DataFrame(rows)


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None:
            continue
        cleaned = frame.dropna(how="all").copy()
        if cleaned.empty and len(cleaned.columns) == 0:
            continue
        if cleaned.empty:
            continue
        valid_frames.append(cleaned)
    if not valid_frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        return pd.concat(valid_frames, ignore_index=True)


def _execute_branch_on_candidates(
    trade_date: date,
    branch: str,
    candidate_df: pd.DataFrame,
    cfg: StrategyConfig,
    logger: Any,
    actual_fee_bps: float,
    slippage_bps_per_side: Optional[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    log_event(
        logger,
        "branch_candidate_loaded",
        trade_date=trade_date.isoformat(),
        branch=branch,
        mode="dry_run",
        scanned_candidates=int(len(candidate_df)),
    )
    if candidate_df.empty:
        empty_df = pd.DataFrame(
            [
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": None,
                    "candidate_count": 0,
                    "entry_count": 0,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": 0.0,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": 0,
                    "trigger_touched": 0,
                    "gate_fail_reason": "no_candidates",
                    "skip_reason": "no_candidates",
                }
            ]
        )
        meta = {
            "scanned_candidates": 0,
            "entries": 0,
            "exits": 0,
            "closed_trades": 0,
            "realized_pnl_usd": 0.0,
            "realized_pnl_pct": None,
            "exit_reason_breakdown": {},
        }
        return empty_df, meta

    branch_rows: List[Dict[str, Any]] = []
    exit_breakdown: Counter[str] = Counter()
    entries = 0
    exits = 0

    for rec in candidate_df.to_dict(orient="records"):
        symbol = str(rec["symbol"])
        minute_rows = rec["minute_rows"]
        regime_label = str(rec["regime_label"])
        first5_range_pos = pd.to_numeric(rec.get("first5_range_pos"), errors="coerce")
        first5_pace = pd.to_numeric(rec.get("first5_pace"), errors="coerce")
        close_vs_vwap = pd.to_numeric(rec.get("close_vs_vwap"), errors="coerce")
        if branch == "continuation_compare":
            regime_ok = regime_label == "regime_a_open_above_prev_close"
            vwap_ok = (
                pd.notna(close_vs_vwap) and float(close_vs_vwap) >= (cfg.day0.min_close_vs_vwap_ratio - 1.0) * 100.0
            )
            range_ok = pd.notna(first5_range_pos) and float(first5_range_pos) >= cfg.day0.min_first5_range_pos
            pace_ok = pd.notna(first5_pace) and float(first5_pace) >= cfg.day0.min_first5_pace
            gate_ok = bool(rec["continuation_gate"])
        else:
            regime_ok = regime_label == "regime_b_open_at_or_below_prev_close"
            vwap_ok = pd.notna(close_vs_vwap) and float(close_vs_vwap) >= 0.0
            range_ok = pd.notna(first5_range_pos) and float(first5_range_pos) >= 0.50
            pace_ok = pd.notna(first5_pace) and float(first5_pace) >= 1.0
            gate_ok = bool(rec["reclaim_gate"])
        gate_fail_parts: List[str] = []
        if not regime_ok:
            gate_fail_parts.append("regime_fail")
        if not vwap_ok:
            gate_fail_parts.append("vwap_fail")
        if not range_ok:
            gate_fail_parts.append("range_fail")
        if not pace_ok:
            gate_fail_parts.append("pace_fail")
        gate_fail_reason = "|".join(gate_fail_parts)
        trigger_time_key = {
            "reclaim_first5_high": "reclaim_first5_high_trigger_time",
            "reclaim_vwap": "reclaim_vwap_trigger_time",
            "continuation_compare": "continuation_trigger_time",
        }[branch]
        trigger_time = rec.get(trigger_time_key)
        gate_pass = int(regime_ok and gate_ok)
        trigger_touched = int(regime_ok and gate_ok and bool(trigger_time))
        skip_reason = None
        entry_reason = branch

        if branch != "continuation_compare" and regime_label != "regime_b_open_at_or_below_prev_close":
            skip_reason = "not_regime_b"
        elif not gate_ok:
            skip_reason = "gate_fail"
        elif not trigger_time:
            skip_reason = "no_trigger"
        elif pd.to_datetime(trigger_time).tz_convert(ET).time() > dtime(10, 0):
            skip_reason = "entry_window_closed"

        log_event(
            logger,
            "branch_entry_check",
            trade_date=trade_date.isoformat(),
            branch=branch,
            symbol=symbol,
            mode="dry_run",
            skip_reason=skip_reason,
            trigger_time=trigger_time,
        )

        if skip_reason:
            branch_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": symbol,
                    "candidate_count": 1,
                    "entry_count": 0,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": None,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": gate_pass,
                    "trigger_touched": trigger_touched,
                    "gate_fail_reason": gate_fail_reason if skip_reason in {"gate_fail", "not_regime_b"} else None,
                    "skip_reason": skip_reason,
                }
            )
            continue

        trigger_dt = pd.to_datetime(trigger_time).tz_convert(ET)
        entry_row = next((row for row in minute_rows if row["ts"] == trigger_dt), None)
        if entry_row is None:
            skip_reason = "missing_price"
            branch_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": symbol,
                    "candidate_count": 1,
                    "entry_count": 0,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": None,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": gate_pass,
                    "trigger_touched": trigger_touched,
                    "gate_fail_reason": None,
                    "skip_reason": skip_reason,
                }
            )
            continue

        raw_entry_price = (
            float(rec["reclaim_vwap_trigger_price"]) if branch == "reclaim_vwap" else float(rec["first5_high"])
        )
        stop_price = float(rec["first5_low"])
        liquidity_bucket = _liquidity_bucket(float(rec.get("avg_dollar_volume_20") or 0.0))
        applied_slippage_bps = (
            _slippage_bps_for_bucket(cfg, liquidity_bucket)
            if slippage_bps_per_side is None
            else float(slippage_bps_per_side)
        )
        qty, risk_per_share = _position_size(cfg, raw_entry_price, stop_price)
        if qty <= 0:
            skip_reason = "invalid_risk"
            branch_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": symbol,
                    "candidate_count": 1,
                    "entry_count": 0,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": None,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": gate_pass,
                    "trigger_touched": trigger_touched,
                    "gate_fail_reason": None,
                    "skip_reason": skip_reason,
                }
            )
            continue

        entries += 1
        trade = _simulate_trade(
            minute_rows,
            entry_row,
            raw_entry_price,
            stop_price,
            qty,
            slippage_bps_per_side=applied_slippage_bps,
            fee_bps_round_trip=actual_fee_bps,
        )
        effective_entry_price = trade["effective_entry_price"]
        effective_stop_price = trade["effective_stop_price"]
        target_price = trade["target_price"]
        if effective_entry_price is None or effective_stop_price is None or target_price is None:
            skip_reason = "invalid_risk"
            branch_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": symbol,
                    "candidate_count": 1,
                    "entry_count": 0,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": None,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": gate_pass,
                    "trigger_touched": trigger_touched,
                    "gate_fail_reason": None,
                    "skip_reason": skip_reason,
                }
            )
            continue
        log_event(
            logger,
            "branch_entry_filled",
            trade_date=trade_date.isoformat(),
            branch=branch,
            symbol=symbol,
            mode="dry_run",
            entry_price=effective_entry_price,
            stop_price=stop_price,
            target_price=target_price,
            qty=qty,
            risk_per_share=risk_per_share,
            entry_reason=entry_reason,
            slippage_bps_per_side=applied_slippage_bps,
            fee_bps_round_trip=actual_fee_bps,
            effective_stop_price=effective_stop_price,
            timestamp=trigger_time,
        )
        if trade["realized_pnl_usd"] is None:
            skip_reason = "force_exit_only"
            branch_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "branch": branch,
                    "symbol": symbol,
                    "candidate_count": 1,
                    "entry_count": 1,
                    "closed_trades": 0,
                    "win_rate": None,
                    "realized_pnl_usd": None,
                    "realized_pnl_pct": None,
                    "avg_hold_min": None,
                    "gate_pass": gate_pass,
                    "trigger_touched": trigger_touched,
                    "gate_fail_reason": None,
                    "skip_reason": skip_reason,
                }
            )
            continue

        exits += 1
        exit_breakdown[str(trade["exit_reason"])] += 1
        exit_event = "branch_force_exit" if trade["exit_reason"] == "time_exit" else "branch_exit_filled"
        log_event(
            logger,
            exit_event,
            trade_date=trade_date.isoformat(),
            branch=branch,
            symbol=symbol,
            mode="dry_run",
            exit_price=trade["exit_price"],
            exit_reason=trade["exit_reason"],
            realized_pnl_usd=trade["realized_pnl_usd"],
            realized_pnl_pct=trade["realized_pnl_pct"],
            mfe_pct=trade["mfe_pct"],
            mae_pct=trade["mae_pct"],
            holding_minutes=trade["holding_minutes"],
            timestamp=trade["exit_time"],
        )

        branch_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "branch": branch,
                "symbol": symbol,
                "candidate_count": 1,
                "entry_count": 1,
                "closed_trades": 1,
                "win_rate": 1.0 if trade["realized_pnl_pct"] > 0 else 0.0,
                "realized_pnl_usd": trade["realized_pnl_usd"],
                "realized_pnl_pct": trade["realized_pnl_pct"],
                "avg_hold_min": trade["holding_minutes"],
                "gate_pass": gate_pass,
                "trigger_touched": trigger_touched,
                "gate_fail_reason": None,
                "entry_price": effective_entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_price": trade["exit_price"],
                "exit_reason": trade["exit_reason"],
                "qty": qty,
                "risk_per_share": risk_per_share,
                "mfe_pct": trade["mfe_pct"],
                "mae_pct": trade["mae_pct"],
                "slippage_bps_per_side": applied_slippage_bps,
                "fee_bps_round_trip": actual_fee_bps,
                "effective_stop_price": effective_stop_price,
                "skip_reason": None,
            }
        )

    branch_df = pd.DataFrame(branch_rows)
    meta = {
        "scanned_candidates": int(len(candidate_df)),
        "entries": entries,
        "exits": exits,
        "closed_trades": int(branch_df["closed_trades"].fillna(0).sum()) if not branch_df.empty else 0,
        "realized_pnl_usd": (
            float(pd.to_numeric(branch_df["realized_pnl_usd"], errors="coerce").sum()) if not branch_df.empty else 0.0
        ),
        "realized_pnl_pct": (
            float(pd.to_numeric(branch_df["realized_pnl_pct"], errors="coerce").mean()) if not branch_df.empty else None
        ),
        "exit_reason_breakdown": dict(exit_breakdown),
    }
    return branch_df, meta


def run_reclaim_executor(
    trade_date: date,
    branch: str,
    dry_run: bool = True,
    from_scanner_csv: str | None = None,
    config: Optional[StrategyConfig] = None,
    slippage_bps_per_side: Optional[float] = None,
    fee_bps_round_trip: Optional[float] = None,
) -> ReclaimExecutorResult:
    if branch not in SUPPORTED_BRANCHES:
        raise ValueError(f"Unsupported branch: {branch}")
    if not dry_run:
        raise ValueError("Only dry_run is supported")
    cfg = config or load_strategy_config()
    actual_fee_bps = float(cfg.cost.fee_bps_round_trip if fee_bps_round_trip is None else fee_bps_round_trip)
    logger, log_path = build_strategy_logger("gap_d1_0935_reclaim_executor", "reclaim_executor")
    log_event(logger, "branch_executor_start", trade_date=trade_date.isoformat(), branch=branch, mode="dry_run")

    if from_scanner_csv:
        candidate_df = _candidate_frame_from_scanner_csv(Path(from_scanner_csv), cfg)
    else:
        candidate_df = _candidate_frame_for_trade_date(trade_date, cfg)
    branch_df, meta = _execute_branch_on_candidates(
        trade_date=trade_date,
        branch=branch,
        candidate_df=candidate_df,
        cfg=cfg,
        logger=logger,
        actual_fee_bps=actual_fee_bps,
        slippage_bps_per_side=slippage_bps_per_side,
    )

    daily_summary = (
        branch_df.groupby(["trade_date", "branch"], dropna=False)
        .agg(
            candidate_count=("candidate_count", "sum"),
            entry_count=("entry_count", "sum"),
            closed_trades=("closed_trades", "sum"),
            win_rate=("win_rate", "mean"),
            realized_pnl_usd=("realized_pnl_usd", "sum"),
            realized_pnl_pct=("realized_pnl_pct", "mean"),
            avg_hold_min=("avg_hold_min", "mean"),
        )
        .reset_index()
    )
    compare_df = branch_df.copy()

    daily_report_path = reports_dir() / f"reclaim_executor_daily_{trade_date:%Y%m%d}.csv"
    branch_compare_path = reports_dir() / f"reclaim_branch_compare_{trade_date:%Y%m%d}.csv"
    if daily_report_path.exists():
        existing = pd.read_csv(daily_report_path)
        daily_summary = _concat_nonempty([existing, daily_summary])
        daily_summary = daily_summary.drop_duplicates(subset=["trade_date", "branch"], keep="last")
    if branch_compare_path.exists():
        existing_compare = pd.read_csv(branch_compare_path)
        compare_df = _concat_nonempty([existing_compare, compare_df])
        compare_df = compare_df.drop_duplicates(subset=["trade_date", "branch", "symbol"], keep="last")
    daily_summary.to_csv(daily_report_path, index=False)
    compare_df.to_csv(branch_compare_path, index=False)

    log_event(
        logger,
        "branch_executor_complete",
        trade_date=trade_date.isoformat(),
        branch=branch,
        mode="dry_run",
        **meta,
    )
    return ReclaimExecutorResult(
        trade_date=trade_date,
        log_path=log_path,
        daily_report_path=daily_report_path,
        branch_compare_path=branch_compare_path,
    )


def replay_reclaim_executor_period(
    branches: List[str],
    months: int = 12,
    end_date: Optional[date] = None,
    trade_dates_override: Optional[List[date]] = None,
    scanner_csv_map: Optional[Dict[date, Path]] = None,
    config: Optional[StrategyConfig] = None,
    slippage_bps_per_side: Optional[float] = None,
    fee_bps_round_trip: Optional[float] = None,
) -> ReclaimPeriodReplayResult:
    cfg = config or load_strategy_config()
    actual_fee_bps = float(cfg.cost.fee_bps_round_trip if fee_bps_round_trip is None else fee_bps_round_trip)
    _, client = load_alpaca_client()
    actual_end_date = end_date or get_previous_trading_day(client, date.today())
    trade_dates = trade_dates_override or _analysis_trade_dates(months=months, end_date=actual_end_date)
    start_date = trade_dates[0]
    universe = load_seed_symbols()
    daily_df = _build_daily_frame(universe, start_date, actual_end_date)
    grouped_daily = {
        symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in daily_df.groupby("symbol")
    }
    profiles_df = _build_symbol_profiles(universe, set(universe), reports_dir() / "symbol_profiles_cache.csv")
    profiles_map = {row["symbol"]: row for row in profiles_df.to_dict(orient="records")}
    logger, log_path = build_strategy_logger("gap_d1_0935_reclaim_executor_period", "reclaim_executor_period")

    period_rows: List[pd.DataFrame] = []
    for trade_date in trade_dates:
        if scanner_csv_map is not None:
            scanner_csv = scanner_csv_map.get(trade_date)
            if scanner_csv is None:
                continue
            candidate_df = _candidate_frame_from_scanner_csv(scanner_csv, cfg)
        else:
            candidate_df = _candidate_frame_for_trade_date_from_context(
                trade_date=trade_date,
                cfg=cfg,
                universe=universe,
                grouped_daily=grouped_daily,
                profiles_map=profiles_map,
                client=client,
            )
        for branch in branches:
            log_event(logger, "branch_executor_start", trade_date=trade_date.isoformat(), branch=branch, mode="dry_run")
            branch_df, meta = _execute_branch_on_candidates(
                trade_date=trade_date,
                branch=branch,
                candidate_df=candidate_df,
                cfg=cfg,
                logger=logger,
                actual_fee_bps=actual_fee_bps,
                slippage_bps_per_side=slippage_bps_per_side,
            )
            log_event(
                logger,
                "branch_executor_complete",
                trade_date=trade_date.isoformat(),
                branch=branch,
                mode="dry_run",
                **meta,
            )
            period_rows.append(branch_df)
    stamp = f"{start_date:%Y%m%d}_{actual_end_date:%Y%m%d}"
    compare_path = reports_dir() / f"reclaim_executor_replay_compare_{stamp}.csv"
    daily_path = reports_dir() / f"reclaim_executor_replay_daily_{stamp}.csv"
    summary_path = reports_dir() / f"reclaim_executor_replay_summary_{stamp}.csv"
    gate_fail_daily_path = reports_dir() / f"reclaim_executor_gate_fail_daily_{stamp}.csv"
    gate_fail_symbol_path = reports_dir() / f"reclaim_executor_gate_fail_symbol_{stamp}.csv"
    if not period_rows:
        compare_df = pd.DataFrame(
            columns=[
                "trade_date",
                "branch",
                "symbol",
                "candidate_count",
                "entry_count",
                "closed_trades",
                "win_rate",
                "realized_pnl_usd",
                "realized_pnl_pct",
                "avg_hold_min",
                "skip_reason",
                "entry_price",
                "stop_price",
                "target_price",
                "exit_price",
                "exit_reason",
            ]
        )
        daily_df_out = pd.DataFrame(
            columns=[
                "trade_date",
                "branch",
                "candidate_count",
                "entry_count",
                "closed_trades",
                "win_rate",
                "realized_pnl_usd",
                "realized_pnl_pct",
                "avg_hold_min",
            ]
        )
        summary_df = pd.DataFrame(
            columns=[
                "branch",
                "trade_days",
                "candidate_count",
                "entry_count",
                "closed_trades",
                "win_rate",
                "realized_pnl_usd",
                "realized_pnl_pct",
                "avg_hold_min",
            ]
        )
        compare_df.to_csv(compare_path, index=False)
        daily_df_out.to_csv(daily_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        pd.DataFrame(columns=["trade_date", "branch", "gate_fail_reason", "count"]).to_csv(
            gate_fail_daily_path, index=False
        )
        pd.DataFrame(columns=["trade_date", "branch", "symbol", "gate_fail_reason", "skip_reason"]).to_csv(
            gate_fail_symbol_path, index=False
        )
        log_event(
            logger,
            "reclaim_executor_period_complete",
            start_date=start_date.isoformat(),
            end_date=actual_end_date.isoformat(),
            trade_days=len(trade_dates),
            compare_path=str(compare_path),
            daily_report_path=str(daily_path),
            summary_path=str(summary_path),
            gate_fail_daily_path=str(gate_fail_daily_path),
            gate_fail_symbol_path=str(gate_fail_symbol_path),
            log_path=str(log_path),
            empty_result=True,
        )
        return ReclaimPeriodReplayResult(
            start_date=start_date,
            end_date=actual_end_date,
            compare_path=compare_path,
            daily_report_path=daily_path,
            summary_path=summary_path,
            gate_fail_daily_path=gate_fail_daily_path,
            gate_fail_symbol_path=gate_fail_symbol_path,
            trade_days=len(trade_dates),
        )

    compare_df = _concat_nonempty(period_rows).sort_values(["trade_date", "branch", "symbol"]).reset_index(drop=True)
    daily_df_out = (
        compare_df.groupby(["trade_date", "branch"], dropna=False)
        .agg(
            candidate_count=("candidate_count", "sum"),
            gate_pass_count=("gate_pass", "sum"),
            trigger_touch_count=("trigger_touched", "sum"),
            entry_count=("entry_count", "sum"),
            closed_trades=("closed_trades", "sum"),
            win_rate=("win_rate", "mean"),
            realized_pnl_usd=("realized_pnl_usd", "sum"),
            realized_pnl_pct=("realized_pnl_pct", "mean"),
            avg_hold_min=("avg_hold_min", "mean"),
            no_trigger_count=("skip_reason", lambda s: int((pd.Series(s).fillna("") == "no_trigger").sum())),
            invalid_risk_count=("skip_reason", lambda s: int((pd.Series(s).fillna("") == "invalid_risk").sum())),
            entry_window_closed_count=(
                "skip_reason",
                lambda s: int((pd.Series(s).fillna("") == "entry_window_closed").sum()),
            ),
        )
        .reset_index()
    )
    exit_counts = (
        compare_df.dropna(subset=["exit_reason"])
        .groupby(["trade_date", "branch", "exit_reason"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if not exit_counts.empty:
        exit_counts = exit_counts.rename(
            columns={col: f"exit_reason_{col}" for col in exit_counts.columns if col not in {"trade_date", "branch"}}
        )
        daily_df_out = daily_df_out.merge(exit_counts, on=["trade_date", "branch"], how="left")
    summary_df = (
        daily_df_out.groupby("branch", dropna=False)
        .agg(
            trade_days=("trade_date", "count"),
            candidate_count=("candidate_count", "sum"),
            gate_pass_count=("gate_pass_count", "sum"),
            trigger_touch_count=("trigger_touch_count", "sum"),
            entry_count=("entry_count", "sum"),
            closed_trades=("closed_trades", "sum"),
            win_rate=("win_rate", "mean"),
            realized_pnl_usd=("realized_pnl_usd", "sum"),
            realized_pnl_pct=("realized_pnl_pct", "mean"),
            avg_hold_min=("avg_hold_min", "mean"),
            no_trigger_count=("no_trigger_count", "sum"),
            invalid_risk_count=("invalid_risk_count", "sum"),
            entry_window_closed_count=("entry_window_closed_count", "sum"),
        )
        .reset_index()
    )
    exit_cols = [col for col in daily_df_out.columns if col.startswith("exit_reason_")]
    if exit_cols:
        summary_df = summary_df.merge(
            daily_df_out.groupby("branch", dropna=False)[exit_cols].sum().reset_index(), on="branch", how="left"
        )

    gate_fail_symbol_df = compare_df[compare_df["gate_fail_reason"].fillna("").ne("") & compare_df["symbol"].notna()][
        ["trade_date", "branch", "symbol", "gate_fail_reason", "skip_reason"]
    ].copy()
    gate_fail_daily_rows: List[Dict[str, Any]] = []
    if not gate_fail_symbol_df.empty:
        for rec in gate_fail_symbol_df.to_dict(orient="records"):
            reasons = str(rec["gate_fail_reason"]).split("|")
            for reason in reasons:
                if not reason:
                    continue
                gate_fail_daily_rows.append(
                    {
                        "trade_date": rec["trade_date"],
                        "branch": rec["branch"],
                        "gate_fail_reason": reason,
                        "count": 1,
                    }
                )
            if len(reasons) > 1:
                gate_fail_daily_rows.append(
                    {
                        "trade_date": rec["trade_date"],
                        "branch": rec["branch"],
                        "gate_fail_reason": "compound_fail",
                        "count": 1,
                    }
                )
        gate_fail_daily_df = (
            pd.DataFrame(gate_fail_daily_rows)
            .groupby(["trade_date", "branch", "gate_fail_reason"], dropna=False)["count"]
            .sum()
            .reset_index()
            .sort_values(["trade_date", "branch", "gate_fail_reason"])
            .reset_index(drop=True)
        )
    else:
        gate_fail_daily_df = pd.DataFrame(columns=["trade_date", "branch", "gate_fail_reason", "count"])

    compare_df.to_csv(compare_path, index=False)
    daily_df_out.to_csv(daily_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    gate_fail_daily_df.to_csv(gate_fail_daily_path, index=False)
    gate_fail_symbol_df.to_csv(gate_fail_symbol_path, index=False)
    log_event(
        logger,
        "reclaim_executor_period_complete",
        start_date=start_date.isoformat(),
        end_date=actual_end_date.isoformat(),
        trade_days=len(trade_dates),
        compare_path=str(compare_path),
        daily_report_path=str(daily_path),
        summary_path=str(summary_path),
        gate_fail_daily_path=str(gate_fail_daily_path),
        gate_fail_symbol_path=str(gate_fail_symbol_path),
        log_path=str(log_path),
    )
    return ReclaimPeriodReplayResult(
        start_date=start_date,
        end_date=actual_end_date,
        compare_path=compare_path,
        daily_report_path=daily_path,
        summary_path=summary_path,
        gate_fail_daily_path=gate_fail_daily_path,
        gate_fail_symbol_path=gate_fail_symbol_path,
        trade_days=len(trade_dates),
    )
