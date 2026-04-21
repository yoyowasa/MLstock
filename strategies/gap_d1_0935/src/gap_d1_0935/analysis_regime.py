from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .analysis_phase1 import _analysis_trade_dates, _build_daily_frame
from .common import ET, fetch_bars_batch, get_previous_trading_day, load_alpaca_client, load_seed_symbols, to_local_ts
from .config import StrategyConfig, load_strategy_config
from .gap_0935_watchlist_scanner import _inspect_first5_window
from .logutil import build_strategy_logger, log_event
from .paths import reports_dir


@dataclass(frozen=True)
class RegimeAnalysisResult:
    detail_path: Path
    summary_path: Path
    trigger_summary_path: Path
    branch_compare_path: Path
    start_date: date
    end_date: date
    trade_days: int


def _fetch_minute_day_map(symbols: List[str], trade_date: date) -> Dict[str, List[Dict[str, Any]]]:
    if not symbols:
        return {}
    cfg, client = load_alpaca_client()
    start_local = datetime.combine(trade_date, dtime(9, 30), tzinfo=ET)
    end_local = datetime.combine(trade_date, dtime(16, 0), tzinfo=ET)
    return fetch_bars_batch(
        client=client,
        symbols=symbols,
        start=start_local,
        end=end_local,
        timeframe="1Min",
        feed=cfg.bars.feed,
        adjustment=cfg.bars.adjustment,
        asof=cfg.bars.asof,
    )


def _valid_minute_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        try:
            ts = to_local_ts(item.get("t"))
            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour > 16:
                continue
            rows.append(
                {
                    "ts": ts,
                    "open": float(item.get("o")),
                    "high": float(item.get("h")),
                    "low": float(item.get("l")),
                    "close": float(item.get("c")),
                    "volume": float(item.get("v")),
                }
            )
        except (TypeError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["ts"])


def _first_reclaim_row(
    minute_rows: List[Dict[str, Any]],
    level: float,
    start_minute: int = 35,
) -> Optional[Dict[str, Any]]:
    for row in minute_rows:
        if row["ts"].hour < 9 or (row["ts"].hour == 9 and row["ts"].minute < start_minute):
            continue
        if row["high"] >= level:
            return row
    return None


def _intraday_vwap_series(minute_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    for row in minute_rows:
        cumulative_pv += row["close"] * row["volume"]
        cumulative_volume += row["volume"]
        vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else row["close"]
        out.append({**row, "intraday_vwap": vwap})
    return out


def _simulate_trade_from_entry(
    minute_rows: List[Dict[str, Any]],
    entry_row: Optional[Dict[str, Any]],
    entry_price: Optional[float],
    stop_price: Optional[float],
) -> Dict[str, Any]:
    if entry_row is None or entry_price is None or stop_price is None or entry_price <= stop_price:
        return {
            "entry_time": None,
            "entry_price": None,
            "high_ret_from_entry": None,
            "low_ret_from_entry": None,
            "trade_ret": None,
            "win": None,
        }
    post_entry = [row for row in minute_rows if row["ts"] >= entry_row["ts"]]
    if not post_entry:
        return {
            "entry_time": entry_row["ts"].isoformat(),
            "entry_price": entry_price,
            "high_ret_from_entry": None,
            "low_ret_from_entry": None,
            "trade_ret": None,
            "win": None,
        }
    risk = entry_price - stop_price
    target_price = entry_price + 2.0 * risk
    exit_price = post_entry[-1]["close"]
    for row in post_entry:
        if row["low"] <= stop_price:
            exit_price = stop_price
            break
        if row["high"] >= target_price:
            exit_price = target_price
            break
    return {
        "entry_time": entry_row["ts"].isoformat(),
        "entry_price": entry_price,
        "high_ret_from_entry": max((row["high"] - entry_price) / entry_price * 100.0 for row in post_entry),
        "low_ret_from_entry": min((row["low"] - entry_price) / entry_price * 100.0 for row in post_entry),
        "trade_ret": (exit_price - entry_price) / entry_price * 100.0,
        "win": exit_price > entry_price,
    }


def _daily_watchlist_rows(
    symbol: str,
    frame: pd.DataFrame,
    trade_date: date,
    previous_date: date,
    cfg: StrategyConfig,
    market_cap: Any,
    security_type: str,
) -> Optional[Dict[str, Any]]:
    eligible = frame[frame["date"] <= previous_date].sort_values("date").reset_index(drop=True)
    if len(eligible) < max(cfg.d1.lookback_days, 2):
        return None
    latest = eligible.iloc[-1]
    prior = eligible.iloc[-2]
    tail = eligible.tail(cfg.d1.lookback_days)
    avg_volume_20 = float(tail["volume"].mean())
    avg_dollar_volume_20 = float((tail["close"] * tail["volume"]).mean())
    close_d1 = float(latest["close"])
    prev_gap_pct = (float(latest["open"]) - float(prior["close"])) / float(prior["close"]) * 100.0
    rel_vol_prev = float(latest["volume"]) / avg_volume_20 if avg_volume_20 > 0 else 0.0
    day_range = float(latest["high"]) - float(latest["low"])
    close_in_range_prev = (float(latest["close"]) - float(latest["low"])) / day_range if day_range > 0 else 0.0
    oc_ret_prev = (
        (float(latest["close"]) - float(latest["open"])) / float(latest["open"]) * 100.0
        if float(latest["open"]) > 0
        else 0.0
    )
    if not (cfg.universe.min_close <= close_d1 <= cfg.universe.max_close):
        return None
    if avg_volume_20 < cfg.universe.min_avg_volume_20 or avg_dollar_volume_20 < cfg.universe.min_avg_dollar_volume_20:
        return None
    if security_type != "common_stock":
        return None
    if market_cap is None or market_cap < cfg.universe.min_market_cap or market_cap > cfg.universe.max_market_cap:
        return None
    if prev_gap_pct < cfg.d1.min_prev_gap_pct:
        return None
    if rel_vol_prev < cfg.d1.min_rel_vol_prev:
        return None
    if close_in_range_prev < cfg.d1.min_close_in_range_prev or oc_ret_prev <= cfg.d1.min_oc_ret_prev:
        return None
    return {
        "close_D-1": close_d1,
        "avg_volume_20": avg_volume_20,
    }


def analyze_watchlist_regime(
    months: int = 3,
    end_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
) -> RegimeAnalysisResult:
    strategy_cfg = config or load_strategy_config()
    _, client = load_alpaca_client()
    actual_end_date = end_date or get_previous_trading_day(client, datetime.now(ET).date() + timedelta(days=1))
    trade_dates = _analysis_trade_dates(months=months, end_date=actual_end_date)
    start_date = trade_dates[0]
    logger, log_path = build_strategy_logger("gap_d1_0935_regime_analysis", "gap_d1_0935_regime_analysis")

    from .analysis_phase1 import _build_symbol_profiles

    universe = load_seed_symbols()
    daily_df = _build_daily_frame(universe, start_date, actual_end_date)
    grouped_daily = {
        symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in daily_df.groupby("symbol")
    }
    profiles_df = _build_symbol_profiles(universe, set(universe), reports_dir() / "symbol_profiles_cache.csv")
    profiles_map = {row["symbol"]: row for row in profiles_df.to_dict(orient="records")}

    detail_rows: List[Dict[str, Any]] = []

    for trade_date in trade_dates:
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
                cfg=strategy_cfg,
                market_cap=profile.get("market_cap"),
                security_type=str(profile.get("security_type") or "common_stock"),
            )
            if context is None:
                continue
            day_frame = frame[frame["date"] == trade_date]
            if day_frame.empty:
                continue
            selected_symbols.append(symbol)
            selected_ctx[symbol] = {
                **context,
                "day_bar": day_frame.iloc[-1].to_dict(),
            }

        minute_map = _fetch_minute_day_map(selected_symbols, trade_date)
        for symbol in selected_symbols:
            ctx = selected_ctx[symbol]
            day_bar = ctx["day_bar"]
            minute_rows = _valid_minute_rows(minute_map.get(symbol, []))
            minute_rows_with_vwap = _intraday_vwap_series(minute_rows)
            inspection = _inspect_first5_window(symbol, minute_map.get(symbol, []))
            agg = inspection.get("aggregate")

            open_d = float(day_bar["open"])
            close_prev = float(ctx["close_D-1"])
            close_d = float(day_bar["close"])
            high_d = float(day_bar["high"])
            low_d = float(day_bar["low"])
            open_gap_vs_prev_close = (open_d - close_prev) / close_prev * 100.0 if close_prev > 0 else None
            day_oc_ret = (close_d - open_d) / open_d * 100.0 if open_d > 0 else None
            intraday_high_ret_from_open = (high_d - open_d) / open_d * 100.0 if open_d > 0 else None
            intraday_low_ret_from_open = (low_d - open_d) / open_d * 100.0 if open_d > 0 else None
            regime_label = (
                "regime_a_open_above_prev_close" if open_d > close_prev else "regime_b_open_at_or_below_prev_close"
            )

            first5_open = agg["first5_open"] if agg else None
            first5_high = agg["first5_high"] if agg else None
            first5_low = agg["first5_low"] if agg else None
            first5_close = agg["first5_close"] if agg else None
            first5_volume = agg["first5_volume"] if agg else None
            first5_oc_ret = (
                (first5_close - first5_open) / first5_open * 100.0 if agg and first5_open and first5_open > 0 else None
            )
            first5_range = (first5_high - first5_low) if agg else None
            first5_range_pos = (
                (first5_close - first5_low) / first5_range if agg and first5_range and first5_range > 0 else None
            )
            close_vs_vwap = (
                (first5_close / agg["vwap"] - 1.0) * 100.0 if agg and agg["vwap"] and agg["vwap"] > 0 else None
            )
            first5_pace = (
                (first5_volume * 78.0) / float(ctx["avg_volume_20"])
                if agg and float(ctx["avg_volume_20"]) > 0
                else None
            )

            continuation_pass = bool(
                agg
                and open_d > close_prev
                and open_gap_vs_prev_close is not None
                and open_gap_vs_prev_close >= strategy_cfg.day0.min_gap_today_pct
                and first5_range_pos is not None
                and first5_range_pos >= strategy_cfg.day0.min_first5_range_pos
                and first5_oc_ret is not None
                and first5_oc_ret >= strategy_cfg.day0.min_first5_oc_ret
                and first5_pace is not None
                and first5_pace >= strategy_cfg.day0.min_first5_pace
                and close_vs_vwap is not None
                and close_vs_vwap >= (strategy_cfg.day0.min_close_vs_vwap_ratio - 1.0) * 100.0
            )

            post_0935 = [
                row for row in minute_rows if (row["ts"].hour > 9) or (row["ts"].hour == 9 and row["ts"].minute >= 35)
            ]
            reclaim_prev_close = any(row["close"] >= close_prev for row in post_0935)
            reclaim_vwap = any(
                ((row["ts"].hour > 9) or (row["ts"].hour == 9 and row["ts"].minute >= 35))
                and row["close"] >= row["intraday_vwap"]
                for row in minute_rows_with_vwap
            )
            reclaim_first5_high = bool(
                first5_high is not None and any(row["close"] >= first5_high for row in post_0935)
            )
            reclaim_branch_candidate = bool(
                open_d <= close_prev and (reclaim_prev_close or reclaim_vwap or reclaim_first5_high)
            )

            reclaim_v01_gate = bool(
                regime_label == "regime_b_open_at_or_below_prev_close"
                and first5_range_pos is not None
                and first5_range_pos >= 0.50
                and first5_pace is not None
                and first5_pace >= 1.0
                and close_vs_vwap is not None
                and close_vs_vwap >= 0.0
            )
            stop_price = first5_low if first5_low is not None else None
            prev_close_entry_row = _first_reclaim_row(minute_rows, close_prev) if reclaim_v01_gate else None
            first5_high_entry_row = (
                _first_reclaim_row(minute_rows, first5_high) if (reclaim_v01_gate and first5_high is not None) else None
            )
            vwap_entry_row = None
            if reclaim_v01_gate:
                for row in minute_rows_with_vwap:
                    if ((row["ts"].hour > 9) or (row["ts"].hour == 9 and row["ts"].minute >= 35)) and row[
                        "high"
                    ] >= row["intraday_vwap"]:
                        vwap_entry_row = row
                        break
            prev_close_trade = _simulate_trade_from_entry(
                minute_rows,
                prev_close_entry_row,
                close_prev if prev_close_entry_row is not None else None,
                stop_price,
            )
            first5_high_trade = _simulate_trade_from_entry(
                minute_rows,
                first5_high_entry_row,
                first5_high if first5_high_entry_row is not None else None,
                stop_price,
            )
            vwap_trade = _simulate_trade_from_entry(
                minute_rows,
                vwap_entry_row,
                float(vwap_entry_row["intraday_vwap"]) if vwap_entry_row is not None else None,
                stop_price,
            )
            continuation_trade = _simulate_trade_from_entry(
                minute_rows,
                {"ts": post_0935[0]["ts"]} if continuation_pass and post_0935 else None,
                first5_high if continuation_pass and first5_high is not None else None,
                stop_price,
            )

            detail_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date.isoformat(),
                    "open_D": open_d,
                    "close_D-1": close_prev,
                    "open_gap_vs_prev_close": open_gap_vs_prev_close,
                    "first5_open": first5_open,
                    "first5_high": first5_high,
                    "first5_low": first5_low,
                    "first5_close": first5_close,
                    "first5_volume": first5_volume,
                    "first5_oc_ret": first5_oc_ret,
                    "first5_range_pos": first5_range_pos,
                    "close_vs_vwap": close_vs_vwap,
                    "day_oc_ret": day_oc_ret,
                    "intraday_high_ret_from_open": intraday_high_ret_from_open,
                    "intraday_low_ret_from_open": intraday_low_ret_from_open,
                    "regime_label": regime_label,
                    "continuation_pass": continuation_pass,
                    "reclaim_prev_close": reclaim_prev_close,
                    "reclaim_vwap": reclaim_vwap,
                    "reclaim_first5_high": reclaim_first5_high,
                    "reclaim_branch_candidate": reclaim_branch_candidate,
                    "reclaim_v01_gate": reclaim_v01_gate,
                    "prev_close_reclaim_trade_ret": prev_close_trade["trade_ret"],
                    "prev_close_reclaim_high_ret_from_entry_proxy": prev_close_trade["high_ret_from_entry"],
                    "prev_close_reclaim_low_ret_from_entry_proxy": prev_close_trade["low_ret_from_entry"],
                    "prev_close_reclaim_win": prev_close_trade["win"],
                    "vwap_reclaim_trade_ret": vwap_trade["trade_ret"],
                    "vwap_reclaim_high_ret_from_entry_proxy": vwap_trade["high_ret_from_entry"],
                    "vwap_reclaim_low_ret_from_entry_proxy": vwap_trade["low_ret_from_entry"],
                    "vwap_reclaim_win": vwap_trade["win"],
                    "first5_high_reclaim_trade_ret": first5_high_trade["trade_ret"],
                    "first5_high_reclaim_high_ret_from_entry_proxy": first5_high_trade["high_ret_from_entry"],
                    "first5_high_reclaim_low_ret_from_entry_proxy": first5_high_trade["low_ret_from_entry"],
                    "first5_high_reclaim_win": first5_high_trade["win"],
                    "continuation_compare_trade_ret": continuation_trade["trade_ret"],
                    "continuation_compare_high_ret_from_entry_proxy": continuation_trade["high_ret_from_entry"],
                    "continuation_compare_low_ret_from_entry_proxy": continuation_trade["low_ret_from_entry"],
                    "continuation_compare_win": continuation_trade["win"],
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    if detail_df.empty:
        raise ValueError("No regime detail rows generated")

    detail_df["open_to_close_win"] = pd.to_numeric(detail_df["day_oc_ret"], errors="coerce") > 0
    summary_df = (
        detail_df.groupby("regime_label")
        .agg(
            regime_count=("symbol", "count"),
            avg_day_oc_ret=("day_oc_ret", "mean"),
            win_rate=("open_to_close_win", "mean"),
            avg_first5_oc_ret=("first5_oc_ret", "mean"),
            p50_first5_oc_ret=("first5_oc_ret", "median"),
            avg_first5_range_pos=("first5_range_pos", "mean"),
            p50_first5_range_pos=("first5_range_pos", "median"),
            avg_close_vs_vwap=("close_vs_vwap", "mean"),
            p50_close_vs_vwap=("close_vs_vwap", "median"),
            continuation_pass_count=("continuation_pass", "sum"),
            reclaim_branch_count=("reclaim_branch_candidate", "sum"),
        )
        .reset_index()
    )

    reclaim_gate_df = detail_df[detail_df["reclaim_v01_gate"].fillna(False)].copy()
    trigger_rows = []
    for label, ret_col, win_col, high_col, low_col in [
        (
            "prev_close_reclaim",
            "prev_close_reclaim_trade_ret",
            "prev_close_reclaim_win",
            "prev_close_reclaim_high_ret_from_entry_proxy",
            "prev_close_reclaim_low_ret_from_entry_proxy",
        ),
        (
            "vwap_reclaim",
            "vwap_reclaim_trade_ret",
            "vwap_reclaim_win",
            "vwap_reclaim_high_ret_from_entry_proxy",
            "vwap_reclaim_low_ret_from_entry_proxy",
        ),
        (
            "first5_high_reclaim",
            "first5_high_reclaim_trade_ret",
            "first5_high_reclaim_win",
            "first5_high_reclaim_high_ret_from_entry_proxy",
            "first5_high_reclaim_low_ret_from_entry_proxy",
        ),
    ]:
        subset = reclaim_gate_df[reclaim_gate_df[ret_col].notna()].copy()
        trigger_rows.append(
            {
                "trigger_label": label,
                "count": int(len(subset)),
                "avg_day_oc_ret": float(subset["day_oc_ret"].mean()) if not subset.empty else None,
                "win_rate": float(subset[win_col].mean()) if not subset.empty else None,
                "avg_intraday_high_ret_from_entry_proxy": float(subset[high_col].mean()) if not subset.empty else None,
                "avg_intraday_low_ret_from_entry_proxy": float(subset[low_col].mean()) if not subset.empty else None,
                "avg_trade_ret": float(subset[ret_col].mean()) if not subset.empty else None,
            }
        )
    trigger_summary_df = pd.DataFrame(trigger_rows)

    branch_rows = []
    for label, source_df, mask_col, ret_col, win_col, high_col, low_col in [
        (
            "reclaim_v01_prev_close",
            reclaim_gate_df,
            "prev_close_reclaim_trade_ret",
            "prev_close_reclaim_trade_ret",
            "prev_close_reclaim_win",
            "prev_close_reclaim_high_ret_from_entry_proxy",
            "prev_close_reclaim_low_ret_from_entry_proxy",
        ),
        (
            "reclaim_v01_first5_high",
            reclaim_gate_df,
            "first5_high_reclaim_trade_ret",
            "first5_high_reclaim_trade_ret",
            "first5_high_reclaim_win",
            "first5_high_reclaim_high_ret_from_entry_proxy",
            "first5_high_reclaim_low_ret_from_entry_proxy",
        ),
        (
            "continuation_compare",
            detail_df[detail_df["continuation_pass"].fillna(False)].copy(),
            "continuation_compare_trade_ret",
            "continuation_compare_trade_ret",
            "continuation_compare_win",
            "continuation_compare_high_ret_from_entry_proxy",
            "continuation_compare_low_ret_from_entry_proxy",
        ),
    ]:
        subset = source_df[source_df[mask_col].notna()].copy()
        branch_rows.append(
            {
                "branch_label": label,
                "count": int(len(subset)),
                "avg_day_oc_ret": float(subset["day_oc_ret"].mean()) if not subset.empty else None,
                "win_rate": float(subset[win_col].mean()) if not subset.empty else None,
                "avg_intraday_high_ret_from_entry_proxy": float(subset[high_col].mean()) if not subset.empty else None,
                "avg_intraday_low_ret_from_entry_proxy": float(subset[low_col].mean()) if not subset.empty else None,
                "avg_trade_ret": float(subset[ret_col].mean()) if not subset.empty else None,
            }
        )
    branch_compare_df = pd.DataFrame(branch_rows)

    out_dir = reports_dir()
    stamp = f"{start_date:%Y%m%d}_{actual_end_date:%Y%m%d}"
    detail_path = out_dir / f"regime_detail_{stamp}.csv"
    summary_path = out_dir / f"regime_summary_{stamp}.csv"
    trigger_summary_path = out_dir / f"regime_trigger_summary_{stamp}.csv"
    branch_compare_path = out_dir / f"regime_branch_compare_{stamp}.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    trigger_summary_df.to_csv(trigger_summary_path, index=False)
    branch_compare_df.to_csv(branch_compare_path, index=False)

    log_event(
        logger,
        "gap_d1_0935_regime_analysis_complete",
        start_date=start_date.isoformat(),
        end_date=actual_end_date.isoformat(),
        trade_days=len(trade_dates),
        detail_path=str(detail_path),
        summary_path=str(summary_path),
        trigger_summary_path=str(trigger_summary_path),
        branch_compare_path=str(branch_compare_path),
    )

    return RegimeAnalysisResult(
        detail_path=detail_path,
        summary_path=summary_path,
        trigger_summary_path=trigger_summary_path,
        branch_compare_path=branch_compare_path,
        start_date=start_date,
        end_date=actual_end_date,
        trade_days=len(trade_dates),
    )
