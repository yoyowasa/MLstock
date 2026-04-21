from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
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


@dataclass(frozen=True)
class ReclaimBacktestResult:
    trades_path: Path
    summary_path: Path
    compare_path: Path
    decomposition_path: Path
    start_date: date
    end_date: date
    trade_days: int


def _entry_row_for_level(minute_rows: List[Dict[str, Any]], level: float) -> Optional[Dict[str, Any]]:
    for row in minute_rows:
        if row["ts"].hour < 9 or (row["ts"].hour == 9 and row["ts"].minute < 35):
            continue
        if row["high"] >= level:
            return row
    return None


def _entry_row_for_vwap(minute_rows_with_vwap: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in minute_rows_with_vwap:
        if row["ts"].hour < 9 or (row["ts"].hour == 9 and row["ts"].minute < 35):
            continue
        if row["high"] >= row["intraday_vwap"]:
            return row
    return None


def _simulate_trade(
    minute_rows: List[Dict[str, Any]],
    entry_row: Optional[Dict[str, Any]],
    entry_price: Optional[float],
    stop_price: Optional[float],
    slippage_bps_per_side: float,
    fee_bps_round_trip: float,
) -> Dict[str, Any]:
    if entry_row is None or entry_price is None or stop_price is None or entry_price <= stop_price:
        return {
            "entry_time": None,
            "entry_price": None,
            "effective_stop_price": None,
            "target_price": None,
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "trade_ret": None,
            "mfe": None,
            "mae": None,
        }
    gross_entry_price = entry_price
    effective_entry_price = gross_entry_price * (1.0 + slippage_bps_per_side / 10000.0)
    effective_stop_price = stop_price * (1.0 - slippage_bps_per_side / 10000.0)
    risk = effective_entry_price - effective_stop_price
    target_price = entry_price + 2.0 * risk
    cutoff = dtime(15, 55)
    post_entry = [row for row in minute_rows if row["ts"] >= entry_row["ts"] and row["ts"].time() <= cutoff]
    if not post_entry:
        return {
            "entry_time": entry_row["ts"].isoformat(),
            "entry_price": effective_entry_price,
            "effective_stop_price": effective_stop_price,
            "target_price": target_price,
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "trade_ret": None,
            "mfe": None,
            "mae": None,
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
    gross_trade_ret = (exit_price - effective_entry_price) / effective_entry_price * 100.0
    net_trade_ret = gross_trade_ret - fee_bps_round_trip / 100.0
    mfe = max((row["high"] - effective_entry_price) / effective_entry_price * 100.0 for row in post_entry)
    mae = min((row["low"] - effective_entry_price) / effective_entry_price * 100.0 for row in post_entry)
    return {
        "entry_time": entry_row["ts"].isoformat(),
        "entry_price": effective_entry_price,
        "effective_stop_price": effective_stop_price,
        "target_price": target_price,
        "exit_time": exit_row["ts"].isoformat(),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_trade_ret": gross_trade_ret,
        "trade_ret": net_trade_ret,
        "mfe": mfe,
        "mae": mae,
    }


def _price_bucket(price: float) -> str:
    if price < 10:
        return "03-10"
    if price < 20:
        return "10-20"
    if price < 40:
        return "20-40"
    return "40+"


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


def _expectancy(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    wins = clean[clean > 0]
    losses = clean[clean <= 0]
    win_rate = len(wins) / len(clean)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    return win_rate * avg_win + (1.0 - win_rate) * avg_loss


def _summarize_branch(df: pd.DataFrame, branch_label: str) -> Dict[str, Any]:
    exit_breakdown = df["exit_reason"].value_counts().to_dict() if not df.empty else {}
    return {
        "branch_label": branch_label,
        "trade_count": int(len(df)),
        "win_rate": float((pd.to_numeric(df["trade_ret"], errors="coerce") > 0).mean()) if not df.empty else None,
        "avg_trade_ret": float(pd.to_numeric(df["trade_ret"], errors="coerce").mean()) if not df.empty else None,
        "expectancy": _expectancy(df["trade_ret"]) if not df.empty else None,
        "avg_MFE": float(pd.to_numeric(df["MFE"], errors="coerce").mean()) if not df.empty else None,
        "avg_MAE": float(pd.to_numeric(df["MAE"], errors="coerce").mean()) if not df.empty else None,
        "exit_reason_stop": int(exit_breakdown.get("stop", 0)),
        "exit_reason_target": int(exit_breakdown.get("target", 0)),
        "exit_reason_time_exit": int(exit_breakdown.get("time_exit", 0)),
    }


def backtest_reclaim_branch(
    months: int = 12,
    end_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
    slippage_bps_per_side: Optional[float] = None,
    fee_bps_round_trip: Optional[float] = None,
) -> ReclaimBacktestResult:
    strategy_cfg = config or load_strategy_config()
    _, client = load_alpaca_client()
    actual_end_date = end_date or get_previous_trading_day(client, datetime.now(ET).date() + timedelta(days=1))
    trade_dates = _analysis_trade_dates(months=months, end_date=actual_end_date)
    start_date = trade_dates[0]
    logger, log_path = build_strategy_logger("gap_d1_0935_reclaim_backtest", "gap_d1_0935_reclaim_backtest")

    actual_fee_bps = float(strategy_cfg.cost.fee_bps_round_trip if fee_bps_round_trip is None else fee_bps_round_trip)
    universe = load_seed_symbols()
    daily_df = _build_daily_frame(universe, start_date, actual_end_date)
    grouped_daily = {
        symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in daily_df.groupby("symbol")
    }
    profiles_df = _build_symbol_profiles(universe, set(universe), reports_dir() / "symbol_profiles_cache.csv")
    profiles_map = {row["symbol"]: row for row in profiles_df.to_dict(orient="records")}

    trade_rows: List[Dict[str, Any]] = []

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
            selected_ctx[symbol] = {**context, "day_bar": day_frame.iloc[-1].to_dict(), "profile": profile}

        minute_map = _fetch_minute_day_map(selected_symbols, trade_date)
        for symbol in selected_symbols:
            ctx = selected_ctx[symbol]
            day_bar = ctx["day_bar"]
            profile = ctx["profile"]
            minute_rows = _valid_minute_rows(minute_map.get(symbol, []))
            minute_rows_with_vwap = _intraday_vwap_series(minute_rows)
            inspection = _inspect_first5_window(symbol, minute_map.get(symbol, []))
            agg = inspection.get("aggregate")
            if agg is None:
                continue

            open_d = float(day_bar["open"])
            close_prev = float(ctx["close_D-1"])
            regime_b = open_d <= close_prev
            regime_a = open_d > close_prev
            first5_high = agg["first5_high"]
            first5_low = agg["first5_low"]
            first5_close = agg["first5_close"]
            first5_open = agg["first5_open"]
            first5_range = first5_high - first5_low
            first5_range_pos = (first5_close - first5_low) / first5_range if first5_range > 0 else None
            first5_oc_ret = (first5_close - first5_open) / first5_open * 100.0 if first5_open > 0 else None
            first5_pace = (
                (agg["first5_volume"] * 78.0) / float(ctx["avg_volume_20"]) if float(ctx["avg_volume_20"]) > 0 else None
            )
            close_vs_vwap = (first5_close / agg["vwap"] - 1.0) * 100.0 if agg["vwap"] > 0 else None
            stop_price = first5_low
            avg_dollar_volume_20 = float(ctx["avg_volume_20"]) * float(ctx["close_D-1"])
            liquidity_bucket = _liquidity_bucket(avg_dollar_volume_20)
            applied_slippage_bps = (
                _slippage_bps_for_bucket(strategy_cfg, liquidity_bucket)
                if slippage_bps_per_side is None
                else float(slippage_bps_per_side)
            )
            market_cap_bucket = str(profile.get("market_cap_bucket") or "unknown")
            sector = str(profile.get("sector") or "")
            year_label = str(trade_date.year)
            quarter_label = f"{trade_date.year}-Q{((trade_date.month - 1) // 3) + 1}"

            reclaim_gate = bool(
                regime_b
                and close_vs_vwap is not None
                and close_vs_vwap >= 0.0
                and first5_range_pos is not None
                and first5_range_pos >= 0.50
                and first5_pace is not None
                and first5_pace >= 1.0
            )
            continuation_gate = bool(
                regime_a
                and (open_d - close_prev) / close_prev * 100.0 >= strategy_cfg.day0.min_gap_today_pct
                and first5_range_pos is not None
                and first5_range_pos >= strategy_cfg.day0.min_first5_range_pos
                and first5_oc_ret is not None
                and first5_oc_ret >= strategy_cfg.day0.min_first5_oc_ret
                and first5_pace is not None
                and first5_pace >= strategy_cfg.day0.min_first5_pace
                and close_vs_vwap is not None
                and close_vs_vwap >= (strategy_cfg.day0.min_close_vs_vwap_ratio - 1.0) * 100.0
            )

            branch_specs = []
            if reclaim_gate:
                branch_specs.append(
                    ("reclaim_first5_high", _entry_row_for_level(minute_rows, first5_high), first5_high)
                )
                first_vwap_row = _entry_row_for_vwap(minute_rows_with_vwap)
                branch_specs.append(
                    (
                        "reclaim_vwap",
                        first_vwap_row,
                        float(first_vwap_row["intraday_vwap"]) if first_vwap_row is not None else None,
                    )
                )
            if continuation_gate:
                branch_specs.append(
                    ("continuation_compare", _entry_row_for_level(minute_rows, first5_high), first5_high)
                )

            for branch_label, entry_row, entry_price in branch_specs:
                trade = _simulate_trade(
                    minute_rows,
                    entry_row,
                    entry_price,
                    stop_price,
                    slippage_bps_per_side=applied_slippage_bps,
                    fee_bps_round_trip=actual_fee_bps,
                )
                if trade["trade_ret"] is None:
                    continue
                trade_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "symbol": symbol,
                        "branch_label": branch_label,
                        "regime_label": (
                            "regime_b_open_at_or_below_prev_close" if regime_b else "regime_a_open_above_prev_close"
                        ),
                        "entry_time": trade["entry_time"],
                        "entry_price": trade["entry_price"],
                        "stop_price": stop_price,
                        "effective_stop_price": trade["effective_stop_price"],
                        "target_price": trade["target_price"],
                        "exit_time": trade["exit_time"],
                        "exit_price": trade["exit_price"],
                        "exit_reason": trade["exit_reason"],
                        "gross_trade_ret": trade["gross_trade_ret"],
                        "trade_ret": trade["trade_ret"],
                        "MFE": trade["mfe"],
                        "MAE": trade["mae"],
                        "sector": sector,
                        "price_bucket": _price_bucket(open_d),
                        "market_cap_bucket": market_cap_bucket,
                        "calendar_year": year_label,
                        "quarter_label": quarter_label,
                        "avg_dollar_volume_20": avg_dollar_volume_20,
                        "liquidity_bucket": liquidity_bucket,
                        "slippage_bps_per_side": applied_slippage_bps,
                        "fee_bps_round_trip": actual_fee_bps,
                        "first5_range_pos": first5_range_pos,
                        "first5_pace": first5_pace,
                        "close_vs_vwap": close_vs_vwap,
                    }
                )

    trades_df = pd.DataFrame(trade_rows).sort_values(["trade_date", "branch_label", "symbol"]).reset_index(drop=True)
    out_dir = reports_dir()
    stamp = f"{start_date:%Y%m%d}_{actual_end_date:%Y%m%d}"
    trades_path = out_dir / f"reclaim_branch_trades_{stamp}.csv"
    summary_path = out_dir / f"reclaim_branch_summary_{stamp}.csv"
    compare_path = out_dir / f"continuation_vs_reclaim_compare_{stamp}.csv"
    decomposition_path = out_dir / f"reclaim_branch_decomposition_{stamp}.csv"

    if trades_df.empty:
        raise ValueError("No reclaim/continuation trades generated")

    summary_rows = []
    for branch_label, branch_df in trades_df.groupby("branch_label"):
        summary_rows.append(_summarize_branch(branch_df, branch_label))
    summary_df = pd.DataFrame(summary_rows).sort_values("branch_label").reset_index(drop=True)
    compare_df = summary_df[
        summary_df["branch_label"].isin(["reclaim_first5_high", "reclaim_vwap", "continuation_compare"])
    ].copy()
    decomposition_rows = []
    for dimension in ["calendar_year", "quarter_label", "sector", "price_bucket", "market_cap_bucket"]:
        grouped = (
            trades_df.groupby(["branch_label", dimension])
            .agg(
                trade_count=("symbol", "count"),
                win_rate=("trade_ret", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
                avg_trade_ret=("trade_ret", "mean"),
                expectancy=("trade_ret", lambda s: _expectancy(pd.Series(s))),
                avg_MFE=("MFE", "mean"),
                avg_MAE=("MAE", "mean"),
            )
            .reset_index()
        )
        for row in grouped.to_dict(orient="records"):
            decomposition_rows.append(
                {
                    "dimension": dimension,
                    "bucket": row[dimension],
                    "branch_label": row["branch_label"],
                    "trade_count": row["trade_count"],
                    "win_rate": row["win_rate"],
                    "avg_trade_ret": row["avg_trade_ret"],
                    "expectancy": row["expectancy"],
                    "avg_MFE": row["avg_MFE"],
                    "avg_MAE": row["avg_MAE"],
                }
            )
    decomposition_df = pd.DataFrame(decomposition_rows)

    trades_df.to_csv(trades_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    decomposition_df.to_csv(decomposition_path, index=False)

    log_event(
        logger,
        "gap_d1_0935_reclaim_backtest_complete",
        start_date=start_date.isoformat(),
        end_date=actual_end_date.isoformat(),
        trade_days=len(trade_dates),
        trades_path=str(trades_path),
        summary_path=str(summary_path),
        compare_path=str(compare_path),
        decomposition_path=str(decomposition_path),
        slippage_bps_per_side=slippage_bps_per_side,
        fee_bps_round_trip=actual_fee_bps,
    )

    return ReclaimBacktestResult(
        trades_path=trades_path,
        summary_path=summary_path,
        compare_path=compare_path,
        decomposition_path=decomposition_path,
        start_date=start_date,
        end_date=actual_end_date,
        trade_days=len(trade_dates),
    )
