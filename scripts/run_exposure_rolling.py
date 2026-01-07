from __future__ import annotations

import argparse
import bisect
import calendar
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import (
    artifacts_backtest_dir,
    snapshots_features_path,
    snapshots_labels_path,
)
from mlstock.data.storage.state import write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def _add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _load_nav_trades(backtest_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nav_path = backtest_dir / "nav.parquet"
    trades_path = backtest_dir / "trades.parquet"
    nav_df = read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
    trades_df = read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    if not nav_df.empty and "week_start" in nav_df.columns:
        nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    return nav_df, trades_df


def _collect_metrics_for_range(
    nav_df: pd.DataFrame, trades_df: pd.DataFrame, start: date, end: date
) -> dict:
    if nav_df.empty or "week_start" not in nav_df.columns:
        return {
            "eval_weeks": 0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "avg_cash_ratio": None,
            "avg_gross_exposure": None,
            "gross_exposure_p95": None,
            "gross_exposure_max": None,
            "avg_n_positions": None,
            "cap_bind_weeks": 0,
            "cap_bind_rate": 0.0,
            "base_applied_weeks": 0,
            "base_applied_rate": 0.0,
        }

    nav_df = nav_df.copy()
    nav_df = nav_df[(nav_df["week_start"] >= start) & (nav_df["week_start"] <= end)]
    if nav_df.empty:
        return {
            "eval_weeks": 0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "avg_cash_ratio": None,
            "avg_gross_exposure": None,
            "gross_exposure_p95": None,
            "gross_exposure_max": None,
            "avg_n_positions": None,
            "cap_bind_weeks": 0,
            "cap_bind_rate": 0.0,
            "base_applied_weeks": 0,
            "base_applied_rate": 0.0,
        }

    eval_weeks = int(len(nav_df))
    nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce")

    first_idx = nav_df.index[0]
    start_nav = nav_series.iloc[0]
    if first_idx > 0:
        start_nav = nav_series.iloc[first_idx - 1]
    end_nav = nav_series.iloc[-1]
    return_pct = float(end_nav / start_nav - 1.0) if start_nav else 0.0

    drawdown_input = pd.concat(
        [
            pd.Series([start_nav], index=[start - timedelta(days=1)]),
            nav_series.reset_index(drop=True),
        ],
        ignore_index=False,
    )
    drawdown = drawdown_input / drawdown_input.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    cash_ratio = None
    if "cash_usd" in nav_df.columns and "nav" in nav_df.columns:
        cash = pd.to_numeric(nav_df["cash_usd"], errors="coerce")
        nav_nonzero = nav_series.mask(nav_series == 0)
        ratio = cash / nav_nonzero
        ratio = ratio.dropna()
        cash_ratio = float(ratio.mean()) if not ratio.empty else None

    gross_exposure = pd.Series(dtype=float)
    nav_nonzero = nav_series.mask(nav_series == 0)
    if "positions_value" in nav_df.columns:
        positions_value = pd.to_numeric(nav_df["positions_value"], errors="coerce")
        gross_exposure = positions_value / nav_nonzero
    elif "cash_usd" in nav_df.columns:
        cash = pd.to_numeric(nav_df["cash_usd"], errors="coerce")
        gross_exposure = 1.0 - (cash / nav_nonzero)
    gross_exposure = gross_exposure.dropna()

    gross_exposure_avg = float(gross_exposure.mean()) if not gross_exposure.empty else None
    gross_exposure_p95 = (
        float(gross_exposure.quantile(0.95)) if not gross_exposure.empty else None
    )
    gross_exposure_max = float(gross_exposure.max()) if not gross_exposure.empty else None

    trades = 0
    if not trades_df.empty and "week_start" in trades_df.columns:
        trades = int(
            trades_df[(trades_df["week_start"] >= start) & (trades_df["week_start"] <= end)].shape[0]
        )

    cap_bind_weeks = 0
    base_applied_weeks = 0
    if "exposure_guard_cap_applied" in nav_df.columns:
        cap_bind_weeks = int(nav_df["exposure_guard_cap_applied"].sum())
    if "exposure_guard_base_applied" in nav_df.columns:
        base_applied_weeks = int(nav_df["exposure_guard_base_applied"].sum())

    cap_bind_rate = float(cap_bind_weeks / eval_weeks) if eval_weeks else 0.0
    base_applied_rate = float(base_applied_weeks / eval_weeks) if eval_weeks else 0.0

    avg_n_positions = float(nav_df["n_positions"].mean()) if "n_positions" in nav_df.columns else None
    avg_nav = float(nav_series.mean()) if not nav_series.empty else None

    return {
        "eval_weeks": eval_weeks,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown,
        "trades": trades,
        "avg_cash_ratio": cash_ratio,
        "avg_gross_exposure": gross_exposure_avg,
        "gross_exposure_p95": gross_exposure_p95,
        "gross_exposure_max": gross_exposure_max,
        "avg_n_positions": avg_n_positions,
        "avg_nav": avg_nav,
        "cap_bind_weeks": cap_bind_weeks,
        "cap_bind_rate": cap_bind_rate,
        "base_applied_weeks": base_applied_weeks,
        "base_applied_rate": base_applied_rate,
    }


def _collect_returns_for_range(nav_df: pd.DataFrame, start: date, end: date) -> List[Tuple[date, float]]:
    if nav_df.empty or "week_start" not in nav_df.columns or "nav" not in nav_df.columns:
        return []
    nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    nav = pd.to_numeric(nav_df["nav"], errors="coerce")
    nav_prev = nav.shift(1)
    nav_prev = nav_prev.mask(nav_prev == 0)
    weekly_return = (nav - nav_prev) / nav_prev
    weekly_return = weekly_return.fillna(0.0)
    mask = (nav_df["week_start"] >= start) & (nav_df["week_start"] <= end)
    return list(zip(nav_df.loc[mask, "week_start"].tolist(), weekly_return.loc[mask].tolist()))


def _collect_cost_adjusted_returns(
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    start: date,
    end: date,
    bps_list: List[float],
) -> Dict[float, List[Tuple[date, float]]]:
    if nav_df.empty or "week_start" not in nav_df.columns or "nav" not in nav_df.columns:
        return {bps: [] for bps in bps_list}

    nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    nav = pd.to_numeric(nav_df["nav"], errors="coerce")
    nav_prev = nav.shift(1)
    nav_prev = nav_prev.mask(nav_prev == 0)
    weekly_return = (nav - nav_prev) / nav_prev
    weekly_return = weekly_return.fillna(0.0)

    turnover_by_week = pd.Series(0.0, index=nav_df["week_start"])
    if not trades_df.empty and "week_start" in trades_df.columns:
        if "entry_price" in trades_df.columns and "exit_price" in trades_df.columns:
            turnover = trades_df["entry_price"] + trades_df["exit_price"]
        elif "entry_price" in trades_df.columns:
            turnover = trades_df["entry_price"] * 2.0
        else:
            turnover = pd.Series(0.0, index=trades_df.index)
        turnover_by_week = turnover.groupby(trades_df["week_start"]).sum()

    mask = (nav_df["week_start"] >= start) & (nav_df["week_start"] <= end)
    weeks = nav_df.loc[mask, "week_start"]
    nav_prev_slice = nav_prev.loc[mask]
    base_return_slice = weekly_return.loc[mask]

    results = {}
    for bps in bps_list:
        adj_returns = []
        for week, base_ret, nav_prev_val in zip(weeks, base_return_slice, nav_prev_slice):
            if nav_prev_val is None or nav_prev_val == 0 or pd.isna(nav_prev_val):
                adj_returns.append((week, float(base_ret)))
                continue
            turnover = float(turnover_by_week.get(week, 0.0))
            cost = turnover * (bps / 10000.0)
            adj_returns.append((week, float(base_ret) - (cost / float(nav_prev_val))))
        results[bps] = adj_returns
    return results


def _collect_turnover(
    trades_df: pd.DataFrame,
    start: date,
    end: date,
    cost_rate: float,
) -> tuple[int, float, float]:
    if trades_df.empty:
        return 0, 0.0, 0.0
    if "week_start" in trades_df.columns:
        trades_df = trades_df[(trades_df["week_start"] >= start) & (trades_df["week_start"] <= end)]
    trades = int(len(trades_df))
    if "entry_price" in trades_df.columns and "exit_price" in trades_df.columns:
        turnover = float((trades_df["entry_price"] + trades_df["exit_price"]).sum())
    elif "entry_price" in trades_df.columns:
        turnover = float(trades_df["entry_price"].sum()) * 2.0
    else:
        turnover = 0.0

    turnover_std = 0.0
    if cost_rate > 0.0 and "buy_cost" in trades_df.columns:
        buy_cost = pd.to_numeric(trades_df["buy_cost"], errors="coerce").fillna(0.0)
        turnover_std = float(buy_cost.sum()) / float(cost_rate)
    elif "entry_price" in trades_df.columns:
        entry_price = pd.to_numeric(trades_df["entry_price"], errors="coerce").fillna(0.0)
        turnover_std = float(entry_price.sum())
    return trades, turnover, turnover_std


def _clip(value: Optional[float], clip_min: Optional[float], clip_max: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if clip_max is None:
        clip_max = 1.0
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min
    if clip_min is not None:
        value = max(value, clip_min)
    if clip_max is not None:
        value = min(value, clip_max)
    return value


def _calc_rolling_metrics(returns: pd.Series, start_nav: float) -> dict:
    if returns.empty:
        return {"return_pct": None, "max_drawdown_pct": None, "avg_nav": None}
    nav = (1.0 + returns).cumprod() * float(start_nav)
    return_pct = float(nav.iloc[-1] / float(start_nav) - 1.0)
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None
    avg_nav = float(nav.mean())
    return {
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown,
        "avg_nav": avg_nav,
    }


def _calc_vol_premium_for_weeks(
    full_df: pd.DataFrame,
    *,
    feature_name: str,
    weeks: List[date],
    high_quantile: float,
    low_quantile: float,
) -> tuple[Optional[float], int]:
    if full_df.empty or feature_name not in full_df.columns or "label_return" not in full_df.columns:
        return None, 0
    if not weeks:
        return None, 0
    window = full_df[full_df["week_start"].isin(weeks)].copy()
    if window.empty:
        return None, 0
    window = window.dropna(subset=[feature_name, "label_return"])
    if window.empty:
        return None, 0
    window["vol_rank"] = window.groupby("week_start")[feature_name].rank(pct=True, method="max")
    high = window[window["vol_rank"] >= high_quantile].groupby("week_start")["label_return"].mean()
    low = window[window["vol_rank"] <= low_quantile].groupby("week_start")["label_return"].mean()
    common_weeks = high.index.intersection(low.index)
    if common_weeks.empty:
        return None, 0
    premium = (high.loc[common_weeks] - low.loc[common_weeks]).mean()
    return float(premium), int(len(common_weeks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--volcap-threshold", type=float, default=0.95)
    parser.add_argument("--volcap-mode", type=str, default=None)
    parser.add_argument("--volcap-penalty-min", type=float, default=None)
    parser.add_argument("--volcap-regime", action="store_true", default=False)
    parser.add_argument("--volcap-regime-lookback-weeks", type=int, default=13)
    parser.add_argument("--volcap-regime-threshold", type=float, default=0.0)
    parser.add_argument("--volcap-regime-high-quantile", type=float, default=0.8)
    parser.add_argument("--volcap-regime-low-quantile", type=float, default=0.2)
    parser.add_argument("--volcap-regime-start-state", type=str, default="on")
    parser.add_argument("--update-months", type=int, default=3)
    parser.add_argument("--cal-window-weeks", type=int, default=104)
    parser.add_argument("--base-scale-clip-min", type=float, default=None)
    parser.add_argument("--base-scale-clip-max", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.risk.vol_cap.enabled:
        raise ValueError("vol_cap.enabled is false. Enable it to run exposure rolling.")
    start_date = args.start or date.fromisoformat(cfg.backtest.start_date)
    end_date = args.end or date.fromisoformat(cfg.backtest.end_date)
    threshold = args.volcap_threshold

    gate_cfg = replace(cfg.risk.regime_gate, enabled=True)
    vol_mode = args.volcap_mode if args.volcap_mode is not None else cfg.risk.vol_cap.mode
    vol_penalty_min = (
        float(args.volcap_penalty_min)
        if args.volcap_penalty_min is not None
        else float(cfg.risk.vol_cap.penalty_min)
    )
    vol_cfg_off = replace(
        cfg.risk.vol_cap,
        enabled=False,
        rank_threshold=threshold,
        mode=vol_mode,
        penalty_min=vol_penalty_min,
    )
    vol_cfg_on = replace(
        cfg.risk.vol_cap,
        enabled=True,
        rank_threshold=threshold,
        mode=vol_mode,
        penalty_min=vol_penalty_min,
    )
    guard_cfg = cfg.risk.exposure_guard

    full_df = pd.DataFrame()
    if args.volcap_regime:
        features_df = read_parquet(snapshots_features_path(cfg))
        labels_df = read_parquet(snapshots_labels_path(cfg))
        if features_df.empty or labels_df.empty:
            raise ValueError("Snapshots features/labels are empty for volcap regime gate")
        features_df["week_start"] = _to_date(features_df["week_start"])
        labels_df["week_start"] = _to_date(labels_df["week_start"])
        exclude_symbols = {symbol.upper() for symbol in cfg.snapshots.exclude_symbols}
        if exclude_symbols:
            features_df = features_df[~features_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
            labels_df = labels_df[~labels_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
        full_df = features_df.merge(labels_df, on=["week_start", "symbol"], how="inner")
        all_weeks = sorted(full_df["week_start"].unique().tolist())
    else:
        all_weeks = []

    backtest_dir = artifacts_backtest_dir(cfg)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    updates = []
    returns_by_variant = {"off": [], "raw": [], "guard": []}
    returns_by_variant_full = {"off": [], "raw": [], "guard": []}
    trades_by_variant = {"off": 0, "raw": 0, "guard": 0}
    trades_by_variant_full = {"off": 0, "raw": 0, "guard": 0}
    turnover_by_variant = {"off": 0.0, "raw": 0.0, "guard": 0.0}
    turnover_by_variant_full = {"off": 0.0, "raw": 0.0, "guard": 0.0}
    turnover_std_by_variant = {"off": 0.0, "raw": 0.0, "guard": 0.0}
    turnover_std_by_variant_full = {"off": 0.0, "raw": 0.0, "guard": 0.0}
    bps_list = [1.0, 5.0, 10.0]
    cost_rate = float(cfg.cost_model.bps_per_side) / 10000.0
    cost_returns_by_variant = {name: {bps: [] for bps in bps_list} for name in ("off", "raw", "guard")}
    cost_returns_by_variant_full = {name: {bps: [] for bps in bps_list} for name in ("off", "raw", "guard")}
    valid_updates = 0
    ema_alpha = 0.3
    prev_base_scale_used = None
    prev_cap_used = None
    gate_start_state = args.volcap_regime_start_state.strip().lower()
    if gate_start_state not in ("on", "off"):
        gate_start_state = "on"
    gate_state = gate_start_state

    current_start = start_date
    update_index = 0
    while current_start <= end_date:
        next_start = _add_months(current_start, args.update_months)
        test_end = min(end_date, next_start - timedelta(days=1))
        cal_end = current_start - timedelta(days=1)
        cal_start = cal_end - timedelta(weeks=args.cal_window_weeks)

        gate_prev = gate_state
        vol_premium = None
        vol_premium_weeks = 0
        gate_reason = None
        if args.volcap_regime:
            idx = bisect.bisect_right(all_weeks, cal_end)
            if idx >= args.volcap_regime_lookback_weeks:
                gate_weeks = all_weeks[idx - args.volcap_regime_lookback_weeks : idx]
                vol_premium, vol_premium_weeks = _calc_vol_premium_for_weeks(
                    full_df,
                    feature_name=vol_cfg_on.feature_name,
                    weeks=gate_weeks,
                    high_quantile=args.volcap_regime_high_quantile,
                    low_quantile=args.volcap_regime_low_quantile,
                )
            if isinstance(vol_premium, (int, float)):
                if vol_premium > args.volcap_regime_threshold:
                    gate_state = "off"
                    gate_reason = "premium_positive"
                elif vol_premium < -args.volcap_regime_threshold:
                    gate_state = "on"
                    gate_reason = "premium_negative"
                else:
                    gate_reason = "hold"
            else:
                gate_reason = "insufficient_history"

        gate_flip = gate_state != gate_prev

        cfg_off_run = replace(
            cfg,
            risk=replace(
                cfg.risk,
                regime_gate=gate_cfg,
                vol_cap=vol_cfg_off,
                exposure_guard=replace(guard_cfg, enabled=False),
            ),
        )
        backtest.run(cfg_off_run, start=cal_start, end=test_end)
        nav_df, trades_df = _load_nav_trades(backtest_dir)
        metrics_off_cal = _collect_metrics_for_range(nav_df, trades_df, cal_start, cal_end)
        metrics_off_test = _collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
        returns_off = _collect_returns_for_range(nav_df, current_start, test_end)
        cost_returns_off = _collect_cost_adjusted_returns(nav_df, trades_df, current_start, test_end, bps_list)
        trades_off, turnover_off, turnover_std_off = _collect_turnover(
            trades_df,
            current_start,
            test_end,
            cost_rate,
        )
        off_turnover_ratio = None
        off_turnover_ratio_std = None
        if isinstance(metrics_off_test.get("avg_nav"), (int, float)) and metrics_off_test["avg_nav"]:
            off_turnover_ratio = float(turnover_off) / float(metrics_off_test["avg_nav"])
            off_turnover_ratio_std = float(turnover_std_off) / float(metrics_off_test["avg_nav"])

        cfg_on_raw_run = replace(
            cfg,
            risk=replace(
                cfg.risk,
                regime_gate=gate_cfg,
                vol_cap=vol_cfg_on,
                exposure_guard=replace(guard_cfg, enabled=False),
            ),
        )
        backtest.run(cfg_on_raw_run, start=cal_start, end=test_end)
        nav_df, trades_df = _load_nav_trades(backtest_dir)
        metrics_on_raw_cal = _collect_metrics_for_range(nav_df, trades_df, cal_start, cal_end)
        metrics_on_raw_test = _collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
        returns_raw = _collect_returns_for_range(nav_df, current_start, test_end)
        cost_returns_raw = _collect_cost_adjusted_returns(nav_df, trades_df, current_start, test_end, bps_list)
        trades_raw, turnover_raw, turnover_std_raw = _collect_turnover(
            trades_df,
            current_start,
            test_end,
            cost_rate,
        )
        raw_turnover_ratio = None
        raw_turnover_ratio_std = None
        if isinstance(metrics_on_raw_test.get("avg_nav"), (int, float)) and metrics_on_raw_test["avg_nav"]:
            raw_turnover_ratio = float(turnover_raw) / float(metrics_on_raw_test["avg_nav"])
            raw_turnover_ratio_std = float(turnover_std_raw) / float(metrics_on_raw_test["avg_nav"])

        avg_gross_off = metrics_off_cal.get("avg_gross_exposure")
        avg_gross_on_raw = metrics_on_raw_cal.get("avg_gross_exposure")
        base_scale_raw = None
        if isinstance(avg_gross_off, (int, float)) and isinstance(avg_gross_on_raw, (int, float)):
            if avg_gross_on_raw > 0:
                base_scale_raw = float(avg_gross_off) / float(avg_gross_on_raw)
        base_scale_raw = _clip(base_scale_raw, args.base_scale_clip_min, args.base_scale_clip_max)

        base_scale_used = None
        if isinstance(base_scale_raw, (int, float)):
            if prev_base_scale_used is None:
                base_scale_used = float(base_scale_raw)
            else:
                base_scale_used = (1.0 - ema_alpha) * prev_base_scale_used + ema_alpha * base_scale_raw

        cap_raw = metrics_off_cal.get("gross_exposure_p95")
        if not isinstance(cap_raw, (int, float)):
            cap_raw = None

        cap_used = None
        if isinstance(cap_raw, (int, float)):
            if prev_cap_used is None:
                cap_used = float(cap_raw)
            elif cap_raw < prev_cap_used:
                cap_used = float(cap_raw)
            else:
                cap_used = (1.0 - ema_alpha) * prev_cap_used + ema_alpha * cap_raw

        cal_valid = (
            isinstance(avg_gross_off, (int, float))
            and avg_gross_off > 0
            and isinstance(avg_gross_on_raw, (int, float))
            and avg_gross_on_raw > 0
            and isinstance(cap_used, (int, float))
            and cap_used > 0
            and isinstance(base_scale_used, (int, float))
            and base_scale_used > 0
        )

        metrics_guard = {
            "eval_weeks": 0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "avg_cash_ratio": None,
            "avg_gross_exposure": None,
            "gross_exposure_p95": None,
            "gross_exposure_max": None,
            "avg_n_positions": None,
            "cap_bind_weeks": 0,
            "cap_bind_rate": 0.0,
            "base_applied_weeks": 0,
            "base_applied_rate": 0.0,
        }
        returns_guard: List[Tuple[date, float]] = []
        cost_returns_guard = {bps: [] for bps in bps_list}
        trades_guard = 0
        turnover_guard = 0.0
        turnover_std_guard = 0.0
        guard_turnover_ratio = None
        guard_turnover_ratio_std = None
        guard_used_off = False

        if cal_valid and (not args.volcap_regime or gate_state == "on"):
            guard_fixed = replace(
                guard_cfg,
                enabled=True,
                base_source="fixed",
                base_scale=base_scale_used,
                cap_source="fixed",
                cap_value=cap_used,
            )

            cfg_guard_test = replace(
                cfg,
                risk=replace(cfg.risk, regime_gate=gate_cfg, vol_cap=vol_cfg_on, exposure_guard=guard_fixed),
            )
            backtest.run(cfg_guard_test, start=cal_start, end=test_end)
            nav_df, trades_df = _load_nav_trades(backtest_dir)
            metrics_guard = _collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
            returns_guard = _collect_returns_for_range(nav_df, current_start, test_end)
            cost_returns_guard = _collect_cost_adjusted_returns(
                nav_df, trades_df, current_start, test_end, bps_list
            )
            trades_guard, turnover_guard, turnover_std_guard = _collect_turnover(
                trades_df,
                current_start,
                test_end,
                cost_rate,
            )
            if isinstance(metrics_guard.get("avg_nav"), (int, float)) and metrics_guard["avg_nav"]:
                guard_turnover_ratio = float(turnover_guard) / float(metrics_guard["avg_nav"])
                guard_turnover_ratio_std = float(turnover_std_guard) / float(metrics_guard["avg_nav"])
        elif cal_valid and args.volcap_regime and gate_state == "off":
            guard_used_off = True
            metrics_guard = metrics_off_test
            returns_guard = returns_off
            cost_returns_guard = cost_returns_off
            trades_guard = trades_off
            turnover_guard = turnover_off
            guard_turnover_ratio = off_turnover_ratio
            turnover_std_guard = turnover_std_off
            guard_turnover_ratio_std = off_turnover_ratio_std

        if cal_valid:
            returns_by_variant["off"].extend(returns_off)
            returns_by_variant["raw"].extend(returns_raw)
            returns_by_variant["guard"].extend(returns_guard)
            for bps in bps_list:
                cost_returns_by_variant["off"][bps].extend(cost_returns_off[bps])
                cost_returns_by_variant["raw"][bps].extend(cost_returns_raw[bps])
                cost_returns_by_variant["guard"][bps].extend(cost_returns_guard[bps])
            trades_by_variant["off"] += trades_off
            trades_by_variant["raw"] += trades_raw
            trades_by_variant["guard"] += trades_guard
            turnover_by_variant["off"] += turnover_off
            turnover_by_variant["raw"] += turnover_raw
            turnover_by_variant["guard"] += turnover_guard
            turnover_std_by_variant["off"] += turnover_std_off
            turnover_std_by_variant["raw"] += turnover_std_raw
            turnover_std_by_variant["guard"] += turnover_std_guard
            valid_updates += 1
            prev_base_scale_used = base_scale_used
            prev_cap_used = cap_used

        returns_by_variant_full["off"].extend(returns_off)
        returns_by_variant_full["raw"].extend(returns_raw)
        for bps in bps_list:
            cost_returns_by_variant_full["off"][bps].extend(cost_returns_off[bps])
            cost_returns_by_variant_full["raw"][bps].extend(cost_returns_raw[bps])
        trades_by_variant_full["off"] += trades_off
        trades_by_variant_full["raw"] += trades_raw
        turnover_by_variant_full["off"] += turnover_off
        turnover_by_variant_full["raw"] += turnover_raw
        turnover_std_by_variant_full["off"] += turnover_std_off
        turnover_std_by_variant_full["raw"] += turnover_std_raw

        if cal_valid and (not args.volcap_regime or gate_state == "on"):
            returns_by_variant_full["guard"].extend(returns_guard)
            for bps in bps_list:
                cost_returns_by_variant_full["guard"][bps].extend(cost_returns_guard[bps])
            trades_by_variant_full["guard"] += trades_guard
            turnover_by_variant_full["guard"] += turnover_guard
            turnover_std_by_variant_full["guard"] += turnover_std_guard
        elif cal_valid and args.volcap_regime and gate_state == "off":
            returns_by_variant_full["guard"].extend(returns_off)
            for bps in bps_list:
                cost_returns_by_variant_full["guard"][bps].extend(cost_returns_off[bps])
            trades_by_variant_full["guard"] += trades_off
            turnover_by_variant_full["guard"] += turnover_off
            turnover_std_by_variant_full["guard"] += turnover_std_off
        else:
            returns_by_variant_full["guard"].extend(returns_raw)
            for bps in bps_list:
                cost_returns_by_variant_full["guard"][bps].extend(cost_returns_raw[bps])
            trades_by_variant_full["guard"] += trades_raw
            turnover_by_variant_full["guard"] += turnover_raw
            turnover_std_by_variant_full["guard"] += turnover_std_raw

        diff_guard_return = None
        diff_guard_maxdd = None
        diff_raw_return = None
        diff_raw_maxdd = None
        if isinstance(metrics_off_test.get("return_pct"), (int, float)) and isinstance(
            metrics_guard.get("return_pct"), (int, float)
        ):
            diff_guard_return = metrics_guard["return_pct"] - metrics_off_test["return_pct"]
        if isinstance(metrics_off_test.get("max_drawdown_pct"), (int, float)) and isinstance(
            metrics_guard.get("max_drawdown_pct"), (int, float)
        ):
            diff_guard_maxdd = metrics_guard["max_drawdown_pct"] - metrics_off_test["max_drawdown_pct"]
        if isinstance(metrics_off_test.get("return_pct"), (int, float)) and isinstance(
            metrics_on_raw_test.get("return_pct"), (int, float)
        ):
            diff_raw_return = metrics_on_raw_test["return_pct"] - metrics_off_test["return_pct"]
        if isinstance(metrics_off_test.get("max_drawdown_pct"), (int, float)) and isinstance(
            metrics_on_raw_test.get("max_drawdown_pct"), (int, float)
        ):
            diff_raw_maxdd = metrics_on_raw_test["max_drawdown_pct"] - metrics_off_test["max_drawdown_pct"]

        updates.append(
            {
                "update_index": update_index,
                "cal_start": cal_start.isoformat(),
                "cal_end": cal_end.isoformat(),
                "test_start": current_start.isoformat(),
                "test_end": test_end.isoformat(),
                "vol_cap_gate_enabled": args.volcap_regime,
                "vol_cap_gate_state": gate_state if args.volcap_regime else None,
                "vol_cap_gate_prev_state": gate_prev if args.volcap_regime else None,
                "vol_cap_gate_flip": gate_flip if args.volcap_regime else None,
                "vol_cap_premium": vol_premium,
                "vol_cap_premium_weeks": vol_premium_weeks,
                "vol_cap_gate_reason": gate_reason,
                "vol_cap_gate_threshold": args.volcap_regime_threshold if args.volcap_regime else None,
                "vol_cap_gate_lookback_weeks": (
                    args.volcap_regime_lookback_weeks if args.volcap_regime else None
                ),
                "base_scale_raw": base_scale_raw,
                "base_scale_used": base_scale_used,
                "cap_raw": cap_raw,
                "cap_used": cap_used,
                "ema_alpha": ema_alpha,
                "base_scale": base_scale_used,
                "cap_value": cap_used,
                "cal_avg_gross_off": avg_gross_off,
                "cal_avg_gross_on_raw": avg_gross_on_raw,
                "cal_gross_p95_off": metrics_off_cal.get("gross_exposure_p95"),
                "valid_eval": cal_valid,
                "off_return_pct": metrics_off_test.get("return_pct"),
                "off_max_drawdown_pct": metrics_off_test.get("max_drawdown_pct"),
                "off_trades": metrics_off_test.get("trades"),
                "off_avg_gross": metrics_off_test.get("avg_gross_exposure"),
                "off_p95": metrics_off_test.get("gross_exposure_p95"),
                "off_max": metrics_off_test.get("gross_exposure_max"),
                "off_turnover_usd": turnover_off,
                "off_turnover_ratio": off_turnover_ratio,
                "off_turnover_std_usd": turnover_std_off,
                "off_turnover_ratio_std": off_turnover_ratio_std,
                "raw_return_pct": metrics_on_raw_test.get("return_pct"),
                "raw_max_drawdown_pct": metrics_on_raw_test.get("max_drawdown_pct"),
                "raw_trades": metrics_on_raw_test.get("trades"),
                "raw_avg_gross": metrics_on_raw_test.get("avg_gross_exposure"),
                "raw_p95": metrics_on_raw_test.get("gross_exposure_p95"),
                "raw_max": metrics_on_raw_test.get("gross_exposure_max"),
                "raw_turnover_usd": turnover_raw,
                "raw_turnover_ratio": raw_turnover_ratio,
                "raw_turnover_std_usd": turnover_std_raw,
                "raw_turnover_ratio_std": raw_turnover_ratio_std,
                "guard_return_pct": metrics_guard.get("return_pct"),
                "guard_max_drawdown_pct": metrics_guard.get("max_drawdown_pct"),
                "guard_trades": metrics_guard.get("trades"),
                "guard_avg_gross": metrics_guard.get("avg_gross_exposure"),
                "guard_p95": metrics_guard.get("gross_exposure_p95"),
                "guard_max": metrics_guard.get("gross_exposure_max"),
                "guard_turnover_usd": turnover_guard,
                "guard_turnover_ratio": guard_turnover_ratio if cal_valid else None,
                "guard_turnover_std_usd": turnover_std_guard,
                "guard_turnover_ratio_std": guard_turnover_ratio_std if cal_valid else None,
                "guard_used_off": guard_used_off if cal_valid else None,
                "diff_guard_minus_off_return": diff_guard_return,
                "diff_guard_minus_off_maxDD": diff_guard_maxdd,
                "diff_raw_minus_off_return": diff_raw_return,
                "diff_raw_minus_off_maxDD": diff_raw_maxdd,
                "cap_bind_weeks": metrics_guard.get("cap_bind_weeks"),
                "cap_bind_rate": metrics_guard.get("cap_bind_rate"),
                "base_applied_weeks": metrics_guard.get("base_applied_weeks"),
                "base_applied_rate": metrics_guard.get("base_applied_rate"),
            }
        )

        update_index += 1
        current_start = next_start

    def _build_metrics(
        returns_list: List[Tuple[date, float]],
        trades: int,
        turnover: float,
    ) -> tuple[dict, Optional[float], dict]:
        returns_df = pd.DataFrame(returns_list, columns=["week_start", "weekly_return"])
        if not returns_df.empty:
            returns_df = returns_df.sort_values("week_start").drop_duplicates("week_start")
        returns_series = (
            returns_df.set_index("week_start")["weekly_return"] if not returns_df.empty else pd.Series(dtype=float)
        )
        metrics = _calc_rolling_metrics(returns_series, float(cfg.backtest.initial_cash_usd))
        ratio = None
        if isinstance(metrics.get("avg_nav"), (int, float)) and metrics["avg_nav"]:
            ratio = float(turnover) / float(metrics["avg_nav"])
        period = {
            "start": returns_df["week_start"].min().isoformat() if not returns_df.empty else None,
            "end": returns_df["week_start"].max().isoformat() if not returns_df.empty else None,
            "weeks": int(len(returns_df)) if not returns_df.empty else 0,
        }
        return metrics, ratio, period

    rolling_metrics_by_variant = {}
    rolling_metrics_by_variant_full = {}
    turnover_ratio_by_variant = {}
    turnover_ratio_by_variant_full = {}
    turnover_ratio_std_by_variant = {}
    turnover_ratio_std_by_variant_full = {}
    valid_period = {"start": None, "end": None, "weeks": 0}
    full_period = {"start": None, "end": None, "weeks": 0}
    for name in ("off", "raw", "guard"):
        metrics, ratio, period = _build_metrics(
            returns_by_variant[name],
            trades_by_variant[name],
            turnover_by_variant[name],
        )
        rolling_metrics_by_variant[name] = metrics
        turnover_ratio_by_variant[name] = ratio
        if isinstance(metrics.get("avg_nav"), (int, float)) and metrics["avg_nav"]:
            turnover_ratio_std_by_variant[name] = float(turnover_std_by_variant[name]) / float(
                metrics["avg_nav"]
            )
        else:
            turnover_ratio_std_by_variant[name] = None
        if name == "guard":
            valid_period = period

        metrics_full, ratio_full, period_full = _build_metrics(
            returns_by_variant_full[name],
            trades_by_variant_full[name],
            turnover_by_variant_full[name],
        )
        rolling_metrics_by_variant_full[name] = metrics_full
        turnover_ratio_by_variant_full[name] = ratio_full
        if isinstance(metrics_full.get("avg_nav"), (int, float)) and metrics_full["avg_nav"]:
            turnover_ratio_std_by_variant_full[name] = float(turnover_std_by_variant_full[name]) / float(
                metrics_full["avg_nav"]
            )
        else:
            turnover_ratio_std_by_variant_full[name] = None
        if name == "guard":
            full_period = period_full

    cost_metrics = {"valid": {}, "full": {}}
    for scope, returns_map, trades_map, turnover_map in [
        ("valid", cost_returns_by_variant, trades_by_variant, turnover_by_variant),
        ("full", cost_returns_by_variant_full, trades_by_variant_full, turnover_by_variant_full),
    ]:
        for name in ("off", "raw", "guard"):
            cost_metrics[scope][name] = {}
            for bps in bps_list:
                metrics, ratio, _ = _build_metrics(
                    returns_map[name][bps],
                    trades_map[name],
                    turnover_map[name],
                )
                cost_metrics[scope][name][bps] = {
                    "return_pct": metrics.get("return_pct"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "turnover_ratio": ratio,
                }

    cost_diffs = {"valid": {}, "full": {}}
    for scope in ("valid", "full"):
        cost_diffs[scope] = {}
        for bps in bps_list:
            off_vals = cost_metrics[scope]["off"][bps]
            raw_vals = cost_metrics[scope]["raw"][bps]
            guard_vals = cost_metrics[scope]["guard"][bps]
            diff_guard = {
                "return_pct": (
                    guard_vals["return_pct"] - off_vals["return_pct"]
                    if isinstance(guard_vals["return_pct"], (int, float))
                    and isinstance(off_vals["return_pct"], (int, float))
                    else None
                ),
                "max_drawdown_pct": (
                    guard_vals["max_drawdown_pct"] - off_vals["max_drawdown_pct"]
                    if isinstance(guard_vals["max_drawdown_pct"], (int, float))
                    and isinstance(off_vals["max_drawdown_pct"], (int, float))
                    else None
                ),
            }
            diff_raw = {
                "return_pct": (
                    raw_vals["return_pct"] - off_vals["return_pct"]
                    if isinstance(raw_vals["return_pct"], (int, float))
                    and isinstance(off_vals["return_pct"], (int, float))
                    else None
                ),
                "max_drawdown_pct": (
                    raw_vals["max_drawdown_pct"] - off_vals["max_drawdown_pct"]
                    if isinstance(raw_vals["max_drawdown_pct"], (int, float))
                    and isinstance(off_vals["max_drawdown_pct"], (int, float))
                    else None
                ),
            }
            cost_diffs[scope][bps] = {
                "guard_minus_off": diff_guard,
                "raw_minus_off": diff_raw,
            }

    cap_rates = [
        row["cap_bind_rate"]
        for row in updates
        if row.get("valid_eval") and isinstance(row.get("cap_bind_rate"), (int, float))
    ]
    cap_rate_series = pd.Series(cap_rates, dtype=float)
    cap_bind_stats = {
        "p50": float(cap_rate_series.quantile(0.5)) if not cap_rate_series.empty else None,
        "p95": float(cap_rate_series.quantile(0.95)) if not cap_rate_series.empty else None,
        "max": float(cap_rate_series.max()) if not cap_rate_series.empty else None,
    }

    def _diff_metric(metrics_by_variant: dict, name: str, key: str) -> Optional[float]:
        off_val = metrics_by_variant["off"].get(key)
        other_val = metrics_by_variant[name].get(key)
        if isinstance(off_val, (int, float)) and isinstance(other_val, (int, float)):
            return float(other_val) - float(off_val)
        return None

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "vol_cap_rank_threshold": threshold,
        "vol_cap_mode": vol_mode,
        "vol_cap_penalty_min": vol_penalty_min,
        "vol_cap_gate_enabled": args.volcap_regime,
        "vol_cap_gate_lookback_weeks": args.volcap_regime_lookback_weeks if args.volcap_regime else None,
        "vol_cap_gate_threshold": args.volcap_regime_threshold if args.volcap_regime else None,
        "vol_cap_gate_high_quantile": (
            args.volcap_regime_high_quantile if args.volcap_regime else None
        ),
        "vol_cap_gate_low_quantile": args.volcap_regime_low_quantile if args.volcap_regime else None,
        "vol_cap_gate_start_state": gate_start_state if args.volcap_regime else None,
        "update_months": args.update_months,
        "cal_window_weeks": args.cal_window_weeks,
        "updates_count": len(updates),
        "valid_updates_count": valid_updates,
        "valid_period": valid_period,
        "rolling": {
            "valid_period": valid_period,
            "full_period": full_period,
            "valid": {
                "off": {
                    "return_pct": rolling_metrics_by_variant["off"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant["off"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant["off"].get("avg_nav"),
                    "trades": trades_by_variant["off"],
                    "turnover_usd": turnover_by_variant["off"],
                    "turnover_ratio": turnover_ratio_by_variant["off"],
                    "turnover_std_usd": turnover_std_by_variant["off"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant["off"],
                },
                "on_raw": {
                    "return_pct": rolling_metrics_by_variant["raw"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant["raw"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant["raw"].get("avg_nav"),
                    "trades": trades_by_variant["raw"],
                    "turnover_usd": turnover_by_variant["raw"],
                    "turnover_ratio": turnover_ratio_by_variant["raw"],
                    "turnover_std_usd": turnover_std_by_variant["raw"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant["raw"],
                },
                "on_guard": {
                    "return_pct": rolling_metrics_by_variant["guard"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant["guard"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant["guard"].get("avg_nav"),
                    "trades": trades_by_variant["guard"],
                    "turnover_usd": turnover_by_variant["guard"],
                    "turnover_ratio": turnover_ratio_by_variant["guard"],
                    "turnover_std_usd": turnover_std_by_variant["guard"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant["guard"],
                },
                "diff_guard_minus_off": {
                    "return_pct": _diff_metric(rolling_metrics_by_variant, "guard", "return_pct"),
                    "max_drawdown_pct": _diff_metric(rolling_metrics_by_variant, "guard", "max_drawdown_pct"),
                },
                "diff_raw_minus_off": {
                    "return_pct": _diff_metric(rolling_metrics_by_variant, "raw", "return_pct"),
                    "max_drawdown_pct": _diff_metric(rolling_metrics_by_variant, "raw", "max_drawdown_pct"),
                },
                "cost_sensitivity_bps": cost_metrics["valid"],
                "cost_diff_vs_off_bps": cost_diffs["valid"],
            },
            "full": {
                "off": {
                    "return_pct": rolling_metrics_by_variant_full["off"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant_full["off"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant_full["off"].get("avg_nav"),
                    "trades": trades_by_variant_full["off"],
                    "turnover_usd": turnover_by_variant_full["off"],
                    "turnover_ratio": turnover_ratio_by_variant_full["off"],
                    "turnover_std_usd": turnover_std_by_variant_full["off"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant_full["off"],
                },
                "on_raw": {
                    "return_pct": rolling_metrics_by_variant_full["raw"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant_full["raw"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant_full["raw"].get("avg_nav"),
                    "trades": trades_by_variant_full["raw"],
                    "turnover_usd": turnover_by_variant_full["raw"],
                    "turnover_ratio": turnover_ratio_by_variant_full["raw"],
                    "turnover_std_usd": turnover_std_by_variant_full["raw"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant_full["raw"],
                },
                "on_guard": {
                    "return_pct": rolling_metrics_by_variant_full["guard"].get("return_pct"),
                    "max_drawdown_pct": rolling_metrics_by_variant_full["guard"].get("max_drawdown_pct"),
                    "avg_nav": rolling_metrics_by_variant_full["guard"].get("avg_nav"),
                    "trades": trades_by_variant_full["guard"],
                    "turnover_usd": turnover_by_variant_full["guard"],
                    "turnover_ratio": turnover_ratio_by_variant_full["guard"],
                    "turnover_std_usd": turnover_std_by_variant_full["guard"],
                    "turnover_ratio_std": turnover_ratio_std_by_variant_full["guard"],
                },
                "diff_guard_minus_off": {
                    "return_pct": _diff_metric(rolling_metrics_by_variant_full, "guard", "return_pct"),
                    "max_drawdown_pct": _diff_metric(rolling_metrics_by_variant_full, "guard", "max_drawdown_pct"),
                },
                "diff_raw_minus_off": {
                    "return_pct": _diff_metric(rolling_metrics_by_variant_full, "raw", "return_pct"),
                    "max_drawdown_pct": _diff_metric(rolling_metrics_by_variant_full, "raw", "max_drawdown_pct"),
                },
                "cost_sensitivity_bps": cost_metrics["full"],
                "cost_diff_vs_off_bps": cost_diffs["full"],
            },
        },
        "cap_bind_rate_distribution": cap_bind_stats,
        "updates": updates,
    }

    prefix = Path(args.output_prefix) if args.output_prefix else backtest_dir / "exposure_rolling"
    summary_path = prefix.with_suffix(".json")
    updates_path = prefix.with_name(prefix.name + "_updates.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")

    write_state(summary, summary_path)

    updates_df = pd.DataFrame(updates)
    updates_df.to_csv(updates_path, index=False)

    summary_row = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "vol_cap_rank_threshold": threshold,
                "vol_cap_mode": vol_mode,
                "vol_cap_penalty_min": vol_penalty_min,
                "vol_cap_gate_enabled": args.volcap_regime,
                "vol_cap_gate_lookback_weeks": (
                    args.volcap_regime_lookback_weeks if args.volcap_regime else None
                ),
                "vol_cap_gate_threshold": args.volcap_regime_threshold if args.volcap_regime else None,
                "vol_cap_gate_high_quantile": (
                    args.volcap_regime_high_quantile if args.volcap_regime else None
                ),
                "vol_cap_gate_low_quantile": (
                    args.volcap_regime_low_quantile if args.volcap_regime else None
                ),
                "vol_cap_gate_start_state": gate_start_state if args.volcap_regime else None,
                "update_months": args.update_months,
                "cal_window_weeks": args.cal_window_weeks,
                "updates_count": len(updates),
                "valid_updates_count": valid_updates,
                "valid_start_date": valid_period["start"],
                "valid_end_date": valid_period["end"],
                "valid_weeks": valid_period["weeks"],
                "full_start_date": full_period["start"],
                "full_end_date": full_period["end"],
                "full_weeks": full_period["weeks"],
                "valid_off_return_pct": rolling_metrics_by_variant["off"].get("return_pct"),
                "valid_off_max_drawdown_pct": rolling_metrics_by_variant["off"].get("max_drawdown_pct"),
                "valid_off_trades": trades_by_variant["off"],
                "valid_off_turnover_usd": turnover_by_variant["off"],
                "valid_off_turnover_ratio": turnover_ratio_by_variant["off"],
                "valid_off_turnover_std_usd": turnover_std_by_variant["off"],
                "valid_off_turnover_ratio_std": turnover_ratio_std_by_variant["off"],
                "valid_raw_return_pct": rolling_metrics_by_variant["raw"].get("return_pct"),
                "valid_raw_max_drawdown_pct": rolling_metrics_by_variant["raw"].get("max_drawdown_pct"),
                "valid_raw_trades": trades_by_variant["raw"],
                "valid_raw_turnover_usd": turnover_by_variant["raw"],
                "valid_raw_turnover_ratio": turnover_ratio_by_variant["raw"],
                "valid_raw_turnover_std_usd": turnover_std_by_variant["raw"],
                "valid_raw_turnover_ratio_std": turnover_ratio_std_by_variant["raw"],
                "valid_guard_return_pct": rolling_metrics_by_variant["guard"].get("return_pct"),
                "valid_guard_max_drawdown_pct": rolling_metrics_by_variant["guard"].get("max_drawdown_pct"),
                "valid_guard_trades": trades_by_variant["guard"],
                "valid_guard_turnover_usd": turnover_by_variant["guard"],
                "valid_guard_turnover_ratio": turnover_ratio_by_variant["guard"],
                "valid_guard_turnover_std_usd": turnover_std_by_variant["guard"],
                "valid_guard_turnover_ratio_std": turnover_ratio_std_by_variant["guard"],
                "valid_diff_guard_minus_off_return": _diff_metric(rolling_metrics_by_variant, "guard", "return_pct"),
                "valid_diff_guard_minus_off_maxDD": _diff_metric(rolling_metrics_by_variant, "guard", "max_drawdown_pct"),
                "valid_diff_raw_minus_off_return": _diff_metric(rolling_metrics_by_variant, "raw", "return_pct"),
                "valid_diff_raw_minus_off_maxDD": _diff_metric(rolling_metrics_by_variant, "raw", "max_drawdown_pct"),
                "full_off_return_pct": rolling_metrics_by_variant_full["off"].get("return_pct"),
                "full_off_max_drawdown_pct": rolling_metrics_by_variant_full["off"].get("max_drawdown_pct"),
                "full_off_trades": trades_by_variant_full["off"],
                "full_off_turnover_usd": turnover_by_variant_full["off"],
                "full_off_turnover_ratio": turnover_ratio_by_variant_full["off"],
                "full_off_turnover_std_usd": turnover_std_by_variant_full["off"],
                "full_off_turnover_ratio_std": turnover_ratio_std_by_variant_full["off"],
                "full_raw_return_pct": rolling_metrics_by_variant_full["raw"].get("return_pct"),
                "full_raw_max_drawdown_pct": rolling_metrics_by_variant_full["raw"].get("max_drawdown_pct"),
                "full_raw_trades": trades_by_variant_full["raw"],
                "full_raw_turnover_usd": turnover_by_variant_full["raw"],
                "full_raw_turnover_ratio": turnover_ratio_by_variant_full["raw"],
                "full_raw_turnover_std_usd": turnover_std_by_variant_full["raw"],
                "full_raw_turnover_ratio_std": turnover_ratio_std_by_variant_full["raw"],
                "full_guard_return_pct": rolling_metrics_by_variant_full["guard"].get("return_pct"),
                "full_guard_max_drawdown_pct": rolling_metrics_by_variant_full["guard"].get("max_drawdown_pct"),
                "full_guard_trades": trades_by_variant_full["guard"],
                "full_guard_turnover_usd": turnover_by_variant_full["guard"],
                "full_guard_turnover_ratio": turnover_ratio_by_variant_full["guard"],
                "full_guard_turnover_std_usd": turnover_std_by_variant_full["guard"],
                "full_guard_turnover_ratio_std": turnover_ratio_std_by_variant_full["guard"],
                "full_diff_guard_minus_off_return": _diff_metric(rolling_metrics_by_variant_full, "guard", "return_pct"),
                "full_diff_guard_minus_off_maxDD": _diff_metric(rolling_metrics_by_variant_full, "guard", "max_drawdown_pct"),
                "full_diff_raw_minus_off_return": _diff_metric(rolling_metrics_by_variant_full, "raw", "return_pct"),
                "full_diff_raw_minus_off_maxDD": _diff_metric(rolling_metrics_by_variant_full, "raw", "max_drawdown_pct"),
                "cap_bind_rate_p50": cap_bind_stats["p50"],
                "cap_bind_rate_p95": cap_bind_stats["p95"],
                "cap_bind_rate_max": cap_bind_stats["max"],
            }

    for scope, label_prefix in (("valid", "valid"), ("full", "full")):
        for bps in bps_list:
            label = f"{bps:.0f}bp"
            guard_diff = cost_diffs[scope][bps]["guard_minus_off"]
            raw_diff = cost_diffs[scope][bps]["raw_minus_off"]
            summary_row[f"{label_prefix}_diff_guard_minus_off_return_{label}"] = guard_diff.get("return_pct")
            summary_row[f"{label_prefix}_diff_guard_minus_off_maxDD_{label}"] = guard_diff.get(
                "max_drawdown_pct"
            )
            summary_row[f"{label_prefix}_diff_raw_minus_off_return_{label}"] = raw_diff.get("return_pct")
            summary_row[f"{label_prefix}_diff_raw_minus_off_maxDD_{label}"] = raw_diff.get("max_drawdown_pct")

    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_csv_path, index=False)

    cost_csv_path = prefix.with_name(prefix.name + "_cost.csv")
    cost_rows = []
    for scope in ("valid", "full"):
        for bps in bps_list:
            off_vals = cost_metrics[scope]["off"][bps]
            raw_vals = cost_metrics[scope]["raw"][bps]
            guard_vals = cost_metrics[scope]["guard"][bps]
            diff_guard = cost_diffs[scope][bps]["guard_minus_off"]
            diff_raw = cost_diffs[scope][bps]["raw_minus_off"]
            cost_rows.append(
                {
                    "scope": scope,
                    "bps": bps,
                    "off_return_pct": off_vals.get("return_pct"),
                    "off_max_drawdown_pct": off_vals.get("max_drawdown_pct"),
                    "raw_return_pct": raw_vals.get("return_pct"),
                    "raw_max_drawdown_pct": raw_vals.get("max_drawdown_pct"),
                    "guard_return_pct": guard_vals.get("return_pct"),
                    "guard_max_drawdown_pct": guard_vals.get("max_drawdown_pct"),
                    "diff_guard_minus_off_return": diff_guard.get("return_pct"),
                    "diff_guard_minus_off_maxDD": diff_guard.get("max_drawdown_pct"),
                    "diff_raw_minus_off_return": diff_raw.get("return_pct"),
                    "diff_raw_minus_off_maxDD": diff_raw.get("max_drawdown_pct"),
                }
            )
    pd.DataFrame(cost_rows).to_csv(cost_csv_path, index=False)

    print(f"saved: {summary_path}")
    print(f"saved: {updates_path}")
    print(f"saved: {summary_csv_path}")
    print(f"saved: {cost_csv_path}")


if __name__ == "__main__":
    main()
