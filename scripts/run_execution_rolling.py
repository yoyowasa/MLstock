from __future__ import annotations

import argparse
import calendar
import importlib.util
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_backtest_dir
from mlstock.data.storage.state import write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    parts = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return [float(item) for item in parts]


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid bool value: {value}")


def _add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _load_helpers() -> object:
    helper_path = Path(__file__).with_name("run_exposure_rolling.py")
    spec = importlib.util.spec_from_file_location("execution_rolling_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_exposure_rolling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--update-months", type=int, default=3)
    parser.add_argument("--cal-window-weeks", type=int, default=104)
    parser.add_argument("--deadband-abs", type=float, default=None)
    parser.add_argument("--deadband-rel", type=float, default=None)
    parser.add_argument("--min-trade-notional", type=float, default=None)
    parser.add_argument(
        "--deadband-enabled",
        type=str,
        default=None,
        help="Override execution.deadband_v2.enabled (true/false).",
    )
    parser.add_argument(
        "--bps",
        type=str,
        default=None,
        help="Comma or space separated bps values for cost sensitivity.",
    )
    parser.add_argument("--guard-deadband-abs", type=float, default=None)
    parser.add_argument("--guard-deadband-rel", type=float, default=None)
    parser.add_argument("--guard-min-trade-notional", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    helpers = _load_helpers()
    collect_metrics_for_range = helpers._collect_metrics_for_range
    collect_returns_for_range = helpers._collect_returns_for_range
    collect_cost_adjusted_returns = helpers._collect_cost_adjusted_returns
    collect_turnover = helpers._collect_turnover
    collect_nav_trades = helpers._load_nav_trades
    calc_rolling_metrics = helpers._calc_rolling_metrics

    def _mean_column(df: pd.DataFrame, column: str) -> Optional[float]:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(series.mean()) if not series.empty else None

    def _collect_deadband_stats(nav_df: pd.DataFrame, start: date, end: date) -> dict:
        if nav_df.empty or "week_start" not in nav_df.columns:
            return {
                "sum_abs_dw_raw": None,
                "sum_abs_dw_filtered": None,
                "filtered_trade_fraction": None,
                "cash_after_exec": None,
            }
        window = nav_df[(nav_df["week_start"] >= start) & (nav_df["week_start"] <= end)]
        return {
            "sum_abs_dw_raw": _mean_column(window, "sum_abs_dw_raw"),
            "sum_abs_dw_filtered": _mean_column(window, "sum_abs_dw_filtered"),
            "filtered_trade_fraction": _mean_column(window, "filtered_trade_fraction"),
            "cash_after_exec": _mean_column(window, "cash_after_exec"),
        }

    cfg = load_config()
    if args.deadband_enabled is not None:
        deadband_enabled = _parse_bool(args.deadband_enabled)
        cfg = replace(
            cfg,
            execution=replace(
                cfg.execution,
                deadband_v2=replace(cfg.execution.deadband_v2, enabled=deadband_enabled),
            ),
        )
    start_date = args.start or date.fromisoformat(cfg.backtest.start_date)
    end_date = args.end or date.fromisoformat(cfg.backtest.end_date)

    deadband_abs = cfg.selection.deadband_abs if args.deadband_abs is None else float(args.deadband_abs)
    deadband_rel = cfg.selection.deadband_rel if args.deadband_rel is None else float(args.deadband_rel)
    min_trade_notional = (
        cfg.selection.min_trade_notional if args.min_trade_notional is None else float(args.min_trade_notional)
    )

    guard_override = any(
        value is not None
        for value in (args.guard_deadband_abs, args.guard_deadband_rel, args.guard_min_trade_notional)
    )
    guard_deadband_abs = deadband_abs if args.guard_deadband_abs is None else float(args.guard_deadband_abs)
    guard_deadband_rel = deadband_rel if args.guard_deadband_rel is None else float(args.guard_deadband_rel)
    guard_min_trade_notional = (
        min_trade_notional
        if args.guard_min_trade_notional is None
        else float(args.guard_min_trade_notional)
    )

    selection_off = replace(cfg.selection, deadband_abs=0.0, deadband_rel=0.0, min_trade_notional=0.0)
    selection_raw = replace(
        cfg.selection,
        deadband_abs=deadband_abs,
        deadband_rel=deadband_rel,
        min_trade_notional=min_trade_notional,
    )
    selection_guard = replace(
        cfg.selection,
        deadband_abs=guard_deadband_abs,
        deadband_rel=guard_deadband_rel,
        min_trade_notional=guard_min_trade_notional,
    )

    cfg_off_run = replace(cfg, selection=selection_off)
    cfg_raw_run = replace(cfg, selection=selection_raw)
    cfg_guard_run = replace(cfg, selection=selection_guard)

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
    bps_list = _parse_float_list(args.bps) or [1.0, 5.0, 10.0]
    cost_rate = float(cfg.cost_model.bps_per_side) / 10000.0
    cost_returns_by_variant = {name: {bps: [] for bps in bps_list} for name in ("off", "raw", "guard")}
    cost_returns_by_variant_full = {name: {bps: [] for bps in bps_list} for name in ("off", "raw", "guard")}
    valid_updates = 0

    current_start = start_date
    update_index = 0
    while current_start <= end_date:
        next_start = _add_months(current_start, args.update_months)
        test_end = min(end_date, next_start - timedelta(days=1))
        cal_end = current_start - timedelta(days=1)
        cal_start = cal_end - timedelta(weeks=args.cal_window_weeks)

        backtest.run(cfg_off_run, start=cal_start, end=test_end)
        nav_df, trades_df = collect_nav_trades(backtest_dir)
        metrics_off_cal = collect_metrics_for_range(nav_df, trades_df, cal_start, cal_end)
        metrics_off_test = collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
        returns_off = collect_returns_for_range(nav_df, current_start, test_end)
        cost_returns_off = collect_cost_adjusted_returns(nav_df, trades_df, current_start, test_end, bps_list)
        trades_off, turnover_off, turnover_std_off = collect_turnover(
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
        deadband_off = _collect_deadband_stats(nav_df, current_start, test_end)

        backtest.run(cfg_raw_run, start=cal_start, end=test_end)
        nav_df, trades_df = collect_nav_trades(backtest_dir)
        metrics_raw_cal = collect_metrics_for_range(nav_df, trades_df, cal_start, cal_end)
        metrics_raw_test = collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
        returns_raw = collect_returns_for_range(nav_df, current_start, test_end)
        cost_returns_raw = collect_cost_adjusted_returns(nav_df, trades_df, current_start, test_end, bps_list)
        trades_raw, turnover_raw, turnover_std_raw = collect_turnover(
            trades_df,
            current_start,
            test_end,
            cost_rate,
        )
        raw_turnover_ratio = None
        raw_turnover_ratio_std = None
        if isinstance(metrics_raw_test.get("avg_nav"), (int, float)) and metrics_raw_test["avg_nav"]:
            raw_turnover_ratio = float(turnover_raw) / float(metrics_raw_test["avg_nav"])
            raw_turnover_ratio_std = float(turnover_std_raw) / float(metrics_raw_test["avg_nav"])
        deadband_raw = _collect_deadband_stats(nav_df, current_start, test_end)

        metrics_guard = metrics_raw_test
        returns_guard = returns_raw
        cost_returns_guard = cost_returns_raw
        trades_guard = trades_raw
        turnover_guard = turnover_raw
        turnover_std_guard = turnover_std_raw
        guard_turnover_ratio = raw_turnover_ratio
        guard_turnover_ratio_std = raw_turnover_ratio_std
        guard_used_raw = True
        deadband_guard = deadband_raw
        if guard_override:
            backtest.run(cfg_guard_run, start=cal_start, end=test_end)
            nav_df, trades_df = collect_nav_trades(backtest_dir)
            metrics_guard = collect_metrics_for_range(nav_df, trades_df, current_start, test_end)
            returns_guard = collect_returns_for_range(nav_df, current_start, test_end)
            cost_returns_guard = collect_cost_adjusted_returns(
                nav_df, trades_df, current_start, test_end, bps_list
            )
            trades_guard, turnover_guard, turnover_std_guard = collect_turnover(
                trades_df,
                current_start,
                test_end,
                cost_rate,
            )
            guard_turnover_ratio = None
            guard_turnover_ratio_std = None
            if isinstance(metrics_guard.get("avg_nav"), (int, float)) and metrics_guard["avg_nav"]:
                guard_turnover_ratio = float(turnover_guard) / float(metrics_guard["avg_nav"])
                guard_turnover_ratio_std = float(turnover_std_guard) / float(metrics_guard["avg_nav"])
            guard_used_raw = False
            deadband_guard = _collect_deadband_stats(nav_df, current_start, test_end)

        cal_valid = metrics_off_cal.get("eval_weeks", 0) >= args.cal_window_weeks

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

        returns_by_variant_full["off"].extend(returns_off)
        returns_by_variant_full["raw"].extend(returns_raw)
        returns_by_variant_full["guard"].extend(returns_guard)
        for bps in bps_list:
            cost_returns_by_variant_full["off"][bps].extend(cost_returns_off[bps])
            cost_returns_by_variant_full["raw"][bps].extend(cost_returns_raw[bps])
            cost_returns_by_variant_full["guard"][bps].extend(cost_returns_guard[bps])
        trades_by_variant_full["off"] += trades_off
        trades_by_variant_full["raw"] += trades_raw
        trades_by_variant_full["guard"] += trades_guard
        turnover_by_variant_full["off"] += turnover_off
        turnover_by_variant_full["raw"] += turnover_raw
        turnover_by_variant_full["guard"] += turnover_guard
        turnover_std_by_variant_full["off"] += turnover_std_off
        turnover_std_by_variant_full["raw"] += turnover_std_raw
        turnover_std_by_variant_full["guard"] += turnover_std_guard

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
            metrics_raw_test.get("return_pct"), (int, float)
        ):
            diff_raw_return = metrics_raw_test["return_pct"] - metrics_off_test["return_pct"]
        if isinstance(metrics_off_test.get("max_drawdown_pct"), (int, float)) and isinstance(
            metrics_raw_test.get("max_drawdown_pct"), (int, float)
        ):
            diff_raw_maxdd = metrics_raw_test["max_drawdown_pct"] - metrics_off_test["max_drawdown_pct"]

        updates.append(
            {
                "update_index": update_index,
                "cal_start": cal_start.isoformat(),
                "cal_end": cal_end.isoformat(),
                "test_start": current_start.isoformat(),
                "test_end": test_end.isoformat(),
                "valid_eval": cal_valid,
                "deadband_abs": deadband_abs,
                "deadband_rel": deadband_rel,
                "min_trade_notional": min_trade_notional,
                "guard_deadband_abs": guard_deadband_abs,
                "guard_deadband_rel": guard_deadband_rel,
                "guard_min_trade_notional": guard_min_trade_notional,
                "guard_used_raw": guard_used_raw,
                "cal_avg_gross_off": metrics_off_cal.get("avg_gross_exposure"),
                "cal_avg_gross_raw": metrics_raw_cal.get("avg_gross_exposure"),
                "cal_gross_p95_off": metrics_off_cal.get("gross_exposure_p95"),
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
                "off_sum_abs_dw_raw": deadband_off["sum_abs_dw_raw"],
                "off_sum_abs_dw_filtered": deadband_off["sum_abs_dw_filtered"],
                "off_filtered_trade_fraction": deadband_off["filtered_trade_fraction"],
                "off_cash_after_exec": deadband_off["cash_after_exec"],
                "raw_return_pct": metrics_raw_test.get("return_pct"),
                "raw_max_drawdown_pct": metrics_raw_test.get("max_drawdown_pct"),
                "raw_trades": metrics_raw_test.get("trades"),
                "raw_avg_gross": metrics_raw_test.get("avg_gross_exposure"),
                "raw_p95": metrics_raw_test.get("gross_exposure_p95"),
                "raw_max": metrics_raw_test.get("gross_exposure_max"),
                "raw_turnover_usd": turnover_raw,
                "raw_turnover_ratio": raw_turnover_ratio,
                "raw_turnover_std_usd": turnover_std_raw,
                "raw_turnover_ratio_std": raw_turnover_ratio_std,
                "raw_sum_abs_dw_raw": deadband_raw["sum_abs_dw_raw"],
                "raw_sum_abs_dw_filtered": deadband_raw["sum_abs_dw_filtered"],
                "raw_filtered_trade_fraction": deadband_raw["filtered_trade_fraction"],
                "raw_cash_after_exec": deadband_raw["cash_after_exec"],
                "guard_return_pct": metrics_guard.get("return_pct"),
                "guard_max_drawdown_pct": metrics_guard.get("max_drawdown_pct"),
                "guard_trades": metrics_guard.get("trades"),
                "guard_avg_gross": metrics_guard.get("avg_gross_exposure"),
                "guard_p95": metrics_guard.get("gross_exposure_p95"),
                "guard_max": metrics_guard.get("gross_exposure_max"),
                "guard_turnover_usd": turnover_guard,
                "guard_turnover_ratio": guard_turnover_ratio,
                "guard_turnover_std_usd": turnover_std_guard,
                "guard_turnover_ratio_std": guard_turnover_ratio_std,
                "guard_sum_abs_dw_raw": deadband_guard["sum_abs_dw_raw"],
                "guard_sum_abs_dw_filtered": deadband_guard["sum_abs_dw_filtered"],
                "guard_filtered_trade_fraction": deadband_guard["filtered_trade_fraction"],
                "guard_cash_after_exec": deadband_guard["cash_after_exec"],
                "diff_guard_minus_off_return": diff_guard_return,
                "diff_guard_minus_off_maxDD": diff_guard_maxdd,
                "diff_raw_minus_off_return": diff_raw_return,
                "diff_raw_minus_off_maxDD": diff_raw_maxdd,
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
        metrics = calc_rolling_metrics(returns_series, float(cfg.backtest.initial_cash_usd))
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

    def _diff_metric(metrics_by_variant: dict, name: str, key: str) -> Optional[float]:
        off_val = metrics_by_variant["off"].get(key)
        other_val = metrics_by_variant[name].get(key)
        if isinstance(off_val, (int, float)) and isinstance(other_val, (int, float)):
            return float(other_val) - float(off_val)
        return None

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "bps_list": bps_list,
        "execution": {
            "deadband_abs": deadband_abs,
            "deadband_rel": deadband_rel,
            "min_trade_notional": min_trade_notional,
            "guard_deadband_abs": guard_deadband_abs,
            "guard_deadband_rel": guard_deadband_rel,
            "guard_min_trade_notional": guard_min_trade_notional,
            "guard_used_raw_default": guard_used_raw,
        },
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
        "updates": updates,
    }

    prefix = Path(args.output_prefix) if args.output_prefix else backtest_dir / "execution_rolling"
    summary_path = prefix.with_suffix(".json")
    updates_path = prefix.with_name(prefix.name + "_updates.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")

    write_state(summary, summary_path)

    updates_df = pd.DataFrame(updates)
    updates_df.to_csv(updates_path, index=False)

    quarterly_df = updates_df
    if "valid_eval" in quarterly_df.columns:
        quarterly_df = quarterly_df[quarterly_df["valid_eval"] == True].copy()
    quarterly_columns = [
        "update_index",
        "test_start",
        "test_end",
        "diff_raw_minus_off_return",
        "diff_raw_minus_off_maxDD",
        "off_turnover_ratio_std",
        "raw_turnover_ratio_std",
        "raw_sum_abs_dw_raw",
        "raw_sum_abs_dw_filtered",
        "raw_filtered_trade_fraction",
        "raw_cash_after_exec",
    ]
    quarterly_columns = [col for col in quarterly_columns if col in quarterly_df.columns]
    quarterly_df = quarterly_df[quarterly_columns]
    if "raw_sum_abs_dw_raw" in quarterly_df.columns and "raw_sum_abs_dw_filtered" in quarterly_df.columns:
        denom = pd.to_numeric(quarterly_df["raw_sum_abs_dw_raw"], errors="coerce")
        num = pd.to_numeric(quarterly_df["raw_sum_abs_dw_filtered"], errors="coerce")
        ratio = pd.Series([None] * len(quarterly_df), index=quarterly_df.index, dtype=object)
        mask = denom > 0
        ratio[mask] = num[mask] / denom[mask]
        quarterly_df["raw_sum_abs_dw_ratio"] = ratio
    quarterly_path = prefix.with_name(prefix.name + "_quarterly.csv")
    quarterly_df.to_csv(quarterly_path, index=False)

    summary_row = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "deadband_abs": deadband_abs,
        "deadband_rel": deadband_rel,
        "min_trade_notional": min_trade_notional,
        "guard_deadband_abs": guard_deadband_abs,
        "guard_deadband_rel": guard_deadband_rel,
        "guard_min_trade_notional": guard_min_trade_notional,
        "guard_used_raw_default": guard_used_raw,
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
        "full_diff_guard_minus_off_maxDD": _diff_metric(
            rolling_metrics_by_variant_full, "guard", "max_drawdown_pct"
        ),
        "full_diff_raw_minus_off_return": _diff_metric(rolling_metrics_by_variant_full, "raw", "return_pct"),
        "full_diff_raw_minus_off_maxDD": _diff_metric(
            rolling_metrics_by_variant_full, "raw", "max_drawdown_pct"
        ),
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
            summary_row[f"{label_prefix}_diff_raw_minus_off_maxDD_{label}"] = raw_diff.get(
                "max_drawdown_pct"
            )

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
    print(f"saved: {quarterly_path}")


if __name__ == "__main__":
    main()
