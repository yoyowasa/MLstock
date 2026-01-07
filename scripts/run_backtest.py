from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_backtest_dir
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.state import read_state, write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _extract_nav_series(nav_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if nav_df.empty or "week_start" not in nav_df.columns or "nav" not in nav_df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    nav = pd.to_numeric(nav_df["nav"], errors="coerce")
    nav_prev = nav.shift(1)
    nav_prev = nav_prev.mask(nav_prev == 0)
    weekly_return = (nav - nav_prev) / nav_prev
    nav_nonzero = nav.mask(nav == 0)

    if "positions_value" in nav_df.columns:
        positions_value = pd.to_numeric(nav_df["positions_value"], errors="coerce")
        gross_exposure = positions_value / nav_nonzero
    elif "cash_usd" in nav_df.columns:
        cash_series = pd.to_numeric(nav_df["cash_usd"], errors="coerce")
        gross_exposure = 1.0 - (cash_series / nav_nonzero)
    else:
        gross_exposure = pd.Series(dtype=float)

    weekly_return.index = nav_df["week_start"]
    gross_exposure.index = nav_df["week_start"]
    return weekly_return, gross_exposure


def _neutralize_series(
    weekly_return: pd.Series,
    scale: pd.Series | float | None,
    start_nav: Optional[float],
) -> dict:
    if start_nav is None or weekly_return.empty or scale is None:
        return {"return_pct": None, "max_drawdown_pct": None}

    weekly_return = weekly_return.fillna(0.0)
    if isinstance(scale, pd.Series):
        scale = scale.reindex(weekly_return.index).fillna(0.0)
        adjusted = weekly_return * scale
    else:
        adjusted = weekly_return * float(scale)

    nav_adj = (1.0 + adjusted).cumprod() * float(start_nav)
    if nav_adj.empty:
        return {"return_pct": None, "max_drawdown_pct": None}
    return_pct = float(nav_adj.iloc[-1] / float(start_nav) - 1.0)
    drawdown = nav_adj / nav_adj.cummax() - 1.0
    max_drawdown_pct = float(drawdown.min()) if not drawdown.empty else None
    return {"return_pct": return_pct, "max_drawdown_pct": max_drawdown_pct}


def _collect_metrics(cfg, start: Optional[date], end: Optional[date]) -> Tuple[dict, pd.DataFrame]:
    summary = backtest.run(cfg, start=start, end=end)

    backtest_dir = artifacts_backtest_dir(cfg)
    nav_path = backtest_dir / "nav.parquet"
    nav_df = read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
    if not nav_df.empty and "week_start" in nav_df.columns:
        nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce") if not nav_df.empty else pd.Series(dtype=float)
    n_positions = nav_df["n_positions"] if "n_positions" in nav_df.columns else pd.Series(dtype=float)
    gross_exposure_series = pd.Series(dtype=float)
    if not nav_df.empty:
        nav_nonzero = nav_series.mask(nav_series == 0)
        if "positions_value" in nav_df.columns:
            positions_value = pd.to_numeric(nav_df["positions_value"], errors="coerce")
            gross_exposure_series = positions_value / nav_nonzero
        elif "cash_usd" in nav_df.columns:
            cash_series = pd.to_numeric(nav_df["cash_usd"], errors="coerce")
            gross_exposure_series = 1.0 - (cash_series / nav_nonzero)

    dd_path = backtest_dir / "audit_max_drawdown.json"
    dd_state = read_state(dd_path) if dd_path.exists() else {}

    gate_closed_weeks = summary.get("regime_gate_closed_weeks") or 0
    total_weeks = int(len(nav_df)) if not nav_df.empty else 0
    gate_closed_rate = float(gate_closed_weeks / total_weeks) if total_weeks else 0.0

    max_positions = max(int(cfg.selection.max_positions), 1)
    underinvested_threshold = min(10, max_positions)
    trade_weeks = int((n_positions > 0).sum()) if not n_positions.empty else 0
    first_trade_week = None
    if trade_weeks > 0 and "week_start" in nav_df.columns:
        first_trade = nav_df.loc[n_positions > 0, "week_start"].iloc[0]
        first_trade_week = first_trade.isoformat() if hasattr(first_trade, "isoformat") else str(first_trade)

    avg_cash_ratio = summary.get("avg_cash_ratio")
    avg_invested_ratio = None
    return_per_invested = None
    if isinstance(avg_cash_ratio, (int, float)):
        avg_invested_ratio = 1.0 - float(avg_cash_ratio)
        if avg_invested_ratio != 0:
            return_pct = summary.get("return_pct")
            if isinstance(return_pct, (int, float)):
                return_per_invested = float(return_pct) / avg_invested_ratio
    avg_gross_exposure = avg_invested_ratio
    avg_position_weight = None
    if isinstance(avg_gross_exposure, (int, float)) and not n_positions.empty:
        avg_positions = float(n_positions.mean())
        if avg_positions > 0:
            avg_position_weight = avg_gross_exposure / avg_positions
    gross_exposure_series = gross_exposure_series.dropna()
    gross_exposure_max = float(gross_exposure_series.max()) if not gross_exposure_series.empty else None
    gross_exposure_p95 = (
        float(gross_exposure_series.quantile(0.95)) if not gross_exposure_series.empty else None
    )

    return {
        "eval_weeks": total_weeks,
        "trade_weeks": trade_weeks,
        "first_trade_week": first_trade_week,
        "valid_eval": trade_weeks > 0,
        "start_nav": summary.get("start_nav"),
        "return_pct": summary.get("return_pct"),
        "end_nav": summary.get("end_nav"),
        "max_drawdown_pct": dd_state.get("max_drawdown_pct"),
        "min_nav": float(nav_series.min()) if not nav_series.empty else None,
        "max_nav": float(nav_series.max()) if not nav_series.empty else None,
        "trades": summary.get("trades"),
        "avg_cash_ratio": avg_cash_ratio,
        "avg_invested_ratio": avg_invested_ratio,
        "return_per_invested": return_per_invested,
        "avg_gross_exposure": avg_gross_exposure,
        "avg_position_weight": avg_position_weight,
        "gross_exposure_max": gross_exposure_max,
        "gross_exposure_p95": gross_exposure_p95,
        "avg_n_positions": float(n_positions.mean()) if not n_positions.empty else None,
        "underinvested_threshold": underinvested_threshold,
        "weeks_underinvested": int((n_positions < underinvested_threshold).sum()) if not n_positions.empty else None,
        "reserve_violation_count": summary.get("reserve_violation_count"),
        "gate_closed_weeks": int(gate_closed_weeks),
        "gate_closed_rate": gate_closed_rate,
        "vol_cap_excluded_rate": summary.get("vol_cap_excluded_rate"),
        "vol_cap_candidates": summary.get("vol_cap_candidates"),
        "vol_cap_excluded": summary.get("vol_cap_excluded"),
        "vol_cap_missing": summary.get("vol_cap_missing"),
        "vol_cap_mode": summary.get("vol_cap_mode"),
        "vol_cap_penalty_min": summary.get("vol_cap_penalty_min"),
        "vol_cap_penalized": summary.get("vol_cap_penalized"),
        "vol_cap_penalized_rate": summary.get("vol_cap_penalized_rate"),
        "vol_cap_penalized_weeks": summary.get("vol_cap_penalized_weeks"),
        "exposure_guard_active": summary.get("exposure_guard_active"),
        "exposure_guard_cap": summary.get("exposure_guard_cap"),
        "exposure_guard_cap_source": summary.get("exposure_guard_cap_source"),
        "exposure_guard_cap_value": summary.get("exposure_guard_cap_value"),
        "exposure_guard_cap_buffer": summary.get("exposure_guard_cap_buffer"),
        "exposure_guard_base_source": summary.get("exposure_guard_base_source"),
        "exposure_guard_base_scale": summary.get("exposure_guard_base_scale"),
        "exposure_guard_applied_weeks": summary.get("exposure_guard_applied_weeks"),
        "exposure_guard_base_applied_weeks": summary.get("exposure_guard_base_applied_weeks"),
        "exposure_guard_cap_applied_weeks": summary.get("exposure_guard_cap_applied_weeks"),
        "exposure_guard_base_applied_rate": summary.get("exposure_guard_base_applied_rate"),
        "exposure_guard_cap_applied_rate": summary.get("exposure_guard_cap_applied_rate"),
        "exposure_guard_scale_avg": summary.get("exposure_guard_scale_avg"),
        "exposure_guard_scale_p95": summary.get("exposure_guard_scale_p95"),
        "exposure_guard_scale_min": summary.get("exposure_guard_scale_min"),
        "exposure_guard_scale_max": summary.get("exposure_guard_scale_max"),
    }, nav_df


def _diff_metrics(base: dict, other: dict, keys: List[str]) -> dict:
    diff = {}
    for key in keys:
        base_val = base.get(key)
        other_val = other.get(key)
        if isinstance(base_val, (int, float)) and isinstance(other_val, (int, float)):
            diff[key] = other_val - base_val
        else:
            diff[key] = None
    return diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--compare-regime", action="store_true", default=False)
    parser.add_argument("--compare-volcap", action="store_true", default=False)
    parser.add_argument("--volcap-threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.compare_regime and args.compare_volcap:
        raise ValueError("Choose only one compare mode: --compare-regime or --compare-volcap")
    if args.compare_volcap and not cfg.risk.vol_cap.enabled:
        raise ValueError("vol_cap.enabled is false. Enable it to run --compare-volcap.")

    if not args.compare_regime and not args.compare_volcap:
        backtest.run(cfg, start=args.start, end=args.end)
        return

    backtest_dir = artifacts_backtest_dir(cfg)
    start_date = args.start or date.fromisoformat(cfg.backtest.start_date)
    end_date = args.end or date.fromisoformat(cfg.backtest.end_date)

    if args.compare_regime:
        gate_cfg = cfg.risk.regime_gate
        cfg_off = replace(cfg, risk=replace(cfg.risk, regime_gate=replace(gate_cfg, enabled=False)))
        cfg_on = replace(cfg, risk=replace(cfg.risk, regime_gate=replace(gate_cfg, enabled=True)))

        metrics_off, _ = _collect_metrics(cfg_off, start_date, end_date)
        metrics_on, _ = _collect_metrics(cfg_on, start_date, end_date)

        compare = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "gate_rule": gate_cfg.rule,
            "gate_action": gate_cfg.action,
            "runs": {
                "gate_off": metrics_off,
                "gate_on": metrics_on,
            },
            "diff_on_minus_off": _diff_metrics(
                metrics_off,
                metrics_on,
                [
                    "return_pct",
                    "end_nav",
                    "max_drawdown_pct",
                    "min_nav",
                    "max_nav",
                    "trades",
                    "avg_cash_ratio",
                    "avg_invested_ratio",
                    "return_per_invested",
                    "avg_gross_exposure",
                    "avg_position_weight",
                    "gross_exposure_max",
                    "gross_exposure_p95",
                    "avg_n_positions",
                    "weeks_underinvested",
                    "reserve_violation_count",
                    "gate_closed_weeks",
                    "gate_closed_rate",
                    "exposure_guard_applied_weeks",
                    "exposure_guard_base_scale",
                    "exposure_guard_scale_avg",
                    "exposure_guard_scale_p95",
                    "exposure_guard_scale_min",
                    "exposure_guard_scale_max",
                ],
            ),
        }
        write_state(compare, backtest_dir / "regime_compare.json")
        return

    gate_cfg = cfg.risk.regime_gate
    vol_cap_cfg = cfg.risk.vol_cap
    threshold = args.volcap_threshold if args.volcap_threshold is not None else vol_cap_cfg.rank_threshold
    cfg_base = replace(
        cfg,
        risk=replace(
            cfg.risk,
            regime_gate=replace(gate_cfg, enabled=True),
            vol_cap=replace(vol_cap_cfg, enabled=False, rank_threshold=threshold),
        ),
    )
    cfg_variant = replace(
        cfg,
        risk=replace(
            cfg.risk,
            regime_gate=replace(gate_cfg, enabled=True),
            vol_cap=replace(vol_cap_cfg, enabled=True, rank_threshold=threshold),
        ),
    )

    metrics_base, nav_base = _collect_metrics(cfg_base, start_date, end_date)

    guard_cfg = cfg.risk.exposure_guard
    guard_cfg_variant = guard_cfg
    if guard_cfg.enabled and guard_cfg.base_scale is None and guard_cfg.base_source == "off_avg_on_avg":
        cfg_on_raw = replace(cfg_variant, risk=replace(cfg_variant.risk, exposure_guard=replace(guard_cfg, enabled=False)))
        metrics_on_raw, _ = _collect_metrics(cfg_on_raw, start_date, end_date)
        avg_gross_off = metrics_base.get("avg_gross_exposure")
        avg_gross_on = metrics_on_raw.get("avg_gross_exposure")
        if isinstance(avg_gross_off, (int, float)) and isinstance(avg_gross_on, (int, float)) and avg_gross_on > 0:
            base_scale = avg_gross_off / avg_gross_on
            if base_scale > 1.0:
                base_scale = 1.0
            guard_cfg_variant = replace(guard_cfg_variant, base_scale=float(base_scale))

    if (
        guard_cfg.enabled
        and guard_cfg.cap_value is None
        and guard_cfg.cap_source in ("off_p95", "off_avg")
    ):
        base_value = metrics_base.get("gross_exposure_p95") if guard_cfg.cap_source == "off_p95" else None
        if guard_cfg.cap_source == "off_avg":
            base_value = metrics_base.get("avg_gross_exposure")
        if isinstance(base_value, (int, float)):
            guard_cfg_variant = replace(guard_cfg_variant, cap_value=float(base_value))

    if guard_cfg_variant is not guard_cfg:
        cfg_variant = replace(cfg_variant, risk=replace(cfg_variant.risk, exposure_guard=guard_cfg_variant))

    metrics_var, nav_var = _collect_metrics(cfg_variant, start_date, end_date)

    neutralized = None
    returns_on, exposure_on = _extract_nav_series(nav_var)
    returns_off, exposure_off = _extract_nav_series(nav_base)
    avg_exposure_on = float(exposure_on.dropna().mean()) if not exposure_on.dropna().empty else None
    avg_exposure_off = float(exposure_off.dropna().mean()) if not exposure_off.dropna().empty else None
    start_nav = metrics_var.get("start_nav")

    neutral_avg = None
    if avg_exposure_on is not None and avg_exposure_on > 0 and avg_exposure_off is not None:
        scale_avg = avg_exposure_off / avg_exposure_on
        neutral_avg = {
            "scale": scale_avg,
            "target_avg_gross_exposure": avg_exposure_off,
            "source_avg_gross_exposure": avg_exposure_on,
            "result": _neutralize_series(returns_on, scale_avg, start_nav),
        }

    neutral_weekly = None
    if not returns_on.empty and not exposure_on.empty and not exposure_off.empty:
        aligned = returns_on.index.intersection(exposure_off.index)
        if not aligned.empty:
            on_returns = returns_on.loc[aligned]
            on_exposure = exposure_on.reindex(aligned)
            off_exposure = exposure_off.reindex(aligned)
            scale_weekly = off_exposure / on_exposure
            scale_weekly = scale_weekly.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
            neutral_weekly = {
                "aligned_weeks": int(len(aligned)),
                "result": _neutralize_series(on_returns, scale_weekly, start_nav),
            }

    if neutral_avg or neutral_weekly:
        neutralized = {
            "target": "vol_cap_off",
            "vol_cap_on_avg": neutral_avg,
            "vol_cap_on_weekly": neutral_weekly,
        }

    compare = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "gate_rule": gate_cfg.rule,
        "gate_action": gate_cfg.action,
        "vol_cap_feature": vol_cap_cfg.feature_name,
        "vol_cap_rank_threshold": float(threshold),
        "runs": {
            "vol_cap_off": metrics_base,
            "vol_cap_on": metrics_var,
        },
        "exposure_neutral": neutralized,
        "diff_on_minus_off": _diff_metrics(
            metrics_base,
            metrics_var,
            [
                "return_pct",
                "end_nav",
                "max_drawdown_pct",
                "min_nav",
                "max_nav",
                "trades",
                "avg_cash_ratio",
                "avg_invested_ratio",
                "return_per_invested",
                "avg_gross_exposure",
                "avg_position_weight",
                "gross_exposure_max",
                "gross_exposure_p95",
                "avg_n_positions",
                "weeks_underinvested",
                "reserve_violation_count",
                "vol_cap_excluded_rate",
                "vol_cap_candidates",
                "vol_cap_excluded",
                "exposure_guard_applied_weeks",
                "exposure_guard_base_scale",
                "exposure_guard_scale_avg",
                "exposure_guard_scale_p95",
                "exposure_guard_scale_min",
                "exposure_guard_scale_max",
            ],
        ),
    }
    thr_label = f"{threshold:.2f}".replace(".", "_")
    file_name = f"volcap_compare_{start_date.isoformat()}_{end_date.isoformat()}_thr_{thr_label}.json"
    write_state(compare, backtest_dir / file_name)


if __name__ == "__main__":
    main()
