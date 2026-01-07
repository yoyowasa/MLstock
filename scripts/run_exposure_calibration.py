from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import artifacts_backtest_dir
from mlstock.data.storage.state import read_state, write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _calc_gross_exposure(nav_df: pd.DataFrame) -> pd.Series:
    if nav_df.empty or "nav" not in nav_df.columns:
        return pd.Series(dtype=float)

    nav = pd.to_numeric(nav_df["nav"], errors="coerce")
    nav_nonzero = nav.mask(nav == 0)
    if "positions_value" in nav_df.columns:
        positions_value = pd.to_numeric(nav_df["positions_value"], errors="coerce")
        gross = positions_value / nav_nonzero
    elif "cash_usd" in nav_df.columns:
        cash = pd.to_numeric(nav_df["cash_usd"], errors="coerce")
        gross = 1.0 - (cash / nav_nonzero)
    else:
        gross = pd.Series(dtype=float)
    return gross.dropna()


def _collect_run_metrics(backtest_dir: Path) -> dict:
    summary_path = backtest_dir / "summary.json"
    summary = read_state(summary_path) if summary_path.exists() else {}

    dd_path = backtest_dir / "audit_max_drawdown.json"
    dd_state = read_state(dd_path) if dd_path.exists() else {}

    nav_path = backtest_dir / "nav.parquet"
    nav_df = read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
    if not nav_df.empty and "week_start" in nav_df.columns:
        nav_df = nav_df.sort_values("week_start").reset_index(drop=True)

    gross_exposure = _calc_gross_exposure(nav_df)
    gross_exposure_avg = float(gross_exposure.mean()) if not gross_exposure.empty else None
    gross_exposure_p95 = float(gross_exposure.quantile(0.95)) if not gross_exposure.empty else None
    gross_exposure_max = float(gross_exposure.max()) if not gross_exposure.empty else None

    n_positions = nav_df["n_positions"] if "n_positions" in nav_df.columns else pd.Series(dtype=float)
    avg_n_positions = float(n_positions.mean()) if not n_positions.empty else None

    return {
        "eval_weeks": int(len(nav_df)) if not nav_df.empty else 0,
        "return_pct": _as_float(summary.get("return_pct")),
        "max_drawdown_pct": _as_float(dd_state.get("max_drawdown_pct")),
        "trades": _as_float(summary.get("trades")),
        "avg_cash_ratio": _as_float(summary.get("avg_cash_ratio")),
        "avg_gross_exposure": gross_exposure_avg,
        "gross_exposure_p95": gross_exposure_p95,
        "gross_exposure_max": gross_exposure_max,
        "avg_n_positions": avg_n_positions,
    }


def _clip_scale(value: Optional[float], clip_min: Optional[float], clip_max: Optional[float]) -> Optional[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--volcap-threshold", type=float, default=None)
    parser.add_argument("--base-scale-clip-min", type=float, default=None)
    parser.add_argument("--base-scale-clip-max", type=float, default=None)
    parser.add_argument("--include-guard-run", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.risk.vol_cap.enabled:
        raise ValueError("vol_cap.enabled is false. Enable it to run exposure calibration.")
    start_date = args.start or date.fromisoformat(cfg.backtest.start_date)
    end_date = args.end or date.fromisoformat(cfg.backtest.end_date)
    backtest_dir = artifacts_backtest_dir(cfg)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    vol_cap_cfg = cfg.risk.vol_cap
    threshold = args.volcap_threshold if args.volcap_threshold is not None else vol_cap_cfg.rank_threshold

    guard_cfg = cfg.risk.exposure_guard
    cap_source = guard_cfg.cap_source
    cap_buffer = float(guard_cfg.cap_buffer)
    base_source = guard_cfg.base_source

    cfg_off = replace(
        cfg,
        risk=replace(
            cfg.risk,
            vol_cap=replace(vol_cap_cfg, enabled=False, rank_threshold=threshold),
            exposure_guard=replace(guard_cfg, enabled=False),
        ),
    )
    backtest.run(cfg_off, start=start_date, end=end_date)
    metrics_off = _collect_run_metrics(backtest_dir)

    cfg_on_raw = replace(
        cfg,
        risk=replace(
            cfg.risk,
            vol_cap=replace(vol_cap_cfg, enabled=True, rank_threshold=threshold),
            exposure_guard=replace(guard_cfg, enabled=False),
        ),
    )
    backtest.run(cfg_on_raw, start=start_date, end=end_date)
    metrics_on_raw = _collect_run_metrics(backtest_dir)

    avg_gross_off = metrics_off.get("avg_gross_exposure")
    avg_gross_on_raw = metrics_on_raw.get("avg_gross_exposure")
    base_scale_raw = None
    if isinstance(avg_gross_off, (int, float)) and isinstance(avg_gross_on_raw, (int, float)):
        if avg_gross_on_raw > 0:
            base_scale_raw = float(avg_gross_off) / float(avg_gross_on_raw)
    base_scale = _clip_scale(base_scale_raw, args.base_scale_clip_min, args.base_scale_clip_max)

    cap_value_raw = None
    if cap_source == "off_p95":
        cap_value_raw = metrics_off.get("gross_exposure_p95")
    elif cap_source == "off_avg":
        cap_value_raw = metrics_off.get("avg_gross_exposure")
    elif cap_source == "fixed":
        cap_value_raw = guard_cfg.cap_value

    cap_value = None
    if isinstance(cap_value_raw, (int, float)):
        cap_value = float(cap_value_raw)

    metrics_on_guard = None
    guard_used = None
    if args.include_guard_run:
        guard_cfg_variant = replace(guard_cfg, enabled=True)
        if guard_cfg_variant.base_source in ("off_avg_on_avg", "fixed") and base_scale is not None:
            guard_cfg_variant = replace(guard_cfg_variant, base_scale=base_scale)
        if guard_cfg_variant.cap_source in ("off_p95", "off_avg", "fixed") and cap_value is not None:
            guard_cfg_variant = replace(guard_cfg_variant, cap_value=cap_value)
        cfg_on_guard = replace(
            cfg,
            risk=replace(
                cfg.risk,
                vol_cap=replace(vol_cap_cfg, enabled=True, rank_threshold=threshold),
                exposure_guard=guard_cfg_variant,
            ),
        )
        backtest.run(cfg_on_guard, start=start_date, end=end_date)
        metrics_on_guard = _collect_run_metrics(backtest_dir)
        guard_used = {
            "base_source": guard_cfg_variant.base_source,
            "base_scale": guard_cfg_variant.base_scale,
            "cap_source": guard_cfg_variant.cap_source,
            "cap_value": guard_cfg_variant.cap_value,
            "cap_buffer": float(guard_cfg_variant.cap_buffer),
        }

    output_path = (
        Path(args.output)
        if args.output
        else backtest_dir / f"exposure_calibration_{start_date.isoformat()}_{end_date.isoformat()}.json"
    )
    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "vol_cap_rank_threshold": float(threshold),
        "base_source": base_source,
        "cap_source": cap_source,
        "cap_buffer": cap_buffer,
        "base_scale_clip_min": args.base_scale_clip_min,
        "base_scale_clip_max": args.base_scale_clip_max,
        "runs": {
            "vol_cap_off": metrics_off,
            "vol_cap_on_raw": metrics_on_raw,
            "vol_cap_on_guard": metrics_on_guard,
        },
        "calibration": {
            "avg_gross_off": avg_gross_off,
            "avg_gross_on_raw": avg_gross_on_raw,
            "gross_p95_off": metrics_off.get("gross_exposure_p95"),
            "base_scale_raw": base_scale_raw,
            "base_scale": base_scale,
            "cap_value_raw": cap_value_raw,
            "cap_value": cap_value,
        },
        "guard_run_used": guard_used,
    }
    write_state(payload, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
