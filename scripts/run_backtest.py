from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from typing import List, Optional

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_backtest_dir
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.state import read_state, write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _collect_metrics(cfg, start: Optional[date], end: Optional[date]) -> dict:
    summary = backtest.run(cfg, start=start, end=end)

    backtest_dir = artifacts_backtest_dir(cfg)
    nav_path = backtest_dir / "nav.parquet"
    nav_df = read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce") if not nav_df.empty else pd.Series(dtype=float)

    dd_path = backtest_dir / "audit_max_drawdown.json"
    dd_state = read_state(dd_path) if dd_path.exists() else {}

    gate_closed_weeks = summary.get("regime_gate_closed_weeks") or 0
    total_weeks = int(len(nav_df)) if not nav_df.empty else 0
    gate_closed_rate = float(gate_closed_weeks / total_weeks) if total_weeks else 0.0

    return {
        "return_pct": summary.get("return_pct"),
        "end_nav": summary.get("end_nav"),
        "max_drawdown_pct": dd_state.get("max_drawdown_pct"),
        "min_nav": float(nav_series.min()) if not nav_series.empty else None,
        "max_nav": float(nav_series.max()) if not nav_series.empty else None,
        "trades": summary.get("trades"),
        "avg_cash_ratio": summary.get("avg_cash_ratio"),
        "reserve_violation_count": summary.get("reserve_violation_count"),
        "gate_closed_weeks": int(gate_closed_weeks),
        "gate_closed_rate": gate_closed_rate,
    }


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
    args = parser.parse_args()

    cfg = load_config()
    if not args.compare_regime:
        backtest.run(cfg, start=args.start, end=args.end)
        return

    gate_cfg = cfg.risk.regime_gate
    cfg_off = replace(cfg, risk=replace(cfg.risk, regime_gate=replace(gate_cfg, enabled=False)))
    cfg_on = replace(cfg, risk=replace(cfg.risk, regime_gate=replace(gate_cfg, enabled=True)))

    metrics_off = _collect_metrics(cfg_off, args.start, args.end)
    metrics_on = _collect_metrics(cfg_on, args.start, args.end)

    backtest_dir = artifacts_backtest_dir(cfg)
    compare = {
        "start_date": (args.start or date.fromisoformat(cfg.backtest.start_date)).isoformat(),
        "end_date": (args.end or date.fromisoformat(cfg.backtest.end_date)).isoformat(),
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
                "reserve_violation_count",
                "gate_closed_weeks",
                "gate_closed_rate",
            ],
        ),
    }
    write_state(compare, backtest_dir / "regime_compare.json")


if __name__ == "__main__":
    main()
