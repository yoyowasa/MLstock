from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from itertools import product
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import artifacts_backtest_dir
from mlstock.data.storage.state import read_state, write_state
from mlstock.jobs import backtest


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    parts = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return [float(item) for item in parts]


def _collect_metrics(backtest_dir: Path, cfg) -> dict:
    summary_path = backtest_dir / "summary.json"
    summary = read_state(summary_path) if summary_path.exists() else {}

    dd_path = backtest_dir / "audit_max_drawdown.json"
    dd_state = read_state(dd_path) if dd_path.exists() else {}

    nav_path = backtest_dir / "nav.parquet"
    nav_df = read_parquet(nav_path) if nav_path.exists() else pd.DataFrame()
    if not nav_df.empty and "week_start" in nav_df.columns:
        nav_df = nav_df.sort_values("week_start").reset_index(drop=True)
    n_positions = nav_df["n_positions"] if "n_positions" in nav_df.columns else pd.Series(dtype=float)

    max_positions = max(int(cfg.selection.max_positions), 1)
    underinvested_threshold = min(10, max_positions)

    return {
        "return_pct": summary.get("return_pct"),
        "max_drawdown_pct": dd_state.get("max_drawdown_pct"),
        "trades": summary.get("trades"),
        "avg_cash_ratio": summary.get("avg_cash_ratio"),
        "avg_n_positions": float(n_positions.mean()) if not n_positions.empty else None,
        "weeks_underinvested": int((n_positions < underinvested_threshold).sum()) if not n_positions.empty else None,
    }


def _score(metrics: dict) -> tuple:
    return (
        metrics.get("return_pct") if metrics.get("return_pct") is not None else float("-inf"),
        metrics.get("max_drawdown_pct") if metrics.get("max_drawdown_pct") is not None else float("-inf"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument(
        "--min-proba-buy",
        type=str,
        default="0.55,0.60,0.65",
        help="Comma or space separated thresholds (0-1)",
    )
    parser.add_argument(
        "--min-proba-keep",
        type=str,
        default="0.50,0.55,0.60",
        help="Comma or space separated thresholds (0-1)",
    )
    parser.add_argument(
        "--delta-bps",
        type=str,
        default=None,
        help="Optional comma/space separated bps values for estimate_entry_buffer_bps",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    start = args.start or date.fromisoformat(cfg.backtest.start_date)
    end = args.end or date.fromisoformat(cfg.backtest.end_date)

    min_buy_list = _parse_float_list(args.min_proba_buy) or [cfg.selection.min_proba_buy]
    min_keep_list = _parse_float_list(args.min_proba_keep) or [cfg.selection.min_proba_keep]
    delta_bps_list = _parse_float_list(args.delta_bps) or [cfg.selection.estimate_entry_buffer_bps]

    backtest_dir = artifacts_backtest_dir(cfg)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else backtest_dir / "param_sweep_summary.csv"

    audit_files = [
        "summary.json",
        "audit_max_jump.json",
        "audit_min_jump.json",
        "audit_max_drawdown.json",
        "audit_top_share_profile.json",
        "top_share_by_week.csv",
    ]
    best_dir = backtest_dir / "sweep_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_idx = None
    best_score = None

    for min_buy, min_keep, delta_bps in product(min_buy_list, min_keep_list, delta_bps_list):
        selection_cfg = replace(
            cfg.selection,
            min_proba_buy=float(min_buy),
            min_proba_keep=float(min_keep),
            estimate_entry_buffer_bps=float(delta_bps),
        )
        cfg_run = replace(cfg, selection=selection_cfg)

        backtest.run(cfg_run, start=start, end=end)

        metrics = _collect_metrics(backtest_dir, cfg_run)
        row = {
            "min_proba_buy": float(min_buy),
            "min_proba_keep": float(min_keep),
            "delta_bps": float(delta_bps),
            **metrics,
        }
        rows.append(row)

        score = _score(metrics)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = len(rows) - 1
            for name in audit_files:
                src = backtest_dir / name
                if src.exists():
                    dst = best_dir / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())

    df = pd.DataFrame(rows)
    if best_idx is not None:
        df["is_best"] = False
        df.loc[best_idx, "is_best"] = True
    df.to_csv(output_path, index=False)

    if best_dir.exists():
        for name in audit_files:
            src = best_dir / name
            if src.exists():
                dst = backtest_dir / name
                dst.write_bytes(src.read_bytes())

    best_payload = rows[best_idx] if best_idx is not None else {}
    write_state(
        {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "best": best_payload,
            "output_csv": output_path.as_posix(),
        },
        backtest_dir / "param_sweep_best.json",
    )


if __name__ == "__main__":
    main()
