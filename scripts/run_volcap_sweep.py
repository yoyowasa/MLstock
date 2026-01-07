from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_backtest_dir


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    parts = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return [float(item) for item in parts]


def _default_thresholds() -> List[float]:
    return [round(0.50 + 0.05 * i, 2) for i in range(11)]


def _parse_periods(value: Optional[str]) -> Optional[List[Tuple[date, date]]]:
    if not value:
        return None
    periods = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        start_str, end_str = part.split(":")
        periods.append((date.fromisoformat(start_str.strip()), date.fromisoformat(end_str.strip())))
    return periods


def _run_compare(start: date, end: date, threshold: float, *, force: bool) -> Path:
    cmd = [
        sys.executable,
        "scripts/run_backtest.py",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--compare-volcap",
        "--volcap-threshold",
        f"{threshold:.2f}",
    ]
    thr_label = f"{threshold:.2f}".replace(".", "_")
    output = Path(f"volcap_compare_{start.isoformat()}_{end.isoformat()}_thr_{thr_label}.json")
    if output.exists() and not force:
        return output
    subprocess.run(cmd, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--periods", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--append", action="store_true", default=False)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.risk.vol_cap.enabled:
        raise ValueError("vol_cap.enabled is false. Enable it to run volcap sweep.")
    backtest_dir = artifacts_backtest_dir(cfg)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_float_list(args.thresholds) or _default_thresholds()
    periods = _parse_periods(args.periods) or [
        (date.fromisoformat("2020-07-27"), date.fromisoformat("2025-12-15")),
        (date.fromisoformat("2020-07-27"), date.fromisoformat("2021-12-31")),
        (date.fromisoformat("2022-01-01"), date.fromisoformat("2023-12-31")),
        (date.fromisoformat("2024-01-01"), date.fromisoformat("2025-12-15")),
    ]

    output_path = Path(args.output) if args.output else backtest_dir / "volcap_sweep_summary.csv"

    rows = []
    for threshold in thresholds:
        for start, end in periods:
            json_name = _run_compare(start, end, threshold, force=args.force)
            json_path = backtest_dir / json_name
            if not json_path.exists():
                raise FileNotFoundError(f"Missing compare output: {json_path}")
            data = json.loads(json_path.read_text(encoding="utf-8"))

            runs = data.get("runs", {})
            off = runs.get("vol_cap_off", {})
            on = runs.get("vol_cap_on", {})
            diff = data.get("diff_on_minus_off", {})

            rows.append(
                {
                    "threshold": float(threshold),
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                    "diff_return_pct": diff.get("return_pct"),
                    "diff_max_drawdown_pct": diff.get("max_drawdown_pct"),
                    "diff_trades": diff.get("trades"),
                    "diff_avg_cash_ratio": diff.get("avg_cash_ratio"),
                    "diff_avg_n_positions": diff.get("avg_n_positions"),
                    "diff_weeks_underinvested": diff.get("weeks_underinvested"),
                    "off_return_pct": off.get("return_pct"),
                    "off_max_drawdown_pct": off.get("max_drawdown_pct"),
                    "on_return_pct": on.get("return_pct"),
                    "on_max_drawdown_pct": on.get("max_drawdown_pct"),
                    "vol_cap_excluded_rate": on.get("vol_cap_excluded_rate"),
                    "exposure_guard_cap": on.get("exposure_guard_cap"),
                    "exposure_guard_base_scale": on.get("exposure_guard_base_scale"),
                }
            )

    df = pd.DataFrame(rows)
    if args.append and output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["threshold", "period_start", "period_end"], keep="last")
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
