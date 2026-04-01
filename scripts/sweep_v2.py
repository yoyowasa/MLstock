"""Sweep confidence_threshold and price_cap to find optimal V2 settings."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path("C:/BOT/MLStock")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
BACKTEST_DIR = BASE_DIR / "artifacts" / "backtest"
RESULTS_DIR = BACKTEST_DIR / "sweep_v2"


def load_yaml():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def run_backtest():
    from mlstock.config.loader import load_config
    from mlstock.jobs import backtest

    cfg = load_config()
    return backtest.run(cfg)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    original = load_yaml()

    # Test configurations: (label, price_cap, confidence_threshold, confidence_sizing)
    configs = [
        # Baseline: old E_70 settings + new features, no confidence sizing
        ("baseline_feat17", 60, 0.0, False),
        # Variable N with different thresholds, price_cap=60
        ("varN_t0000_p60", 60, 0.0, True),
        ("varN_t0005_p60", 60, 0.005, True),
        ("varN_t0010_p60", 60, 0.01, True),
        ("varN_t0020_p60", 60, 0.02, True),
        # price_cap=100 variants
        ("varN_t0005_p100", 100, 0.005, True),
        ("varN_t0010_p100", 100, 0.01, True),
    ]

    results = {}
    for label, price_cap, threshold, sizing_on in configs:
        print(f"\n{'='*60}")
        print(f"Running: {label} (price_cap={price_cap}, threshold={threshold}, sizing={sizing_on})")
        print(f"{'='*60}")

        cfg = copy.deepcopy(original)
        cfg["selection"]["price_cap"] = price_cap
        cfg["selection"]["confidence_threshold"] = threshold
        cfg["selection"]["confidence_sizing"] = sizing_on
        # Keep min_price at 5.0 for all
        cfg["snapshots"]["min_price"] = 5.0
        save_yaml(cfg)

        try:
            summary = run_backtest()

            # Save artifacts
            for fname in ["nav.parquet", "trades.parquet", "summary.json"]:
                src = BACKTEST_DIR / fname
                dst = RESULTS_DIR / f"{fname.split('.')[0]}_{label}.{fname.split('.')[1]}"
                if src.exists():
                    shutil.copy2(src, dst)

            import pandas as pd
            nav_df = pd.read_parquet(BACKTEST_DIR / "nav.parquet")
            nav_series = pd.to_numeric(nav_df["nav"], errors="coerce")
            dd = nav_series / nav_series.cummax() - 1
            max_dd = float(dd.min())

            avg_pos = float(nav_df["n_positions"].mean()) if "n_positions" in nav_df.columns else None
            pos_dist = nav_df["n_positions"].value_counts().to_dict() if "n_positions" in nav_df.columns else {}

            results[label] = {
                "price_cap": price_cap,
                "confidence_threshold": threshold,
                "confidence_sizing": sizing_on,
                "end_nav": summary.get("end_nav"),
                "return_pct": summary.get("return_pct"),
                "trades": summary.get("trades"),
                "avg_cash_ratio": summary.get("avg_cash_ratio"),
                "max_drawdown_pct": max_dd,
                "avg_n_positions": avg_pos,
                "pos_dist": {str(k): v for k, v in pos_dist.items()},
            }
            r = results[label]
            print(f"  Return: {r['return_pct']*100:.2f}%  DD: {r['max_drawdown_pct']*100:.2f}%  "
                  f"AvgPos: {r['avg_n_positions']:.1f}  Cash: {r['avg_cash_ratio']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[label] = {"error": str(e)}

    # Restore original config
    save_yaml(original)

    # Save results
    with open(RESULTS_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n\n{'='*90}")
    print(f"{'Label':<25} {'Return%':>8} {'MaxDD%':>8} {'AvgPos':>8} {'Cash%':>8} {'Trades':>8}")
    print(f"{'='*90}")
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<25} ERROR: {r['error'][:40]}")
            continue
        print(f"{label:<25} {r['return_pct']*100:>8.2f} {r['max_drawdown_pct']*100:>8.2f} "
              f"{r['avg_n_positions']:>8.1f} {r['avg_cash_ratio']*100:>8.2f} {r['trades']:>8d}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
