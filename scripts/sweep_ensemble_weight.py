"""Sweep ensemble_weight_ridge from 0.3 to 0.7 and compare results."""
from __future__ import annotations

import subprocess
import shutil
import json
from pathlib import Path

import yaml

CONFIG_PATH = Path(r"C:\BOT\MLStock\config\config.yaml")
BACKTEST_DIR = Path(r"C:\BOT\MLStock\artifacts\backtest")
PYTHON = r"C:\BOT\MLStock\.venv\Scripts\python.exe"
SCRIPT = r"C:\BOT\MLStock\scripts\run_backtest.py"

weights = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for w in weights:
    print(f"\n{'='*60}")
    print(f"Running ensemble_weight_ridge = {w}")
    print(f"{'='*60}")

    # Update config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["training"]["model_type"] = "ensemble"
    config["training"]["ensemble_weight_ridge"] = w

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Run backtest
    result = subprocess.run([PYTHON, SCRIPT], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        continue

    # Read summary
    summary_path = BACKTEST_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Save copies
    suffix = f"ensemble_{int(w*100):02d}"
    shutil.copy2(BACKTEST_DIR / "nav.parquet", BACKTEST_DIR / f"nav_{suffix}.parquet")
    shutil.copy2(BACKTEST_DIR / "trades.parquet", BACKTEST_DIR / f"trades_{suffix}.parquet")
    shutil.copy2(summary_path, BACKTEST_DIR / f"summary_{suffix}.json")

    results.append({
        "weight": w,
        "end_nav": summary["end_nav"],
        "return_pct": summary["return_pct"],
        "trades": summary["trades"],
        "avg_cash_ratio": summary["avg_cash_ratio"],
    })
    print(f"  end_nav={summary['end_nav']:.2f}  return={summary['return_pct']*100:.2f}%  trades={summary['trades']}")

# Final comparison table
print(f"\n{'='*80}")
print("ENSEMBLE WEIGHT SWEEP RESULTS")
print(f"{'='*80}")
print(f"{'Weight':<10} {'End NAV':>12} {'Return %':>12} {'Trades':>10} {'Cash Ratio':>12}")
print("-" * 60)

# Add Ridge and LGBM baselines
ridge_path = BACKTEST_DIR / "summary_ridge.json"
lgbm_path = BACKTEST_DIR / "summary_lgbm.json"
if ridge_path.exists():
    with open(ridge_path) as f:
        sr = json.load(f)
    print(f"{'Ridge':<10} {sr['end_nav']:>12.2f} {sr['return_pct']*100:>12.2f} {sr['trades']:>10} {sr['avg_cash_ratio']:>12.4f}")
if lgbm_path.exists():
    with open(lgbm_path) as f:
        sl = json.load(f)
    print(f"{'LGBM':<10} {sl['end_nav']:>12.2f} {sl['return_pct']*100:>12.2f} {sl['trades']:>10} {sl['avg_cash_ratio']:>12.4f}")
print("-" * 60)

for r in results:
    print(f"{'E_'+str(r['weight']):<10} {r['end_nav']:>12.2f} {r['return_pct']*100:>12.2f} {r['trades']:>10} {r['avg_cash_ratio']:>12.4f}")

best = max(results, key=lambda x: x["return_pct"])
print(f"\nBest: weight={best['weight']}  return={best['return_pct']*100:.2f}%  nav={best['end_nav']:.2f}")
