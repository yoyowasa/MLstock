"""Extended sweep: 0.75, 0.80, 0.85, 0.90, 0.95."""
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

weights = [0.75, 0.80, 0.85, 0.90, 0.95]
results = []

for w in weights:
    print(f"\nRunning ensemble_weight_ridge = {w}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["training"]["model_type"] = "ensemble"
    config["training"]["ensemble_weight_ridge"] = w

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    result = subprocess.run([PYTHON, SCRIPT], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        continue

    summary_path = BACKTEST_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

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

# Full comparison
print(f"\n{'='*80}")
print("FULL SWEEP RESULTS (including previous)")
print(f"{'='*80}")
print(f"{'Model':<12} {'End NAV':>12} {'Return %':>12} {'Trades':>10}")
print("-" * 50)

# Load all previous results
baselines = [
    ("Ridge", "summary_ridge.json"),
    ("LGBM", "summary_lgbm.json"),
    ("E_30", "summary_ensemble_30.json"),
    ("E_40", "summary_ensemble_40.json"),
    ("E_50", "summary_ensemble_50.json"),
    ("E_60", "summary_ensemble_60.json"),
    ("E_70", "summary_ensemble_70.json"),
]
for label, fname in baselines:
    p = BACKTEST_DIR / fname
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        print(f"{label:<12} {s['end_nav']:>12.2f} {s['return_pct']*100:>12.2f} {s['trades']:>10}")

for r in results:
    label = f"E_{int(r['weight']*100):02d}"
    print(f"{label:<12} {r['end_nav']:>12.2f} {r['return_pct']*100:>12.2f} {r['trades']:>10}")
