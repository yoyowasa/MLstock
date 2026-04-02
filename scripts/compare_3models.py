"""3-model comparison: Ridge vs LGBM vs Ensemble(0.70)."""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(r"C:\BOT\MLStock\artifacts\backtest")

models = {
    "Ridge": ("nav_ridge.parquet", "trades_ridge.parquet", "summary_ridge.json"),
    "LGBM": ("nav_lgbm.parquet", "trades_lgbm.parquet", "summary_lgbm.json"),
    "E_70": ("nav_ensemble_70.parquet", "trades_ensemble_70.parquet", "summary_ensemble_70.json"),
}

data = {}
for label, (nav_f, trades_f, summary_f) in models.items():
    nav = pd.read_parquet(BASE / nav_f)
    nav["week_start"] = pd.to_datetime(nav["week_start"])
    trades = pd.read_parquet(BASE / trades_f)
    trades["week_start"] = pd.to_datetime(trades["week_start"])
    with open(BASE / summary_f) as f:
        summary = json.load(f)
    data[label] = {"nav": nav, "trades": trades, "summary": summary}

# ── SECTION 1: SUMMARY ──────────────────────────────────────────
print("=" * 90)
print("SECTION 1: OVERALL SUMMARY")
print("=" * 90)
print(f"{'Metric':<30} {'Ridge':>15} {'LGBM':>15} {'E_70':>15}")
print("-" * 75)
for m in ["end_nav", "return_pct", "trades", "avg_cash_ratio"]:
    vals = [data[l]["summary"].get(m, 0) for l in ["Ridge", "LGBM", "E_70"]]
    print(f"{m:<30} {vals[0]:>15.4f} {vals[1]:>15.4f} {vals[2]:>15.4f}")

# ── SECTION 2: PNL SOURCE BREAKDOWN ──────────────────────────────
print()
print("=" * 90)
print("SECTION 2: PNL SOURCE BREAKDOWN (Real Trades vs KEEPs)")
print("=" * 90)
print(f"{'Metric':<30} {'Ridge':>15} {'LGBM':>15} {'E_70':>15}")
print("-" * 75)

for label in ["Ridge", "LGBM", "E_70"]:
    t = data[label]["trades"]
    real = t[(t["buy_cost"] > 0) | (t["sell_cost"] > 0)]
    keep = t[(t["buy_cost"] == 0) & (t["sell_cost"] == 0)]
    data[label]["real_pnl"] = real["pnl"].sum()
    data[label]["keep_pnl"] = keep["pnl"].sum()
    data[label]["real_count"] = len(real)
    data[label]["keep_count"] = len(keep)
    data[label]["real_wr"] = (real["pnl"] > 0).mean() * 100 if len(real) > 0 else 0
    data[label]["keep_wr"] = (keep["pnl"] > 0).mean() * 100 if len(keep) > 0 else 0
    wins = real[real["pnl"] > 0]
    losses = real[real["pnl"] <= 0]
    data[label]["real_wl"] = abs(wins["pnl"].mean() / losses["pnl"].mean()) if len(wins) > 0 and len(losses) > 0 else 0

for m in ["real_count", "keep_count", "real_pnl", "keep_pnl", "real_wr", "keep_wr", "real_wl"]:
    vals = [data[l][m] for l in ["Ridge", "LGBM", "E_70"]]
    fmt = ".2f" if isinstance(vals[0], float) else "d"
    print(f"{m:<30} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f}")

# ── SECTION 3: YEARLY ──────────────────────────────────────────
print()
print("=" * 90)
print("SECTION 3: YEARLY RETURNS")
print("=" * 90)


def yearly_ret(nav_df):
    nav_df = nav_df.sort_values("week_start").copy()
    nav_df["year"] = nav_df["week_start"].dt.year
    rows = {}
    for yr in sorted(nav_df["year"].unique()):
        yd = nav_df[nav_df["year"] == yr].sort_values("week_start")
        if len(yd) < 2:
            continue
        rows[yr] = (yd.iloc[-1]["nav"] / yd.iloc[0]["nav"] - 1) * 100
    return rows


yr_data = {l: yearly_ret(data[l]["nav"]) for l in models}
all_years = sorted(set().union(*(yr_data[l].keys() for l in models)))
print(f"{'Year':<8} {'Ridge %':>10} {'LGBM %':>10} {'E_70 %':>10} {'Best':>10}")
print("-" * 55)
for yr in all_years:
    vals = {l: yr_data[l].get(yr, 0) for l in models}
    best = max(vals, key=lambda k: vals[k])
    print(f"{yr:<8} {vals['Ridge']:>+10.2f} {vals['LGBM']:>+10.2f} {vals['E_70']:>+10.2f} {best:>10}")

# ── SECTION 4: RISK METRICS ──────────────────────────────────────
print()
print("=" * 90)
print("SECTION 4: RISK-ADJUSTED METRICS")
print("=" * 90)


def calc_risk(nav_df, label):
    nav_df = nav_df.sort_values("week_start").copy()
    navs = nav_df["nav"].values.astype(float)
    wr = np.diff(navs) / navs[:-1]
    wr = wr[np.isfinite(wr)]
    ann_ret = np.mean(wr) * 52
    ann_vol = np.std(wr) * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    down = wr[wr < 0]
    down_vol = np.std(down) * np.sqrt(52) if len(down) > 0 else 0
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    max_dd = abs(dd.min())
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    pos = wr[wr > 0].sum()
    neg = abs(wr[wr < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")

    # DD period details
    max_dd_idx = np.argmin(dd)
    peak_idx = np.argmax(navs[: max_dd_idx + 1]) if max_dd_idx > 0 else 0
    peak_date = nav_df.iloc[peak_idx]["week_start"].strftime("%Y-%m-%d")
    trough_date = nav_df.iloc[max_dd_idx]["week_start"].strftime("%Y-%m-%d")
    recovery = np.where(navs[max_dd_idx:] >= peaks[max_dd_idx])[0]
    recovery_weeks = (max_dd_idx + recovery[0] - peak_idx) if len(recovery) > 0 else "N/R"

    return {
        "ann_ret%": ann_ret * 100,
        "ann_vol%": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd%": -max_dd * 100,
        "profit_factor": pf,
        "weekly_wr%": (wr > 0).mean() * 100,
        "best_wk%": wr.max() * 100,
        "worst_wk%": wr.min() * 100,
        "dd_peak": peak_date,
        "dd_trough": trough_date,
        "dd_recovery_wk": recovery_weeks,
        "max_lose_streak": 0,
    }


risk = {}
for label in models:
    risk[label] = calc_risk(data[label]["nav"], label)

print(f"{'Metric':<25} {'Ridge':>15} {'LGBM':>15} {'E_70':>15}")
print("-" * 70)
for m in [
    "ann_ret%",
    "ann_vol%",
    "sharpe",
    "sortino",
    "calmar",
    "max_dd%",
    "profit_factor",
    "weekly_wr%",
    "best_wk%",
    "worst_wk%",
    "dd_peak",
    "dd_trough",
    "dd_recovery_wk",
]:
    vals = [risk[l].get(m, "") for l in models]
    if all(isinstance(v, (int, float)) for v in vals):
        print(f"{m:<25} {vals[0]:>15.3f} {vals[1]:>15.3f} {vals[2]:>15.3f}")
    else:
        print(f"{m:<25} {str(vals[0]):>15} {str(vals[1]):>15} {str(vals[2]):>15}")

# ── SECTION 5: QUARTERLY WINS ──────────────────────────────────────
print()
print("=" * 90)
print("SECTION 5: QUARTERLY WIN COMPARISON")
print("=" * 90)


def quarterly_ret(nav_df):
    nav_df = nav_df.sort_values("week_start").copy()
    nav_df["q"] = nav_df["week_start"].dt.to_period("Q")
    rows = {}
    for q in sorted(nav_df["q"].unique()):
        qd = nav_df[nav_df["q"] == q].sort_values("week_start")
        if len(qd) < 2:
            continue
        rows[str(q)] = (qd.iloc[-1]["nav"] / qd.iloc[0]["nav"] - 1) * 100
    return rows


qr = {l: quarterly_ret(data[l]["nav"]) for l in models}
all_q = sorted(set().union(*(qr[l].keys() for l in models)))

wins = {"Ridge": 0, "LGBM": 0, "E_70": 0}
print(f"{'Quarter':<10} {'Ridge':>10} {'LGBM':>10} {'E_70':>10} {'Best':>10}")
print("-" * 55)
for q in all_q:
    vals = {l: qr[l].get(q, 0) for l in models}
    best = max(vals, key=lambda k: vals[k])
    if vals[best] != 0:
        wins[best] += 1
    print(f"{q:<10} {vals['Ridge']:>+10.2f} {vals['LGBM']:>+10.2f} {vals['E_70']:>+10.2f} {best:>10}")

print(f"\nQuarterly wins: Ridge={wins['Ridge']}  LGBM={wins['LGBM']}  E_70={wins['E_70']}")

# ── SECTION 6: TOTAL COST COMPARISON ──────────────────────────────
print()
print("=" * 90)
print("SECTION 6: TRANSACTION COSTS")
print("=" * 90)
for label in models:
    t = data[label]["trades"]
    total_cost = t["buy_cost"].sum() + t["sell_cost"].sum()
    print(f"  {label}: total_cost=${total_cost:.2f}  trades={len(t)}")

print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
