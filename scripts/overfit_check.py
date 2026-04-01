"""Overfitting check: compare ensemble weights across sub-periods."""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\BOT\MLStock\artifacts\backtest")

weights = [30, 40, 50, 60, 70, 75, 80, 85, 90, 95]
labels_pure = {"Ridge": "nav_ridge.parquet", "LGBM": "nav_lgbm.parquet"}

# Load all nav data
all_navs = {}
for label, fname in labels_pure.items():
    df = pd.read_parquet(BASE / fname)
    df["week_start"] = pd.to_datetime(df["week_start"])
    all_navs[label] = df.sort_values("week_start")

for w in weights:
    fname = f"nav_ensemble_{w:02d}.parquet"
    p = BASE / fname
    if p.exists():
        df = pd.read_parquet(p)
        df["week_start"] = pd.to_datetime(df["week_start"])
        all_navs[f"E_{w}"] = df.sort_values("week_start")

# Define sub-periods
periods = [
    ("Full 2020-2024", None, None),
    ("2020H2-2021", "2020-07-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("Bear 2021Q4-2022Q1", "2021-10-01", "2022-03-31"),
    ("Bull 2023Q1-Q2", "2023-01-01", "2023-06-30"),
]

def period_return(nav_df: pd.DataFrame, start, end):
    df = nav_df.copy()
    if start:
        df = df[df["week_start"] >= pd.Timestamp(start)]
    if end:
        df = df[df["week_start"] <= pd.Timestamp(end)]
    if len(df) < 2:
        return None
    return (df.iloc[-1]["nav"] / df.iloc[0]["nav"] - 1) * 100

def period_max_dd(nav_df: pd.DataFrame, start, end):
    df = nav_df.copy()
    if start:
        df = df[df["week_start"] >= pd.Timestamp(start)]
    if end:
        df = df[df["week_start"] <= pd.Timestamp(end)]
    if len(df) < 2:
        return None
    navs = df["nav"].values.astype(float)
    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    return dd.min() * 100

def period_sharpe(nav_df: pd.DataFrame, start, end):
    df = nav_df.copy()
    if start:
        df = df[df["week_start"] >= pd.Timestamp(start)]
    if end:
        df = df[df["week_start"] <= pd.Timestamp(end)]
    if len(df) < 3:
        return None
    navs = df["nav"].values.astype(float)
    wr = np.diff(navs) / navs[:-1]
    wr = wr[np.isfinite(wr)]
    if len(wr) < 2 or np.std(wr) == 0:
        return None
    return (np.mean(wr) / np.std(wr)) * np.sqrt(52)

# ── SECTION 1: RETURN BY PERIOD ──────────────────────────────
print("=" * 120)
print("SECTION 1: RETURN % BY PERIOD (Overfitting stability check)")
print("=" * 120)

model_labels = ["Ridge", "LGBM"] + [f"E_{w}" for w in weights if f"E_{w}" in all_navs]
header = f"{'Period':<25}" + "".join(f"{m:>10}" for m in model_labels)
print(header)
print("-" * len(header))

for pname, start, end in periods:
    row = f"{pname:<25}"
    for m in model_labels:
        ret = period_return(all_navs[m], start, end)
        row += f"{ret:>+10.2f}" if ret is not None else f"{'N/A':>10}"
    print(row)

# ── SECTION 2: BEST ENSEMBLE WEIGHT PER PERIOD ──────────────────
print()
print("=" * 120)
print("SECTION 2: BEST ENSEMBLE WEIGHT PER PERIOD")
print("=" * 120)

ew_labels = [f"E_{w}" for w in weights if f"E_{w}" in all_navs]
print(f"{'Period':<25} {'Best':>8} {'Ret%':>8} {'2nd':>8} {'Ret%':>8} {'Worst':>8} {'Ret%':>8}  {'Ridge':>8} {'LGBM':>8}")
print("-" * 110)

for pname, start, end in periods:
    rets = {}
    for m in ew_labels:
        r = period_return(all_navs[m], start, end)
        if r is not None:
            rets[m] = r
    ridge_ret = period_return(all_navs["Ridge"], start, end)
    lgbm_ret = period_return(all_navs["LGBM"], start, end)

    if rets:
        sorted_rets = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        best = sorted_rets[0]
        second = sorted_rets[1] if len(sorted_rets) > 1 else ("N/A", 0)
        worst = sorted_rets[-1]
        rr = f"{ridge_ret:>+8.2f}" if ridge_ret is not None else f"{'N/A':>8}"
        lr = f"{lgbm_ret:>+8.2f}" if lgbm_ret is not None else f"{'N/A':>8}"
        print(f"{pname:<25} {best[0]:>8} {best[1]:>+8.2f} {second[0]:>8} {second[1]:>+8.2f} "
              f"{worst[0]:>8} {worst[1]:>+8.2f}  {rr} {lr}")

# ── SECTION 3: SHARPE BY PERIOD ──────────────────────────────
print()
print("=" * 120)
print("SECTION 3: SHARPE RATIO BY PERIOD")
print("=" * 120)

header = f"{'Period':<25}" + "".join(f"{m:>10}" for m in model_labels)
print(header)
print("-" * len(header))

for pname, start, end in periods:
    row = f"{pname:<25}"
    for m in model_labels:
        s = period_sharpe(all_navs[m], start, end)
        row += f"{s:>10.3f}" if s is not None else f"{'N/A':>10}"
    print(row)

# ── SECTION 4: MAX DRAWDOWN BY PERIOD ────────────────────────
print()
print("=" * 120)
print("SECTION 4: MAX DRAWDOWN % BY PERIOD")
print("=" * 120)

header = f"{'Period':<25}" + "".join(f"{m:>10}" for m in model_labels)
print(header)
print("-" * len(header))

for pname, start, end in periods:
    row = f"{pname:<25}"
    for m in model_labels:
        dd = period_max_dd(all_navs[m], start, end)
        row += f"{dd:>10.2f}" if dd is not None else f"{'N/A':>10}"
    print(row)

# ── SECTION 5: WEIGHT CONSISTENCY SCORE ──────────────────────
print()
print("=" * 120)
print("SECTION 5: WEIGHT CONSISTENCY ANALYSIS")
print("=" * 120)

# For each sub-period, rank the ensemble weights by return
# Check if any weight is consistently good
rank_sums = {m: 0 for m in ew_labels}
rank_counts = {m: 0 for m in ew_labels}

for pname, start, end in periods[1:]:  # Skip full period
    rets = {}
    for m in ew_labels:
        r = period_return(all_navs[m], start, end)
        if r is not None:
            rets[m] = r
    if rets:
        sorted_models = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        for rank, (m, _) in enumerate(sorted_models, 1):
            rank_sums[m] += rank
            rank_counts[m] += 1

print(f"{'Weight':<10} {'Avg Rank':>10} {'Times Ranked':>15}")
print("-" * 40)
sorted_by_rank = sorted(rank_sums.items(), key=lambda x: x[1] / max(rank_counts[x[0]], 1))
for m, rs in sorted_by_rank:
    cnt = rank_counts[m]
    avg = rs / cnt if cnt > 0 else 999
    print(f"{m:<10} {avg:>10.1f} {cnt:>15}")

# Also check: does E_70 beat Ridge in EACH sub-period?
print()
print("E_70 vs Ridge by sub-period:")
for pname, start, end in periods[1:]:
    r70 = period_return(all_navs.get("E_70", pd.DataFrame()), start, end)
    rr = period_return(all_navs["Ridge"], start, end)
    if r70 is not None and rr is not None:
        winner = "E_70" if r70 > rr else "Ridge"
        diff = r70 - rr
        print(f"  {pname:<25} E_70={r70:>+7.2f}%  Ridge={rr:>+7.2f}%  diff={diff:>+7.2f}  → {winner}")

print()
print("E_70 vs LGBM by sub-period:")
for pname, start, end in periods[1:]:
    r70 = period_return(all_navs.get("E_70", pd.DataFrame()), start, end)
    rl = period_return(all_navs["LGBM"], start, end)
    if r70 is not None and rl is not None:
        winner = "E_70" if r70 > rl else "LGBM"
        diff = r70 - rl
        print(f"  {pname:<25} E_70={r70:>+7.2f}%  LGBM={rl:>+7.2f}%  diff={diff:>+7.2f}  → {winner}")

# ── SECTION 6: IS THE CURVE SHAPE NOISE OR SIGNAL? ──────────
print()
print("=" * 120)
print("SECTION 6: WEIGHT-RETURN CURVE SMOOTHNESS")
print("=" * 120)
print("If the curve is smooth (monotonic or single peak), it's likely signal.")
print("If jagged with multiple peaks, it's likely noise/overfit.\n")

for pname, start, end in periods:
    rets = []
    for w in sorted(weights):
        m = f"E_{w}"
        if m in all_navs:
            r = period_return(all_navs[m], start, end)
            rets.append((w, r))

    if rets:
        vals = [r for _, r in rets if r is not None]
        # Count direction changes
        changes = 0
        for i in range(1, len(vals)):
            if i > 0 and (vals[i] - vals[i-1]) * (vals[i-1] - vals[max(0,i-2)]) < 0:
                changes += 1

        line = f"  {pname:<25}: "
        for w, r in rets:
            line += f"{w/100:.2f}→{r:>+.1f}  "
        line += f"  [direction changes: {changes}]"
        print(line)

print()
print("=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)
