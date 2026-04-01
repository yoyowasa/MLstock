"""Ridge vs LGBM backtest deep comparison analysis."""
from __future__ import annotations

import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(r"C:\BOT\MLStock\artifacts\backtest")

# ── Load data ──────────────────────────────────────────────────────────────
nav_r = pd.read_parquet(BASE / "nav_ridge.parquet")
nav_l = pd.read_parquet(BASE / "nav_lgbm.parquet")
trades_r = pd.read_parquet(BASE / "trades_ridge.parquet")
trades_l = pd.read_parquet(BASE / "trades_lgbm.parquet")

for df in [nav_r, nav_l]:
    df["week_start"] = pd.to_datetime(df["week_start"])
for df in [trades_r, trades_l]:
    df["week_start"] = pd.to_datetime(df["week_start"])

print("=" * 90)
print("SECTION 1: BASIC SUMMARY COMPARISON")
print("=" * 90)

with open(BASE / "summary_ridge.json") as f:
    sr = json.load(f)
with open(BASE / "summary_lgbm.json") as f:
    sl = json.load(f)

metrics = ["start_nav", "end_nav", "return_pct", "trades",
           "avg_cash_ratio", "regime_gate_closed_weeks"]
print(f"{'Metric':<35} {'Ridge':>15} {'LGBM':>15} {'Diff':>15}")
print("-" * 80)
for m in metrics:
    rv, lv = sr.get(m, "N/A"), sl.get(m, "N/A")
    if isinstance(rv, (int, float)) and isinstance(lv, (int, float)):
        diff = lv - rv
        print(f"{m:<35} {rv:>15.4f} {lv:>15.4f} {diff:>+15.4f}")
    else:
        print(f"{m:<35} {str(rv):>15} {str(lv):>15}")

# ── SECTION 2: YEARLY RETURN BREAKDOWN ─────────────────────────────────────
print()
print("=" * 90)
print("SECTION 2: YEARLY RETURN BREAKDOWN")
print("=" * 90)

def yearly_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    nav_df = nav_df.sort_values("week_start").copy()
    nav_df["year"] = nav_df["week_start"].dt.year
    rows = []
    years = sorted(nav_df["year"].unique())
    for yr in years:
        yr_data = nav_df[nav_df["year"] == yr].sort_values("week_start")
        if len(yr_data) < 2:
            continue
        start_nav = yr_data.iloc[0]["nav"]
        end_nav = yr_data.iloc[-1]["nav"]
        ret = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0
        rows.append({"year": yr, "start_nav": start_nav, "end_nav": end_nav,
                      "return_pct": ret, "weeks": len(yr_data)})
    return pd.DataFrame(rows)

yr_r = yearly_returns(nav_r)
yr_l = yearly_returns(nav_l)
merged_yr = yr_r.merge(yr_l, on="year", suffixes=("_ridge", "_lgbm"))

print(f"{'Year':<6} {'Ridge %':>10} {'LGBM %':>10} {'Diff %':>10} {'Ridge NAV':>12} {'LGBM NAV':>12}")
print("-" * 70)
for _, row in merged_yr.iterrows():
    diff = row["return_pct_lgbm"] - row["return_pct_ridge"]
    print(f"{int(row['year']):<6} {row['return_pct_ridge']:>+10.2f} {row['return_pct_lgbm']:>+10.2f} "
          f"{diff:>+10.2f} {row['end_nav_ridge']:>12.2f} {row['end_nav_lgbm']:>12.2f}")

# ── SECTION 3: TRADE-LEVEL STATISTICS ──────────────────────────────────────
print()
print("=" * 90)
print("SECTION 3: TRADE-LEVEL STATISTICS")
print("=" * 90)

def trade_stats(trades_df: pd.DataFrame, label: str) -> dict:
    t = trades_df.copy()
    # Exclude KEEP trades (buy_cost=0 and sell_cost=0 might be keeps)
    all_trades = t
    real_trades = t[(t["buy_cost"] > 0) | (t["sell_cost"] > 0)]
    keep_trades = t[(t["buy_cost"] == 0) & (t["sell_cost"] == 0)]

    wins = all_trades[all_trades["pnl"] > 0]
    losses = all_trades[all_trades["pnl"] <= 0]

    stats = {
        "label": label,
        "total_records": len(all_trades),
        "real_trades": len(real_trades),
        "keep_holds": len(keep_trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / len(all_trades) * 100 if len(all_trades) > 0 else 0,
        "avg_pnl": float(all_trades["pnl"].mean()),
        "median_pnl": float(all_trades["pnl"].median()),
        "avg_win": float(wins["pnl"].mean()) if len(wins) > 0 else 0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) > 0 else 0,
        "max_win": float(all_trades["pnl"].max()),
        "max_loss": float(all_trades["pnl"].min()),
        "total_pnl": float(all_trades["pnl"].sum()),
        "total_buy_cost": float(all_trades["buy_cost"].sum()),
        "total_sell_cost": float(all_trades["sell_cost"].sum()),
        "avg_return": float(all_trades["return"].mean()) * 100,
        "median_return": float(all_trades["return"].median()) * 100,
        "std_return": float(all_trades["return"].std()) * 100,
        "unique_symbols": all_trades["symbol"].nunique(),
    }
    return stats

sr_stats = trade_stats(trades_r, "Ridge")
sl_stats = trade_stats(trades_l, "LGBM")

for key in sr_stats:
    if key == "label":
        continue
    rv, lv = sr_stats[key], sl_stats[key]
    if isinstance(rv, float):
        print(f"{key:<30} Ridge: {rv:>12.4f}  LGBM: {lv:>12.4f}  Diff: {lv - rv:>+12.4f}")
    else:
        print(f"{key:<30} Ridge: {rv:>12}  LGBM: {lv:>12}  Diff: {lv - rv:>+12}")

# ── SECTION 3b: TRADE STATS FOR REAL TRADES ONLY (excluding keeps) ─────────
print()
print("--- Real Trades Only (excluding KEEP holds) ---")
real_r = trades_r[(trades_r["buy_cost"] > 0) | (trades_r["sell_cost"] > 0)]
real_l = trades_l[(trades_l["buy_cost"] > 0) | (trades_l["sell_cost"] > 0)]

for label, df in [("Ridge", real_r), ("LGBM", real_l)]:
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    print(f"\n  {label} (real trades only): {len(df)} trades")
    print(f"    Win rate: {len(wins)/len(df)*100:.1f}%")
    print(f"    Avg PnL: ${df['pnl'].mean():.4f}")
    print(f"    Total PnL: ${df['pnl'].sum():.2f}")
    print(f"    Total cost: ${df['buy_cost'].sum() + df['sell_cost'].sum():.2f}")
    print(f"    Avg win: ${wins['pnl'].mean():.4f}" if len(wins) > 0 else "    Avg win: N/A")
    print(f"    Avg loss: ${losses['pnl'].mean():.4f}" if len(losses) > 0 else "    Avg loss: N/A")
    print(f"    Win/Loss ratio: {abs(wins['pnl'].mean()/losses['pnl'].mean()):.2f}" if len(wins) > 0 and len(losses) > 0 else "    Win/Loss ratio: N/A")

# ── SECTION 4: KEEP/HOLD ANALYSIS ──────────────────────────────────────────
print()
print("=" * 90)
print("SECTION 4: KEEP/HOLD ANALYSIS")
print("=" * 90)

keep_r = trades_r[(trades_r["buy_cost"] == 0) & (trades_r["sell_cost"] == 0)]
keep_l = trades_l[(trades_l["buy_cost"] == 0) & (trades_l["sell_cost"] == 0)]

print(f"{'Metric':<35} {'Ridge':>15} {'LGBM':>15}")
print("-" * 65)
print(f"{'Keep records':<35} {len(keep_r):>15} {len(keep_l):>15}")
print(f"{'Keep total PnL':<35} {keep_r['pnl'].sum():>15.2f} {keep_l['pnl'].sum():>15.2f}")
print(f"{'Keep avg PnL':<35} {keep_r['pnl'].mean():>15.4f} {keep_l['pnl'].mean():>15.4f}")
print(f"{'Keep win rate %':<35} {(keep_r['pnl']>0).mean()*100:>15.1f} {(keep_l['pnl']>0).mean()*100:>15.1f}")

# ── SECTION 5: YEARLY TRADE STATISTICS ─────────────────────────────────────
print()
print("=" * 90)
print("SECTION 5: YEARLY TRADE COUNTS & WIN RATES")
print("=" * 90)

def yearly_trade_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    trades_df = trades_df.copy()
    trades_df["year"] = trades_df["week_start"].dt.year
    rows = []
    for yr in sorted(trades_df["year"].unique()):
        yt = trades_df[trades_df["year"] == yr]
        wins = yt[yt["pnl"] > 0]
        rows.append({
            "year": yr,
            "trades": len(yt),
            "win_rate": len(wins)/len(yt)*100 if len(yt) > 0 else 0,
            "total_pnl": yt["pnl"].sum(),
            "avg_pnl": yt["pnl"].mean(),
            "unique_symbols": yt["symbol"].nunique(),
        })
    return pd.DataFrame(rows)

ytr = yearly_trade_stats(trades_r)
ytl = yearly_trade_stats(trades_l)
ytm = ytr.merge(ytl, on="year", suffixes=("_r", "_l"))

print(f"{'Year':<6} {'Trades_R':>9} {'Trades_L':>9} {'WR_R%':>8} {'WR_L%':>8} {'PnL_R':>10} {'PnL_L':>10} {'Sym_R':>7} {'Sym_L':>7}")
print("-" * 85)
for _, row in ytm.iterrows():
    print(f"{int(row['year']):<6} {int(row['trades_r']):>9} {int(row['trades_l']):>9} "
          f"{row['win_rate_r']:>8.1f} {row['win_rate_l']:>8.1f} "
          f"{row['total_pnl_r']:>+10.2f} {row['total_pnl_l']:>+10.2f} "
          f"{int(row['unique_symbols_r']):>7} {int(row['unique_symbols_l']):>7}")

# ── SECTION 6: DRAWDOWN ANALYSIS ──────────────────────────────────────────
print()
print("=" * 90)
print("SECTION 6: DRAWDOWN ANALYSIS")
print("=" * 90)

def drawdown_analysis(nav_df: pd.DataFrame, label: str):
    nav_df = nav_df.sort_values("week_start").copy()
    navs = nav_df["nav"].values.astype(float)
    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    max_dd_idx = np.argmin(dd)
    peak_idx = np.argmax(navs[:max_dd_idx+1]) if max_dd_idx > 0 else 0

    # Drawdown periods > 5%
    dd_periods = []
    in_dd = False
    start_i = 0
    for i, d in enumerate(dd):
        if d < -0.05 and not in_dd:
            in_dd = True
            start_i = i
        elif d >= 0 and in_dd:
            in_dd = False
            dd_periods.append((start_i, i, np.min(dd[start_i:i])))
    if in_dd:
        dd_periods.append((start_i, len(dd)-1, np.min(dd[start_i:])))

    print(f"\n  [{label}]")
    print(f"    Max Drawdown: {dd[max_dd_idx]*100:.2f}%")
    print(f"    Peak date: {nav_df.iloc[peak_idx]['week_start'].strftime('%Y-%m-%d')}, NAV: {navs[peak_idx]:.2f}")
    print(f"    Trough date: {nav_df.iloc[max_dd_idx]['week_start'].strftime('%Y-%m-%d')}, NAV: {navs[max_dd_idx]:.2f}")

    if max_dd_idx < len(dd) - 1:
        recovery = np.where(navs[max_dd_idx:] >= peaks[max_dd_idx])[0]
        if len(recovery) > 0:
            rec_idx = max_dd_idx + recovery[0]
            print(f"    Recovery date: {nav_df.iloc[rec_idx]['week_start'].strftime('%Y-%m-%d')}")
            print(f"    Drawdown duration: {rec_idx - peak_idx} weeks (peak to recovery)")
        else:
            print(f"    Recovery: NOT YET RECOVERED")

    print(f"    Drawdown periods > 5%: {len(dd_periods)}")
    for s, e, d in dd_periods:
        print(f"      {nav_df.iloc[s]['week_start'].strftime('%Y-%m-%d')} → "
              f"{nav_df.iloc[e]['week_start'].strftime('%Y-%m-%d')}: {d*100:.2f}%")

    # Avg drawdown
    avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0
    print(f"    Avg drawdown (when in DD): {avg_dd*100:.2f}%")
    print(f"    % of time in drawdown: {(dd < 0).mean()*100:.1f}%")

    return dd

dd_r = drawdown_analysis(nav_r, "Ridge")
dd_l = drawdown_analysis(nav_l, "LGBM")

# ── SECTION 7: RISK-ADJUSTED METRICS ──────────────────────────────────────
print()
print("=" * 90)
print("SECTION 7: RISK-ADJUSTED METRICS")
print("=" * 90)

def risk_metrics(nav_df: pd.DataFrame, label: str):
    nav_df = nav_df.sort_values("week_start").copy()
    navs = nav_df["nav"].values.astype(float)
    weekly_returns = np.diff(navs) / navs[:-1]
    weekly_returns = weekly_returns[np.isfinite(weekly_returns)]

    ann_return = np.mean(weekly_returns) * 52
    ann_vol = np.std(weekly_returns) * np.sqrt(52)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = weekly_returns[weekly_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(52) if len(downside) > 0 else 0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    max_dd = abs(dd.min())
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Profit factor
    pos_weeks = weekly_returns[weekly_returns > 0].sum()
    neg_weeks = abs(weekly_returns[weekly_returns < 0].sum())
    profit_factor = pos_weeks / neg_weeks if neg_weeks > 0 else float('inf')

    print(f"\n  [{label}]")
    print(f"    Annualized Return: {ann_return*100:.2f}%")
    print(f"    Annualized Volatility: {ann_vol*100:.2f}%")
    print(f"    Sharpe Ratio: {sharpe:.3f}")
    print(f"    Sortino Ratio: {sortino:.3f}")
    print(f"    Calmar Ratio: {calmar:.3f}")
    print(f"    Max Drawdown: {max_dd*100:.2f}%")
    print(f"    Profit Factor: {profit_factor:.3f}")
    print(f"    Weekly win rate: {(weekly_returns > 0).mean()*100:.1f}%")
    print(f"    Best week: {weekly_returns.max()*100:+.2f}%")
    print(f"    Worst week: {weekly_returns.min()*100:+.2f}%")
    print(f"    Avg +week: {weekly_returns[weekly_returns>0].mean()*100:+.3f}%")
    print(f"    Avg -week: {weekly_returns[weekly_returns<0].mean()*100:+.3f}%")

    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "ann_ret": ann_return, "ann_vol": ann_vol, "max_dd": max_dd,
            "profit_factor": profit_factor}

rm_r = risk_metrics(nav_r, "Ridge")
rm_l = risk_metrics(nav_l, "LGBM")

# ── SECTION 8: SYMBOL OVERLAP ANALYSIS ─────────────────────────────────────
print()
print("=" * 90)
print("SECTION 8: SYMBOL OVERLAP ANALYSIS")
print("=" * 90)

syms_r = set(trades_r["symbol"].unique())
syms_l = set(trades_l["symbol"].unique())
only_r = syms_r - syms_l
only_l = syms_l - syms_r
both = syms_r & syms_l

print(f"  Ridge-only symbols: {len(only_r)}")
print(f"  LGBM-only symbols: {len(only_l)}")
print(f"  Both: {len(both)}")
print(f"  Ridge total unique: {len(syms_r)}")
print(f"  LGBM total unique: {len(syms_l)}")

# Top symbols by total PnL
print("\n  Top 10 symbols by total PnL (Ridge):")
sym_pnl_r = trades_r.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
for sym, pnl in sym_pnl_r.head(10).items():
    cnt = len(trades_r[trades_r["symbol"] == sym])
    print(f"    {sym:<8} PnL: {pnl:>+8.2f}  trades: {cnt}")

print("\n  Top 10 symbols by total PnL (LGBM):")
sym_pnl_l = trades_l.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
for sym, pnl in sym_pnl_l.head(10).items():
    cnt = len(trades_l[trades_l["symbol"] == sym])
    print(f"    {sym:<8} PnL: {pnl:>+8.2f}  trades: {cnt}")

print("\n  Bottom 10 symbols by total PnL (Ridge):")
for sym, pnl in sym_pnl_r.tail(10).items():
    cnt = len(trades_r[trades_r["symbol"] == sym])
    print(f"    {sym:<8} PnL: {pnl:>+8.2f}  trades: {cnt}")

print("\n  Bottom 10 symbols by total PnL (LGBM):")
for sym, pnl in sym_pnl_l.tail(10).items():
    cnt = len(trades_l[trades_l["symbol"] == sym])
    print(f"    {sym:<8} PnL: {pnl:>+8.2f}  trades: {cnt}")

# ── SECTION 9: POSITION COUNT COMPARISON ───────────────────────────────────
print()
print("=" * 90)
print("SECTION 9: POSITION COUNT & CASH RATIO COMPARISON")
print("=" * 90)

print(f"{'Metric':<35} {'Ridge':>15} {'LGBM':>15}")
print("-" * 65)
print(f"{'Avg n_positions':<35} {nav_r['n_positions'].mean():>15.2f} {nav_l['n_positions'].mean():>15.2f}")
print(f"{'Max n_positions':<35} {nav_r['n_positions'].max():>15} {nav_l['n_positions'].max():>15}")
print(f"{'Min n_positions':<35} {nav_r['n_positions'].min():>15} {nav_l['n_positions'].min():>15}")
print(f"{'Avg cash_usd':<35} {nav_r['cash_usd'].mean():>15.2f} {nav_l['cash_usd'].mean():>15.2f}")
print(f"{'Avg positions_value':<35} {nav_r['positions_value'].mean():>15.2f} {nav_l['positions_value'].mean():>15.2f}")
print(f"{'Avg cash_ratio':<35} {(nav_r['cash_usd']/nav_r['nav']).mean():>15.4f} {(nav_l['cash_usd']/nav_l['nav']).mean():>15.4f}")

# ── SECTION 10: QUARTERLY RETURN COMPARISON ────────────────────────────────
print()
print("=" * 90)
print("SECTION 10: QUARTERLY RETURN COMPARISON")
print("=" * 90)

def quarterly_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    nav_df = nav_df.sort_values("week_start").copy()
    nav_df["quarter"] = nav_df["week_start"].dt.to_period("Q")
    rows = []
    for q in sorted(nav_df["quarter"].unique()):
        qd = nav_df[nav_df["quarter"] == q].sort_values("week_start")
        if len(qd) < 2:
            continue
        start_nav = qd.iloc[0]["nav"]
        end_nav = qd.iloc[-1]["nav"]
        ret = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0
        rows.append({"quarter": str(q), "return_pct": ret})
    return pd.DataFrame(rows)

qr = quarterly_returns(nav_r)
ql = quarterly_returns(nav_l)
qm = qr.merge(ql, on="quarter", suffixes=("_ridge", "_lgbm"))

print(f"{'Quarter':<10} {'Ridge %':>10} {'LGBM %':>10} {'Diff %':>10} {'Better':>10}")
print("-" * 55)
for _, row in qm.iterrows():
    diff = row["return_pct_lgbm"] - row["return_pct_ridge"]
    better = "LGBM" if diff > 0 else "Ridge" if diff < 0 else "Tie"
    print(f"{row['quarter']:<10} {row['return_pct_ridge']:>+10.2f} {row['return_pct_lgbm']:>+10.2f} "
          f"{diff:>+10.2f} {better:>10}")

# Count quarterly wins
r_wins = sum(1 for _, row in qm.iterrows() if row["return_pct_ridge"] > row["return_pct_lgbm"])
l_wins = sum(1 for _, row in qm.iterrows() if row["return_pct_lgbm"] > row["return_pct_ridge"])
print(f"\nQuarterly wins:  Ridge={r_wins}  LGBM={l_wins}  Total={len(qm)}")

# ── SECTION 11: CONSECUTIVE LOSS STREAKS ───────────────────────────────────
print()
print("=" * 90)
print("SECTION 11: WEEKLY PNL STREAK ANALYSIS")
print("=" * 90)

def streak_analysis(nav_df: pd.DataFrame, label: str):
    nav_df = nav_df.sort_values("week_start").copy()
    navs = nav_df["nav"].values.astype(float)
    wr = np.diff(navs) / navs[:-1]

    max_win_streak = 0
    max_loss_streak = 0
    cur_streak = 0
    for r in wr:
        if r > 0:
            if cur_streak > 0:
                cur_streak += 1
            else:
                cur_streak = 1
            max_win_streak = max(max_win_streak, cur_streak)
        elif r < 0:
            if cur_streak < 0:
                cur_streak -= 1
            else:
                cur_streak = -1
            max_loss_streak = max(max_loss_streak, abs(cur_streak))
        else:
            cur_streak = 0

    print(f"  [{label}]")
    print(f"    Max winning streak: {max_win_streak} weeks")
    print(f"    Max losing streak: {max_loss_streak} weeks")

streak_analysis(nav_r, "Ridge")
streak_analysis(nav_l, "LGBM")

print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
