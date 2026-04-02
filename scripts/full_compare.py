"""Full comparison: Ridge / LGBM / E_50 / E_70."""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(r"C:\BOT\MLStock\artifacts\backtest")

MODELS = {
    "Ridge": ("nav_ridge.parquet", "trades_ridge.parquet", "summary_ridge.json"),
    "LGBM": ("nav_lgbm.parquet", "trades_lgbm.parquet", "summary_lgbm.json"),
    "E_50": ("nav_ensemble_50.parquet", "trades_ensemble_50.parquet", "summary_ensemble_50.json"),
    "E_70": ("nav_ensemble_70.parquet", "trades_ensemble_70.parquet", "summary_ensemble_70.json"),
}

data = {}
for label, (nf, tf, sf) in MODELS.items():
    if not (BASE / nf).exists():
        continue
    nav = pd.read_parquet(BASE / nf)
    nav["week_start"] = pd.to_datetime(nav["week_start"])
    trades = pd.read_parquet(BASE / tf)
    trades["week_start"] = pd.to_datetime(trades["week_start"])
    with open(BASE / sf) as f:
        summary = json.load(f)
    data[label] = {"nav": nav.sort_values("week_start"), "trades": trades, "summary": summary}

LABELS = list(data.keys())
W = 14


def fmt(v, spec):
    if isinstance(v, (int, float)):
        return format(v, spec).rjust(W)
    return str(v).rjust(W)


def hdr(title):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def sep():
    print("-" * 100)


def trow(name, getter, spec, name_w=30):
    vals = [getter(l) for l in LABELS]
    line = name.ljust(name_w) + "".join(fmt(v, spec) for v in vals)
    print(line)


summ = lambda l, k: data[l]["summary"].get(k, 0)

# ============================================================
# SECTION 0
# ============================================================
hdr("MODELS")
print("  Ridge  : Ridge regression (L2, alpha=1.0)")
print("  LGBM   : LightGBM (n_estimators=200, max_depth=4)  <- PREV comparison")
print("  E_50   : Ensemble Ridge*0.5 + LGBM*0.5 (lowest overfit risk)")
print("  E_70   : Ensemble Ridge*0.7 + LGBM*0.3 (best backtest, current config)  <- NOW")

# ============================================================
# SECTION 1: Overall summary
# ============================================================
hdr("SECTION 1: FULL PERIOD SUMMARY (2018-2024)")
col_hdr = "Metric".ljust(30) + "".join(l.rjust(W) for l in LABELS)
print(col_hdr)
sep()

trow("End NAV ($)", lambda l: summ(l, "end_nav"), ".2f")
trow("Total Return (%)", lambda l: summ(l, "return_pct") * 100, "+.2f")
trow("Trades", lambda l: summ(l, "trades"), ".0f")
trow("Avg Cash Ratio", lambda l: summ(l, "avg_cash_ratio"), ".4f")
trow("Min Cash ($)", lambda l: summ(l, "min_cash_usd"), ".2f")
trow("Reserve Violations", lambda l: summ(l, "reserve_violation_count"), ".0f")

lgbm_nav = summ("LGBM", "end_nav")
print()
for l in LABELS:
    diff = summ(l, "end_nav") - lgbm_nav
    print(f"  vs LGBM  {l}: {diff:>+.2f} ({diff/lgbm_nav*100:>+.2f}%)")

# ============================================================
# SECTION 2: PnL breakdown
# ============================================================
hdr("SECTION 2: PnL SOURCE BREAKDOWN  *** KEY SECTION ***")
col_hdr = " ".ljust(30) + "".join(l.rjust(W) for l in LABELS)
print(col_hdr)
sep()

stats = {}
for l in LABELS:
    t = data[l]["trades"]
    real = t[(t["buy_cost"] > 0) | (t["sell_cost"] > 0)]
    keep = t[(t["buy_cost"] == 0) & (t["sell_cost"] == 0)]
    wr = real[real["pnl"] > 0]
    wl = real[real["pnl"] <= 0]
    wk = keep[keep["pnl"] > 0]
    stats[l] = {
        "real_cnt": len(real),
        "keep_cnt": len(keep),
        "real_pnl": real["pnl"].sum(),
        "keep_pnl": keep["pnl"].sum(),
        "real_wr": len(wr) / len(real) * 100 if len(real) > 0 else 0,
        "keep_wr": len(wk) / len(keep) * 100 if len(keep) > 0 else 0,
        "avg_win": wr["pnl"].mean() if len(wr) > 0 else 0,
        "avg_loss": wl["pnl"].mean() if len(wl) > 0 else 0,
        "wl_ratio": abs(wr["pnl"].mean() / wl["pnl"].mean()) if len(wr) > 0 and len(wl) > 0 else 0,
        "total_cost": real["buy_cost"].sum() + real["sell_cost"].sum(),
    }

rows2 = [
    ("Real trades (buy/sell)", "real_cnt", ".0f"),
    ("KEEP holds", "keep_cnt", ".0f"),
    ("Real trade PnL ($)", "real_pnl", "+.2f"),
    ("KEEP hold PnL ($)", "keep_pnl", "+.2f"),
    ("Real trade win-rate (%)", "real_wr", ".1f"),
    ("KEEP hold win-rate (%)", "keep_wr", ".1f"),
    ("Avg win ($)", "avg_win", "+.4f"),
    ("Avg loss ($)", "avg_loss", "+.4f"),
    ("Win/Loss ratio", "wl_ratio", ".3f"),
    ("Total trade cost ($)", "total_cost", ".2f"),
]
for name, key, spec in rows2:
    line = name.ljust(30) + "".join(fmt(stats[l][key], spec) for l in LABELS)
    print(line)

print()
print("  NOTE: KEEP PnL = unrealized PnL of deadband-held positions (zero transaction cost)")
print("        High KEEP PnL => stable rankings => same stocks held for multiple weeks")
print("        LGBM KEEP PnL is negative => noisy rankings => held stocks then fall")

# ============================================================
# SECTION 3: Yearly returns
# ============================================================
hdr("SECTION 3: YEARLY RETURNS (%)")
col_hdr = "Year".ljust(8) + "".join(l.rjust(W) for l in LABELS) + "Best".rjust(W)
print(col_hdr)
sep()


def yr_rets(nav_df):
    nav_df = nav_df.copy()
    nav_df["y"] = nav_df["week_start"].dt.year
    d = {}
    for yr in sorted(nav_df["y"].unique()):
        yd = nav_df[nav_df["y"] == yr].sort_values("week_start")
        if len(yd) >= 2:
            d[yr] = (yd.iloc[-1]["nav"] / yd.iloc[0]["nav"] - 1) * 100
    return d


yr_data = {l: yr_rets(data[l]["nav"]) for l in LABELS}
all_years = sorted(set().union(*(yr_data[l].keys() for l in LABELS)))
yr_wins = {l: 0 for l in LABELS}

for yr in all_years:
    vals = {l: yr_data[l].get(yr) for l in LABELS}
    filt = {l: v for l, v in vals.items() if v is not None}
    best = max(filt, key=lambda k: filt[k]) if filt else ""
    if best:
        yr_wins[best] += 1
    line = str(yr).ljust(8)
    for l in LABELS:
        v = vals[l]
        line += fmt(v, "+.2f") if v is not None else "N/A".rjust(W)
    line += best.rjust(W)
    print(line)

sep()
print("Yearly wins".ljust(8) + "".join(str(yr_wins[l]).rjust(W) for l in LABELS))

# ============================================================
# SECTION 4: Quarterly returns
# ============================================================
hdr("SECTION 4: QUARTERLY RETURNS (%)")
col_hdr = "Quarter".ljust(10) + "".join(l.rjust(W) for l in LABELS) + "Best".rjust(W)
print(col_hdr)
sep()


def qr_rets(nav_df):
    nav_df = nav_df.copy()
    nav_df["q"] = nav_df["week_start"].dt.to_period("Q")
    d = {}
    for q in sorted(nav_df["q"].unique()):
        qd = nav_df[nav_df["q"] == q].sort_values("week_start")
        if len(qd) >= 2:
            d[str(q)] = (qd.iloc[-1]["nav"] / qd.iloc[0]["nav"] - 1) * 100
    return d


qr_data = {l: qr_rets(data[l]["nav"]) for l in LABELS}
all_q = sorted(set().union(*(qr_data[l].keys() for l in LABELS)))
q_wins = {l: 0 for l in LABELS}

for q in all_q:
    vals = {l: qr_data[l].get(q) for l in LABELS}
    filt = {l: v for l, v in vals.items() if v is not None and abs(v) > 0.001}
    best = max(filt, key=lambda k: filt[k]) if filt else ""
    if best:
        q_wins[best] += 1
    line = q.ljust(10)
    for l in LABELS:
        v = vals[l]
        line += fmt(v, "+.2f") if v is not None else "N/A".rjust(W)
    line += best.rjust(W)
    print(line)

sep()
print("Q-wins".ljust(10) + "".join(str(q_wins[l]).rjust(W) for l in LABELS))

# ============================================================
# SECTION 5: Risk metrics
# ============================================================
hdr("SECTION 5: RISK-ADJUSTED METRICS")
col_hdr = "Metric".ljust(30) + "".join(l.rjust(W) for l in LABELS)
print(col_hdr)
sep()


def calc_risk(nav_df):
    navs = nav_df.sort_values("week_start")["nav"].values.astype(float)
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
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    pos = wr[wr > 0].sum()
    neg = abs(wr[wr < 0].sum())
    pf = pos / neg if neg > 0 else 9999
    trough_i = np.argmin(dd)
    peak_i = np.argmax(navs[: trough_i + 1]) if trough_i > 0 else 0
    rec_arr = np.where(navs[trough_i:] >= peaks[trough_i])[0]
    rec_wk = int(trough_i + rec_arr[0] - peak_i) if len(rec_arr) > 0 else None
    # streaks
    win_s = loss_s = cur = 0
    for r in wr:
        if r > 0:
            cur = cur + 1 if cur >= 0 else 1
        elif r < 0:
            cur = cur - 1 if cur <= 0 else -1
        else:
            cur = 0
        win_s = max(win_s, cur if cur > 0 else 0)
        loss_s = max(loss_s, -cur if cur < 0 else 0)
    return {
        "ann_ret": ann_ret * 100,
        "ann_vol": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd * 100,
        "profit_factor": pf,
        "weekly_wr": (wr > 0).mean() * 100,
        "best_wk": wr.max() * 100,
        "worst_wk": wr.min() * 100,
        "avg_up": wr[wr > 0].mean() * 100,
        "avg_dn": wr[wr < 0].mean() * 100,
        "pct_in_dd": (dd < 0).mean() * 100,
        "avg_dd": dd[dd < 0].mean() * 100 if (dd < 0).any() else 0,
        "rec_wk": rec_wk,
        "win_streak": win_s,
        "loss_streak": loss_s,
    }


risk = {l: calc_risk(data[l]["nav"]) for l in LABELS}

risk_rows = [
    ("Annualized Return (%)", "ann_ret", "+.2f"),
    ("Annualized Volatility (%)", "ann_vol", ".2f"),
    ("Sharpe Ratio", "sharpe", ".3f"),
    ("Sortino Ratio", "sortino", ".3f"),
    ("Calmar Ratio", "calmar", ".3f"),
    ("Profit Factor", "profit_factor", ".3f"),
    ("Max Drawdown (%)", "max_dd", ".2f"),
    ("Avg DD when in DD (%)", "avg_dd", ".2f"),
    ("% Time in Drawdown", "pct_in_dd", ".1f"),
    ("DD Recovery (weeks)", "rec_wk", ""),
    ("Weekly Win Rate (%)", "weekly_wr", ".1f"),
    ("Best Week (%)", "best_wk", "+.2f"),
    ("Worst Week (%)", "worst_wk", "+.2f"),
    ("Avg Up Week (%)", "avg_up", "+.3f"),
    ("Avg Down Week (%)", "avg_dn", "+.3f"),
    ("Max Win Streak (wks)", "win_streak", ""),
    ("Max Loss Streak (wks)", "loss_streak", ""),
]

for name, key, spec in risk_rows:
    line = name.ljust(30)
    for l in LABELS:
        v = risk[l].get(key)
        if v is None:
            line += "N/R".rjust(W)
        elif spec:
            line += fmt(v, spec)
        else:
            line += str(v).rjust(W)
    print(line)

# ============================================================
# SECTION 6: Drawdown detail
# ============================================================
hdr("SECTION 6: DRAWDOWN DETAIL")

for l in LABELS:
    nav_df = data[l]["nav"].sort_values("week_start")
    navs = nav_df["nav"].values.astype(float)
    dates = nav_df["week_start"].values
    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    trough_i = np.argmin(dd)
    peak_i = np.argmax(navs[: trough_i + 1]) if trough_i > 0 else 0
    rec_arr = np.where(navs[trough_i:] >= peaks[trough_i])[0]
    rec_str = pd.Timestamp(dates[trough_i + rec_arr[0]]).strftime("%Y-%m-%d") if len(rec_arr) > 0 else "NOT RECOVERED"

    dd_periods = []
    in_dd = False
    start_i = 0
    for i, d in enumerate(dd):
        if d < -0.05 and not in_dd:
            in_dd = True
            start_i = i
        elif d >= -0.01 and in_dd:
            in_dd = False
            dd_periods.append((start_i, i, dd[start_i:i].min()))
    if in_dd:
        dd_periods.append((start_i, len(dd) - 1, dd[start_i:].min()))

    print(f"\n  [{l}]")
    print(f"    Max DD:       {dd[trough_i]*100:.2f}%")
    print(f"    Peak:         {pd.Timestamp(dates[peak_i]).strftime('%Y-%m-%d')}  NAV={navs[peak_i]:.2f}")
    print(f"    Trough:       {pd.Timestamp(dates[trough_i]).strftime('%Y-%m-%d')}  NAV={navs[trough_i]:.2f}")
    print(f"    Recovery:     {rec_str}")
    print(f"    DD>5% periods: {len(dd_periods)}")
    for s, e, d in dd_periods:
        print(
            f"      {pd.Timestamp(dates[s]).strftime('%Y-%m-%d')} -> "
            f"{pd.Timestamp(dates[e]).strftime('%Y-%m-%d')}: {d*100:.2f}%"
        )

# ============================================================
# SECTION 7: Symbol analysis
# ============================================================
hdr("SECTION 7: TOP/BOTTOM SYMBOL PnL")

for l in LABELS:
    t = data[l]["trades"]
    sp = t.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    print(f"\n  [{l}]  Unique symbols: {t['symbol'].nunique()}")
    print("    Top5: " + "  ".join(f"{s}({v:+.1f})" for s, v in sp.head(5).items()))
    print("    Bot5: " + "  ".join(f"{s}({v:+.1f})" for s, v in sp.tail(5).items()))

    # Symbol overlap with other models
    syms = set(t["symbol"])
    for l2 in LABELS:
        if l2 == l:
            continue
        s2 = set(data[l2]["trades"]["symbol"])
        only_this = len(syms - s2)
        only_other = len(s2 - syms)
        both = len(syms & s2)
        print(f"    vs {l2}: only-{l}={only_this}  only-{l2}={only_other}  both={both}")

# ============================================================
# SECTION 8: PREV(LGBM) vs NOW(E_70) delta summary
# ============================================================
hdr("SECTION 8: PREV(LGBM) vs NOW(E_70) - DELTA SUMMARY")

if "LGBM" in data and "E_70" in data:
    lv_s = data["LGBM"]["summary"]
    ev_s = data["E_70"]["summary"]
    lr = risk["LGBM"]
    er = risk["E_70"]
    ls = stats["LGBM"]
    es = stats["E_70"]

    comparisons = [
        ("End NAV ($)", lv_s["end_nav"], ev_s["end_nav"], True, ".2f"),
        ("Total Return (%)", lv_s["return_pct"] * 100, ev_s["return_pct"] * 100, True, "+.2f"),
        ("Sharpe Ratio", lr["sharpe"], er["sharpe"], True, ".3f"),
        ("Sortino Ratio", lr["sortino"], er["sortino"], True, ".3f"),
        ("Calmar Ratio", lr["calmar"], er["calmar"], True, ".3f"),
        ("Max DD (%)", lr["max_dd"], er["max_dd"], False, ".2f"),
        ("Annualized Vol (%)", lr["ann_vol"], er["ann_vol"], False, ".2f"),
        ("Profit Factor", lr["profit_factor"], er["profit_factor"], True, ".3f"),
        ("Worst Week (%)", lr["worst_wk"], er["worst_wk"], False, ".2f"),
        ("Weekly Win Rate (%)", lr["weekly_wr"], er["weekly_wr"], True, ".1f"),
        ("Real trade PnL ($)", ls["real_pnl"], es["real_pnl"], True, "+.2f"),
        ("KEEP hold PnL ($)", ls["keep_pnl"], es["keep_pnl"], True, "+.2f"),
        ("Trade cost ($)", ls["total_cost"], es["total_cost"], False, ".2f"),
        ("Trade count", lv_s["trades"], ev_s["trades"], False, ".0f"),
    ]

    print(f"\n  {'Metric':<28} {'LGBM (prev)':>14} {'E_70 (now)':>14} {'Delta':>12} {'Judge':>10}")
    sep()
    for name, lv, ev, hib, spec in comparisons:
        diff = ev - lv
        if hib:
            judge = "[BETTER]" if diff > 0.001 else ("[~SAME]" if abs(diff) < 0.001 else "[WORSE]")
        else:
            judge = "[BETTER]" if diff < -0.001 else ("[~SAME]" if abs(diff) < 0.001 else "[WORSE]")
        print(f"  {name:<28} {format(lv, spec):>14} {format(ev, spec):>14} {diff:>+12.3f} {judge:>10}")

print()
hdr("OVERFITTING CAVEAT")
print("  E_70 weight was selected by sweeping 0.3-0.95 on THE SAME backtest period.")
print("  Sub-period analysis shows E_70 beats Ridge in only 3 of 6 sub-periods.")
print("  E_50 has LOWER overfit risk (avg sub-period rank: 4.0 vs 5.3 for E_70)")
print()
print("  RECOMMENDATION:")
print("    Conservative (robust): ensemble_weight_ridge: 0.5  -> return +23.96%, Sharpe 0.470, DD -17.14%")
print("    Aggressive (backtest): ensemble_weight_ridge: 0.7  -> return +27.18%, Sharpe 0.527, DD -18.50%")
print()
print("=" * 100)
print("  ANALYSIS COMPLETE")
print("=" * 100)
