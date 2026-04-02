"""Sub-period stability + regime gate interaction check.

(3) Sub-period: saved nav files を年別スライスして比較。
(4) Regime gate: dataclasses.replace で config 書き換えなし。
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("C:/BOT/MLStock")
BACKTEST_DIR = BASE_DIR / "artifacts" / "backtest"
SWEEP_DIR = BACKTEST_DIR / "sweep_v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_nav(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    return df


def yearly_returns(nav_df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Return {year: (return_pct, max_dd)} sliced from full nav."""
    results = {}
    nav_s = pd.to_numeric(nav_df["nav"], errors="coerce")
    for yr in sorted(nav_df["week_start"].apply(lambda x: x.year).unique()):
        mask = nav_df["week_start"].apply(lambda x: x.year) == yr
        sub = nav_s[mask].dropna()
        if len(sub) < 2:
            results[yr] = (0.0, 0.0)
            continue
        ret = float(sub.iloc[-1] / sub.iloc[0] - 1)
        dd = float((sub / sub.cummax() - 1).min())
        results[yr] = (ret, dd)
    return results


def run_full_backtest(cfg, quiet: bool = True) -> tuple[float, float, pd.DataFrame]:
    """Run full period backtest, return (return_pct, max_dd, nav_df)."""
    from mlstock.jobs import backtest

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = backtest.run(cfg)
    else:
        summary = backtest.run(cfg)

    nav_df = pd.read_parquet(BACKTEST_DIR / "nav.parquet")
    nav_s = pd.to_numeric(nav_df["nav"], errors="coerce").dropna()
    if len(nav_s) < 2:
        return 0.0, 0.0, nav_df
    dd = float((nav_s / nav_s.cummax() - 1).min())
    return float(summary.get("return_pct", 0.0)), dd, nav_df


# ---------------------------------------------------------------------------
# (3) Sub-period stability from saved nav files
# ---------------------------------------------------------------------------


def subperiod_stability():
    print("=" * 70)
    print("(3) SUB-PERIOD STABILITY (year-by-year from saved nav files)")
    print("=" * 70)
    print()
    print("Note: actual backtest window = 2020-07-27 to 2024-12-30")
    print("      (snapshots have no data before 2020)")
    print()

    # Load saved navs
    nav_e70 = load_nav(BACKTEST_DIR / "nav_ensemble_70_old.parquet")
    nav_f17 = load_nav(SWEEP_DIR / "nav_baseline_feat17.parquet")
    nav_v2 = load_nav(SWEEP_DIR / "nav_varN_t0005_p60.parquet")

    yrs_e70 = yearly_returns(nav_e70)
    yrs_f17 = yearly_returns(nav_f17)
    yrs_v2 = yearly_returns(nav_v2)

    all_years = sorted(set(yrs_e70) | set(yrs_f17) | set(yrs_v2))

    # ---- Return comparison ----
    print(f"{'Year':<6} {'E70(old)':>10} {'feat17':>10} {'V2':>10} " f"| {'f17-E70':>9} {'V2-E70':>9}")
    print("-" * 68)

    results = {}
    for yr in all_years:
        r_e70, dd_e70 = yrs_e70.get(yr, (0.0, 0.0))
        r_f17, dd_f17 = yrs_f17.get(yr, (0.0, 0.0))
        r_v2, dd_v2 = yrs_v2.get(yr, (0.0, 0.0))
        d_f17 = (r_f17 - r_e70) * 100
        d_v2 = (r_v2 - r_e70) * 100
        results[yr] = dict(e70=r_e70, f17=r_f17, v2=r_v2, dd_e70=dd_e70, dd_f17=dd_f17, dd_v2=dd_v2)
        print(f"{yr:<6} {r_e70*100:>9.2f}% {r_f17*100:>9.2f}% {r_v2*100:>9.2f}% " f"| {d_f17:>+8.2f}pt {d_v2:>+8.2f}pt")

    print()

    # ---- Aggregate stats ----
    full_years = [yr for yr in all_years if yr not in (2020,)]  # 2020 is partial
    rets_e70 = [results[yr]["e70"] for yr in full_years]
    rets_f17 = [results[yr]["f17"] for yr in full_years]
    rets_v2 = [results[yr]["v2"] for yr in full_years]
    wins_f17 = sum(1 for yr in full_years if results[yr]["f17"] > results[yr]["e70"])
    wins_v2 = sum(1 for yr in full_years if results[yr]["v2"] > results[yr]["e70"])
    n = len(full_years)

    print(f"Full years only (2021-2024, n={n}):")
    print(f"{'':6} {'mean':>10} {'std':>8} {'wins':>8}")
    print("-" * 38)
    print(f"{'E70':<6} {np.mean(rets_e70)*100:>9.2f}% {np.std(rets_e70)*100:>7.2f}%")
    print(f"{'feat17':<6} {np.mean(rets_f17)*100:>9.2f}% {np.std(rets_f17)*100:>7.2f}% " f"{wins_f17:>4}/{n}")
    print(f"{'V2':<6} {np.mean(rets_v2)*100:>9.2f}% {np.std(rets_v2)*100:>7.2f}% " f"{wins_v2:>4}/{n}")

    print()
    if wins_f17 >= n - 1:
        print(f"[OK] feat17 beats E70 in {wins_f17}/{n} full years -- ROBUST")
    elif wins_f17 >= n // 2 + 1:
        print(f"[MARGINAL] feat17 beats E70 in {wins_f17}/{n} full years")
    else:
        print(f"[WARN] feat17 beats E70 in only {wins_f17}/{n} full years" " -- improvement may be concentrated")

    if wins_v2 >= n - 1:
        print(f"[OK] V2 beats E70 in {wins_v2}/{n} full years -- ROBUST")
    elif wins_v2 >= n // 2 + 1:
        print(f"[MARGINAL] V2 beats E70 in {wins_v2}/{n} full years")
    else:
        print(f"[WARN] V2 beats E70 in only {wins_v2}/{n} full years" " -- DD benefit may dominate")

    # ---- Drawdown comparison ----
    print()
    print("Max Drawdown by year:")
    print(f"{'Year':<6} {'E70':>9} {'feat17':>9} {'V2':>9} " f"| {'f17-E70':>9} {'V2-E70':>9}")
    print("-" * 68)
    for yr in all_years:
        r = results[yr]
        print(
            f"{yr:<6} {r['dd_e70']*100:>8.2f}% {r['dd_f17']*100:>8.2f}% "
            f"{r['dd_v2']*100:>8.2f}% | "
            f"{(r['dd_f17']-r['dd_e70'])*100:>+8.2f}pt "
            f"{(r['dd_v2']-r['dd_e70'])*100:>+8.2f}pt"
        )

    # ---- Weekly relative performance histogram ----
    print()
    print("Weekly relative return (feat17 - E70) distribution:")
    _print_weekly_diff_dist(nav_e70, nav_f17, "feat17 - E70")
    print()
    print("Weekly relative return (V2 - E70) distribution:")
    _print_weekly_diff_dist(nav_e70, nav_v2, "V2 - E70")

    return results


def _print_weekly_diff_dist(nav_a: pd.DataFrame, nav_b: pd.DataFrame, label: str):
    """Show text-based distribution of weekly log-return differences."""

    def weekly_log_ret(df):
        s = pd.to_numeric(df.set_index("week_start")["nav"], errors="coerce").dropna()
        return np.log(s / s.shift(1)).dropna()

    lr_a = weekly_log_ret(nav_a)
    lr_b = weekly_log_ret(nav_b)
    common = lr_a.index.intersection(lr_b.index)
    diff = (lr_b[common] - lr_a[common]) * 100  # pct

    pos = (diff > 0).sum()
    neg = (diff <= 0).sum()
    total = len(diff)
    print(f"  n={total}  positive={pos}({pos/total*100:.0f}%)  " f"negative={neg}({neg/total*100:.0f}%)")
    print(f"  mean={diff.mean():+.3f}%  std={diff.std():.3f}%  " f"median={diff.median():+.3f}%")
    print(f"  min={diff.min():+.3f}%  max={diff.max():+.3f}%")


# ---------------------------------------------------------------------------
# (4) Regime gate interaction
# ---------------------------------------------------------------------------


def regime_gate_check():
    print()
    print("=" * 70)
    print("(4) REGIME GATE INTERACTION")
    print("    Does adding SPY features reduce the gate's value?")
    print("=" * 70)

    from mlstock.config.loader import load_config

    cfg = load_config()
    gate_on = replace(cfg.risk.regime_gate, enabled=True)
    gate_off = replace(cfg.risk.regime_gate, enabled=False)

    cfgs = {
        "feat17 + gate ON": replace(
            cfg, selection=replace(cfg.selection, confidence_sizing=False), risk=replace(cfg.risk, regime_gate=gate_on)
        ),
        "feat17 + gate OFF": replace(
            cfg, selection=replace(cfg.selection, confidence_sizing=False), risk=replace(cfg.risk, regime_gate=gate_off)
        ),
        "V2     + gate ON": replace(cfg, risk=replace(cfg.risk, regime_gate=gate_on)),
        "V2     + gate OFF": replace(cfg, risk=replace(cfg.risk, regime_gate=gate_off)),
    }

    print(f"\n{'Config':<24} {'Return%':>9} {'MaxDD%':>9} {'Ret/|DD|':>9}")
    print("-" * 55)
    gate_results = {}
    for name, c in cfgs.items():
        sys.stdout.write(f"  Running {name} ...\r")
        sys.stdout.flush()
        ret, dd, _ = run_full_backtest(c, quiet=True)
        ratio = abs(ret / dd) if dd != 0 else 0.0
        gate_results[name] = (ret, dd, ratio)
        print(f"{name:<24} {ret*100:>8.2f}% {dd*100:>8.2f}% {ratio:>9.2f}")

    print()
    # Interpret: gate effect with feat17 vs V2
    r_f17_on, dd_f17_on, _ = gate_results["feat17 + gate ON"]
    r_f17_off, dd_f17_off, _ = gate_results["feat17 + gate OFF"]
    r_v2_on, dd_v2_on, _ = gate_results["V2     + gate ON"]
    r_v2_off, dd_v2_off, _ = gate_results["V2     + gate OFF"]

    gate_effect_f17 = (r_f17_on - r_f17_off) * 100  # positive = gate helps
    gate_effect_v2 = (r_v2_on - r_v2_off) * 100
    dd_effect_f17 = (dd_f17_on - dd_f17_off) * 100  # negative = gate reduces DD
    dd_effect_v2 = (dd_v2_on - dd_v2_off) * 100

    print(f"Gate effect on RETURN: feat17 {gate_effect_f17:+.2f}pt  V2 {gate_effect_v2:+.2f}pt")
    print(f"Gate effect on MaxDD:  feat17 {dd_effect_f17:+.2f}pt  V2 {dd_effect_v2:+.2f}pt")
    print()
    if abs(gate_effect_v2) < abs(gate_effect_f17) * 0.6:
        print("[FINDING] Gate effect SHRANK with V2 features -- SPY features partially")
        print("          absorb market regime information. Gate is less important.")
    elif abs(gate_effect_v2) > abs(gate_effect_f17) * 1.4:
        print("[FINDING] Gate effect GREW with V2 features -- complementary signals.")
    else:
        print("[FINDING] Gate effect SIMILAR with or without V2 features -- independent.")

    return gate_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  MLStock V2 Deep Analysis")
    print("=" * 70)

    subperiod_stability()
    regime_gate_check()

    print()
    print("=== Analysis Complete ===")
