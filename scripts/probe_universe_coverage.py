"""
universe_coverage_probe.py

seed universe 2000銘柄に対して
  - 銘柄タイプ分類 (common / ETF / warrant / preferred / right / unit / unknown)
  - Alpaca IEX 1-min bar coverage (--probe-date で指定日を実測)
を集計し、CSV と要約テーブルを出力する。

Usage:
  # 静的分析のみ（API不要）
  python scripts/probe_universe_coverage.py --static-only

  # 指定日の IEX 1-min coverage を実測（当日または過去の市場日）
  python scripts/probe_universe_coverage.py --probe-date 2026-04-17

  # 出力先を変更
  python scripts/probe_universe_coverage.py --probe-date 2026-04-17 --out-dir artifacts/coverage
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlstock.data.alpaca.client import AlpacaClient


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe IEX coverage of seed universe")
    p.add_argument("--probe-date", type=str, default=None,
                   help="Market date (YYYY-MM-DD) to query 9:30-9:35 ET 1-min bars. "
                        "If omitted, only static analysis is run.")
    p.add_argument("--static-only", action="store_true",
                   help="Skip API probe and only run static classification.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "coverage",
                   help="Output directory (default: artifacts/coverage)")
    p.add_argument("--seed-path", type=Path,
                   default=ROOT / "data" / "reference" / "seed_symbols.parquet")
    p.add_argument("--assets-path", type=Path,
                   default=ROOT / "data" / "reference" / "assets.parquet")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--feed", type=str, default="iex",
                   help="Alpaca data feed (iex or sip)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Static classification
# ---------------------------------------------------------------------------

def _parse_asset_raw(raw: Any) -> Tuple[str, str, List[str]]:
    """(asset_class, full_name, attributes)"""
    try:
        d = json.loads(raw)
        return str(d.get("class", "")), str(d.get("name", "")), d.get("attributes", [])
    except Exception:
        return "", "", []


def _classify_symbol(sym: str, name: str) -> str:
    """
    Returns: common | etf | warrant | preferred | right | unit | unknown
    Priority: name-based annotation > symbol suffix.
    """
    s = sym.upper()
    n = str(name).lower()

    # warrant: symbol ends W / WS / .WS, or name contains 'warrant'
    if s.endswith("WS") or ".WS" in s or s.endswith("W"):
        return "warrant"
    if "warrant" in n:
        return "warrant"

    # unit: .U suffix or name says 'unit'
    if ".U" in s or (s.endswith("U") and len(s) > 2 and s[-2].isdigit()):
        return "unit"
    if "unit " in n and "united" not in n and "unity" not in n:
        return "unit"

    # preferred: .PRA/.PRB/... or name contains 'preferred'/'pfd'
    if re.search(r"(\.PR[A-Z]?|PRA$|PRB$|PRC$|PRD$|PRE$|PRF$|PRG$|PRH$)", s):
        return "preferred"
    if "preferred" in n or " pfd " in n or n.endswith(" pfd"):
        return "preferred"

    # right: symbol ends R/RT + name has 'right', or standalone name check
    if (s.endswith("R") or s.endswith("RT")) and "right" in n:
        return "right"
    if "right" in n and "rightmove" not in n and "upright" not in n and "copyright" not in n:
        return "right"

    # ETF: name keywords
    if "etf" in n or " fund" in n and "fundamental" not in n:
        return "etf"
    if any(k in n for k in ("ishares", "spdr", "invesco", "vanguard", "xtrackers")):
        return "etf"

    # unknown: not in assets table
    if not name or name == "nan":
        return "unknown"

    return "common"


def build_static_df(seed_path: Path, assets_path: Path) -> pd.DataFrame:
    """Return per-symbol DataFrame with classification columns."""
    seed_df = pd.read_parquet(seed_path)
    assets_df = pd.read_parquet(assets_path)

    parsed = assets_df["raw_json"].apply(lambda r: pd.Series(_parse_asset_raw(r)))
    parsed.columns = ["asset_class", "full_name", "attrs"]
    assets_df = pd.concat([assets_df[["symbol", "exchange", "tradable", "shortable"]], parsed], axis=1)

    merged = seed_df[["symbol", "seed_rank"]].merge(
        assets_df[["symbol", "exchange", "tradable", "shortable", "full_name", "attrs"]],
        on="symbol", how="left",
    )
    merged["full_name"] = merged["full_name"].fillna("")
    merged["exchange"] = merged["exchange"].fillna("UNKNOWN")
    merged["tradable"] = merged["tradable"].fillna(False)
    merged["shortable"] = merged["shortable"].fillna(False)
    merged["has_options"] = merged["attrs"].apply(
        lambda a: "has_options" in a if isinstance(a, list) else False
    )
    merged["sym_class"] = merged.apply(
        lambda r: _classify_symbol(r["symbol"], r["full_name"]), axis=1
    )
    # market_cap_bucket: unknown until probed
    merged["has_daily_bar"] = pd.NA
    merged["has_open_1m_window"] = pd.NA
    return merged.drop(columns=["attrs"])


# ---------------------------------------------------------------------------
# Alpaca probe
# ---------------------------------------------------------------------------

def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()


def _probe_daily_coverage(
    client: AlpacaClient,
    symbols: List[str],
    probe_date: date,
    batch_size: int,
    feed: str,
) -> Dict[str, bool]:
    """Return symbol -> has_daily_bar for the day before probe_date."""
    tz = ZoneInfo("America/New_York")
    # look back 5 calendar days to find prev_close
    from datetime import timedelta
    start = datetime.combine(probe_date - timedelta(days=10), dtime(0, 0), tzinfo=tz)
    end = datetime.combine(probe_date, dtime(0, 0), tzinfo=tz)
    result: Dict[str, bool] = {}
    for batch in _chunk(symbols, batch_size):
        resp = client.get_bars(
            symbols=batch,
            start=_iso_utc(start),
            end=_iso_utc(end),
            timeframe="1Day",
            feed=feed,
            adjustment="raw",
            asof=None,
            page_token=None,
        )
        bars = resp.get("bars", {}) if isinstance(resp, dict) else {}
        if isinstance(bars, dict):
            for sym in batch:
                result[sym] = bool(bars.get(sym))
        else:
            for sym in batch:
                result[sym] = False
    return result


def _probe_open_coverage(
    client: AlpacaClient,
    symbols: List[str],
    probe_date: date,
    batch_size: int,
    feed: str,
) -> Dict[str, bool]:
    """Return symbol -> has_open_1m_window (9:30-9:35 ET 1-min bars)."""
    tz = ZoneInfo("America/New_York")
    start = datetime.combine(probe_date, dtime(9, 30), tzinfo=tz)
    end = datetime.combine(probe_date, dtime(9, 35), tzinfo=tz)
    result: Dict[str, bool] = {}
    for batch in _chunk(symbols, batch_size):
        resp = client.get_bars(
            symbols=batch,
            start=_iso_utc(start),
            end=_iso_utc(end),
            timeframe="1Min",
            feed=feed,
            adjustment="raw",
            asof=None,
            page_token=None,
        )
        bars = resp.get("bars", {}) if isinstance(resp, dict) else {}
        if isinstance(bars, dict):
            for sym in batch:
                result[sym] = bool(bars.get(sym))
        else:
            for sym in batch:
                result[sym] = False
    return result


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d else "n/a"


def print_summary(df: pd.DataFrame, probed: bool) -> None:
    print("\n" + "=" * 65)
    print("=== universe coverage probe: summary ===")
    print("=" * 65)

    print(f"\n[1] 銘柄タイプ別件数 (seed 2000)")
    cls_counts = df["sym_class"].value_counts()
    for k, v in cls_counts.items():
        print(f"  {k:20s}: {v:5d} ({_pct(v, len(df))})")

    print(f"\n[2] exchange 別件数")
    exch_counts = df["exchange"].value_counts()
    for k, v in exch_counts.items():
        print(f"  {k:10s}: {v:5d}")

    if not probed:
        print("\n  (API probe skipped -- add --probe-date YYYY-MM-DD for coverage data)")
        return

    print(f"\n[3] 銘柄タイプ別 daily_bar / open_1m 取得率")
    print(f"  {'type':20s} {'count':>6} {'daily':>7} {'daily%':>8} {'open1m':>7} {'open1m%':>9}")
    print("  " + "-" * 62)
    for cls in df["sym_class"].unique():
        sub = df[df["sym_class"] == cls]
        n = len(sub)
        daily_n = sub["has_daily_bar"].sum()
        open_n = sub["has_open_1m_window"].sum()
        print(f"  {cls:20s} {n:6d} {daily_n:7d} {_pct(daily_n, n):>8} {open_n:7d} {_pct(open_n, n):>9}")
    total = len(df)
    d_tot = df["has_daily_bar"].sum()
    o_tot = df["has_open_1m_window"].sum()
    print("  " + "-" * 62)
    print(f"  {'TOTAL':20s} {total:6d} {d_tot:7d} {_pct(d_tot, total):>8} {o_tot:7d} {_pct(o_tot, total):>9}")

    print(f"\n[4] exchange 別 open_1m 取得率")
    print(f"  {'exchange':10s} {'count':>6} {'open1m':>7} {'open1m%':>9}")
    print("  " + "-" * 36)
    for exch in df["exchange"].unique():
        sub = df[df["exchange"] == exch]
        n = len(sub)
        o = sub["has_open_1m_window"].sum()
        print(f"  {exch:10s} {n:6d} {o:7d} {_pct(o, n):>9}")

    # [5] common-only hypothetical
    common_df = df[df["sym_class"].isin(["common", "etf"])]
    common_open = common_df["has_open_1m_window"].sum()
    print(f"\n[5] common + ETF のみ仮定した場合の open_1m coverage")
    print(f"  common+etf count : {len(common_df)}")
    print(f"  open_1m count    : {common_open}")
    print(f"  open_1m rate     : {_pct(common_open, len(common_df))}")
    print(f"  (vs full seed)   : {_pct(common_open, total)}")

    # [6] non-standard contribution
    nonstd = df[~df["sym_class"].isin(["common", "etf"])]
    nonstd_open = nonstd["has_open_1m_window"].sum()
    print(f"\n[6] 非標準銘柄 (warrant/preferred/right/unit/unknown) の寄与")
    print(f"  count        : {len(nonstd)}")
    print(f"  open_1m hits : {nonstd_open}")
    print(f"  → これを除外しても open_count は {nonstd_open} しか減らない")
    print(f"    (全 open_count {int(o_tot)} に対して {_pct(nonstd_open, int(o_tot))} 相当)")

    print(f"\n[7] IEX要因 / universe構成要因 の寄与推定")
    missing_total = total - int(o_tot)
    noncommon_no_open = len(nonstd) - nonstd_open
    iex_gap = missing_total - noncommon_no_open
    print(f"  全 missing_open       : {missing_total}")
    print(f"  非標準銘柄の open未取得: {noncommon_no_open}  ← universe構成要因上限")
    print(f"  common/ETF の open未取得: {iex_gap}  ← IEX coverage要因下限")
    print(f"  → universe整理で改善できるのは最大 {_pct(noncommon_no_open, missing_total)}")
    print(f"  → IEX feed切替で改善できるのは少なくとも {_pct(iex_gap, missing_total)}")

    print("\n[8] 直ちに除外してよい候補（open_1m = 0 の非標準銘柄）")
    zero_nonstd = nonstd[nonstd["has_open_1m_window"] == False]
    for cls in ["warrant", "preferred", "right", "unit", "right_or_other", "unknown"]:
        sub = zero_nonstd[zero_nonstd["sym_class"] == cls]
        if len(sub):
            print(f"  {cls:20s}: {len(sub)} 件 → scan対象から除外しても open_count は変わらない")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe_universe_coverage] seed={args.seed_path}")
    df = build_static_df(args.seed_path, args.assets_path)
    print(f"  seed loaded: {len(df)} symbols")

    probed = False
    if not args.static_only and args.probe_date:
        probe_date = date.fromisoformat(args.probe_date)
        print(f"  probing Alpaca feed={args.feed} date={probe_date} ...")
        client = AlpacaClient.from_env("https://data.alpaca.markets")

        symbols = df["symbol"].tolist()

        print("  → daily bar coverage ...")
        daily_cov = _probe_daily_coverage(client, symbols, probe_date, args.batch_size, args.feed)
        df["has_daily_bar"] = df["symbol"].map(daily_cov).fillna(False)

        # open probe: only symbols with daily bars (mirrors gap_scanner behaviour)
        daily_syms = [s for s in symbols if daily_cov.get(s)]
        print(f"  → open 1-min coverage ({len(daily_syms)} daily-bar symbols) ...")
        open_cov = _probe_open_coverage(client, daily_syms, probe_date, args.batch_size, args.feed)
        df["has_open_1m_window"] = df["symbol"].map(open_cov).fillna(False)
        probed = True
        print(f"  daily_count : {df['has_daily_bar'].sum()}")
        print(f"  open_count  : {df['has_open_1m_window'].sum()}")
    elif args.static_only:
        print("  --static-only: skipping API probe")
    else:
        print("  no --probe-date given; static analysis only")

    print_summary(df, probed)

    # CSV出力
    csv_path = args.out_dir / f"universe_coverage{'_' + args.probe_date if args.probe_date else '_static'}.csv"
    out_cols = [
        "symbol", "seed_rank", "exchange", "tradable", "shortable",
        "sym_class", "has_options", "full_name",
        "has_daily_bar", "has_open_1m_window",
    ]
    df[out_cols].to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
