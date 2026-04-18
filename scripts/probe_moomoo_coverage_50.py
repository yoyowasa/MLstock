"""
probe_moomoo_coverage_50.py

moomoo 無料権限の coverage を 50銘柄 basket で確認する。
  - snapshot 可否
  - K_1M 可否 (9:31-9:35 ET の指定日)
  - Unknown stock 発生率
  - large / mid / small 別集計
  - オプションで Alpaca IEX の同一 basket と比較

Usage:
  # snapshot のみ (live)
  python scripts/probe_moomoo_coverage_50.py --snapshot-only

  # K_1M replay (指定日)
  python scripts/probe_moomoo_coverage_50.py --kline-date 2026-04-17

  # snapshot + K_1M + Alpaca 比較
  python scripts/probe_moomoo_coverage_50.py --kline-date 2026-04-17 --alpaca-compare

前提: OpenD が 127.0.0.1:11111 で起動済み
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# 50-symbol basket
# category: large / mid / small
# NYSE/Nasdaq 混在、普通株のみ
# ---------------------------------------------------------------------------
BASKET: List[Dict[str, str]] = [
    # --- large-cap (15) ---
    {"symbol": "AAPL",  "cap": "large", "exchange": "NASDAQ", "name": "Apple"},
    {"symbol": "MSFT",  "cap": "large", "exchange": "NASDAQ", "name": "Microsoft"},
    {"symbol": "NVDA",  "cap": "large", "exchange": "NASDAQ", "name": "NVIDIA"},
    {"symbol": "AMZN",  "cap": "large", "exchange": "NASDAQ", "name": "Amazon"},
    {"symbol": "GOOGL", "cap": "large", "exchange": "NASDAQ", "name": "Alphabet A"},
    {"symbol": "META",  "cap": "large", "exchange": "NASDAQ", "name": "Meta"},
    {"symbol": "TSLA",  "cap": "large", "exchange": "NASDAQ", "name": "Tesla"},
    {"symbol": "JPM",   "cap": "large", "exchange": "NYSE",   "name": "JPMorgan"},
    {"symbol": "XOM",   "cap": "large", "exchange": "NYSE",   "name": "Exxon"},
    {"symbol": "JNJ",   "cap": "large", "exchange": "NYSE",   "name": "J&J"},
    {"symbol": "WMT",   "cap": "large", "exchange": "NYSE",   "name": "Walmart"},
    {"symbol": "BAC",   "cap": "large", "exchange": "NYSE",   "name": "BofA"},
    {"symbol": "GS",    "cap": "large", "exchange": "NYSE",   "name": "Goldman"},
    {"symbol": "UNH",   "cap": "large", "exchange": "NYSE",   "name": "UnitedHealth"},
    {"symbol": "HD",    "cap": "large", "exchange": "NYSE",   "name": "Home Depot"},
    # --- mid-cap (20) ---
    {"symbol": "AMD",   "cap": "mid",   "exchange": "NASDAQ", "name": "AMD"},
    {"symbol": "PLTR",  "cap": "mid",   "exchange": "NASDAQ", "name": "Palantir"},
    {"symbol": "SNAP",  "cap": "mid",   "exchange": "NYSE",   "name": "Snap"},
    {"symbol": "SOFI",  "cap": "mid",   "exchange": "NASDAQ", "name": "SoFi"},
    {"symbol": "RIVN",  "cap": "mid",   "exchange": "NASDAQ", "name": "Rivian"},
    {"symbol": "LCID",  "cap": "mid",   "exchange": "NASDAQ", "name": "Lucid"},
    {"symbol": "HOOD",  "cap": "mid",   "exchange": "NASDAQ", "name": "Robinhood"},
    {"symbol": "AFRM",  "cap": "mid",   "exchange": "NASDAQ", "name": "Affirm"},
    {"symbol": "ALLY",  "cap": "mid",   "exchange": "NYSE",   "name": "Ally Financial"},
    {"symbol": "CCL",   "cap": "mid",   "exchange": "NYSE",   "name": "Carnival"},
    {"symbol": "CUK",   "cap": "mid",   "exchange": "NYSE",   "name": "Carnival UK"},
    {"symbol": "F",     "cap": "mid",   "exchange": "NYSE",   "name": "Ford"},
    {"symbol": "GM",    "cap": "mid",   "exchange": "NYSE",   "name": "GM"},
    {"symbol": "UBER",  "cap": "mid",   "exchange": "NYSE",   "name": "Uber"},
    {"symbol": "LYFT",  "cap": "mid",   "exchange": "NASDAQ", "name": "Lyft"},
    {"symbol": "DASH",  "cap": "mid",   "exchange": "NYSE",   "name": "DoorDash"},
    {"symbol": "COIN",  "cap": "mid",   "exchange": "NASDAQ", "name": "Coinbase"},
    {"symbol": "MARA",  "cap": "mid",   "exchange": "NASDAQ", "name": "Marathon Digital"},
    {"symbol": "RIOT",  "cap": "mid",   "exchange": "NASDAQ", "name": "Riot Platforms"},
    {"symbol": "APTV",  "cap": "mid",   "exchange": "NYSE",   "name": "Aptiv"},
    # --- small-cap (15) ---
    {"symbol": "AAL",   "cap": "small", "exchange": "NASDAQ", "name": "American Airlines"},
    {"symbol": "ASST",  "cap": "small", "exchange": "NYSE",   "name": "Asset Entities"},
    {"symbol": "WKHS",  "cap": "small", "exchange": "NASDAQ", "name": "Workhorse"},
    {"symbol": "HIMS",  "cap": "small", "exchange": "NYSE",   "name": "Hims & Hers"},
    {"symbol": "SPCE",  "cap": "small", "exchange": "NYSE",   "name": "Virgin Galactic"},
    {"symbol": "NKLA",  "cap": "small", "exchange": "NASDAQ", "name": "Nikola"},
    {"symbol": "OPEN",  "cap": "small", "exchange": "NASDAQ", "name": "Opendoor"},
    {"symbol": "BKKT",  "cap": "small", "exchange": "NASDAQ", "name": "Bakkt"},
    {"symbol": "OPAD",  "cap": "small", "exchange": "NASDAQ", "name": "Offerpad"},
    {"symbol": "GREE",  "cap": "small", "exchange": "NASDAQ", "name": "Greenidge Gen"},
    {"symbol": "IMNM",  "cap": "small", "exchange": "NASDAQ", "name": "Immunome"},
    {"symbol": "SNDL",  "cap": "small", "exchange": "NASDAQ", "name": "SNDL Inc"},
    {"symbol": "CLOV",  "cap": "small", "exchange": "NASDAQ", "name": "Clover Health"},
    {"symbol": "SKLZ",  "cap": "small", "exchange": "NASDAQ", "name": "Skillz"},
    {"symbol": "XELA",  "cap": "small", "exchange": "NASDAQ", "name": "Xerox Holdings alt"},
]

assert len(BASKET) == 50, f"basket size = {len(BASKET)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_opend(host: str, port: int, timeout: float = 3.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i: i + size]


# ---------------------------------------------------------------------------
# moomoo snapshot probe
# ---------------------------------------------------------------------------

def probe_snapshot(
    quote_ctx: Any,
    codes_mm: List[str],
) -> Dict[str, Dict[str, Any]]:
    """symbol -> snapshot row dict. Per-symbol fallback on batch error."""
    import moomoo as ft
    result: Dict[str, Dict[str, Any]] = {}

    def _try(codes: List[str]) -> bool:
        ret, snap = quote_ctx.get_market_snapshot(codes)
        if ret != ft.RET_OK:
            return False
        for row in snap.to_dict(orient="records"):
            sym = str(row.get("code", "")).replace("US.", "")
            result[sym] = row
        return True

    for batch in _chunk(codes_mm, 50):
        if not _try(batch):
            for code in batch:
                _try([code])
    return result


# ---------------------------------------------------------------------------
# moomoo K_1M probe
# ---------------------------------------------------------------------------

def probe_kline(
    quote_ctx: Any,
    codes_mm: List[str],
    kline_date: date,
    window_start: str = "09:31:00",
    window_end: str = "09:36:00",
) -> Dict[str, Dict[str, Any]]:
    """symbol -> kline probe result dict."""
    import moomoo as ft

    start_str = f"{kline_date.isoformat()} {window_start}"
    end_str = f"{kline_date.isoformat()} {window_end}"

    # subscribe batch (required before request_history_kline)
    for batch in _chunk(codes_mm, 100):
        quote_ctx.subscribe(batch, [ft.SubType.K_1M], subscribe_push=False)

    result: Dict[str, Dict[str, Any]] = {}
    for code in codes_mm:
        sym = code.replace("US.", "")
        ret, kdata, _ = quote_ctx.request_history_kline(
            code,
            start=start_str,
            end=end_str,
            ktype=ft.KLType.K_1M,
            autype=ft.AuType.NONE,
            max_count=10,
        )
        if ret != 0:
            result[sym] = {
                "ok": False, "bars": 0,
                "error": str(kdata)[:120],
                "first_time_key": None, "last_time_key": None,
                "open_avail": False, "close_avail": False, "volume_avail": False,
            }
            continue
        if kdata is None or kdata.empty:
            result[sym] = {
                "ok": False, "bars": 0, "error": "empty",
                "first_time_key": None, "last_time_key": None,
                "open_avail": False, "close_avail": False, "volume_avail": False,
            }
            continue
        n = len(kdata)
        first_t = str(kdata["time_key"].iloc[0]) if "time_key" in kdata.columns else None
        last_t = str(kdata["time_key"].iloc[-1]) if "time_key" in kdata.columns else None
        result[sym] = {
            "ok": True, "bars": n, "error": None,
            "first_time_key": first_t, "last_time_key": last_t,
            "open_avail": "open" in kdata.columns and kdata["open"].notna().any(),
            "close_avail": "close" in kdata.columns and kdata["close"].notna().any(),
            "volume_avail": "volume" in kdata.columns and kdata["volume"].notna().any(),
        }
    return result


# ---------------------------------------------------------------------------
# Alpaca IEX probe
# ---------------------------------------------------------------------------

def probe_alpaca(
    symbols: List[str],
    kline_date: date,
    feed: str = "iex",
    window_start_t: dtime = dtime(9, 31),
    window_end_t: dtime = dtime(9, 36),
) -> Dict[str, Dict[str, Any]]:
    """symbol -> alpaca probe result dict."""
    from mlstock.data.alpaca.client import AlpacaClient
    from mlstock.config.loader import load_config
    cfg = load_config(ROOT / "config" / "config.yaml")
    client = AlpacaClient.from_env(cfg.alpaca.data_base_url)

    start = datetime.combine(kline_date, window_start_t, tzinfo=ET)
    end = datetime.combine(kline_date, window_end_t, tzinfo=ET)

    result: Dict[str, Dict[str, Any]] = {}
    for batch in _chunk(symbols, 200):
        resp = client.get_bars(
            symbols=batch,
            start=_iso_utc(start),
            end=_iso_utc(end),
            timeframe="1Min",
            feed=feed,
            adjustment="raw",
        )
        bars_map = resp.get("bars", {}) if isinstance(resp, dict) else {}
        for sym in batch:
            sym_bars = bars_map.get(sym, [])
            if not sym_bars:
                result[sym] = {"ok": False, "bars": 0, "first_time_key": None, "last_time_key": None}
            else:
                result[sym] = {
                    "ok": True, "bars": len(sym_bars),
                    "first_time_key": sym_bars[0].get("t"),
                    "last_time_key": sym_bars[-1].get("t"),
                }
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d else "n/a"


def build_result_df(
    basket: List[Dict[str, str]],
    snap_result: Dict[str, Dict],
    kline_result: Optional[Dict[str, Dict]],
    alpaca_result: Optional[Dict[str, Dict]],
) -> pd.DataFrame:
    rows = []
    for item in basket:
        sym = item["symbol"]
        cap = item["cap"]
        exch = item["exchange"]

        snap = snap_result.get(sym, {})
        snap_ok = bool(snap)
        snap_open = float(snap.get("open_price", 0) or 0) if snap_ok else None
        snap_update = str(snap.get("update_time", "")) if snap_ok else ""

        kline = kline_result.get(sym, {}) if kline_result else None
        kline_ok = bool(kline and kline.get("ok"))
        kline_bars = int(kline.get("bars", 0)) if kline else 0
        kline_first = kline.get("first_time_key") if kline else None
        kline_err = kline.get("error") if kline else None

        ap = alpaca_result.get(sym, {}) if alpaca_result else None
        ap_ok = bool(ap and ap.get("ok"))
        ap_bars = int(ap.get("bars", 0)) if ap else 0

        if snap_ok and kline_ok:
            verdict = "BOTH_OK"
        elif snap_ok and not kline_ok:
            verdict = "SNAP_ONLY"
        elif not snap_ok and kline_ok:
            verdict = "KLINE_ONLY"
        else:
            verdict = "BOTH_FAIL"

        rows.append({
            "symbol": sym, "cap": cap, "exchange": exch,
            "snap_ok": snap_ok, "snap_open": snap_open, "snap_update": snap_update,
            "kline_ok": kline_ok, "kline_bars": kline_bars, "kline_first": kline_first,
            "kline_err": kline_err,
            "ap_ok": ap_ok, "ap_bars": ap_bars,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, kline_date: Optional[date], alpaca_compare: bool) -> None:
    n = len(df)
    snap_ok = df["snap_ok"].sum()
    kline_ok = df["kline_ok"].sum() if "kline_ok" in df.columns else 0
    both_ok = (df["verdict"] == "BOTH_OK").sum()
    snap_only = (df["verdict"] == "SNAP_ONLY").sum()
    kline_only = (df["verdict"] == "KLINE_ONLY").sum()
    both_fail = (df["verdict"] == "BOTH_FAIL").sum()
    ap_ok = df["ap_ok"].sum() if alpaca_compare else None

    print(f"\n{'='*65}")
    print("=== moomoo 50-basket coverage probe ===")
    if kline_date:
        print(f"    K_1M date: {kline_date}  window: 09:31-09:36 ET")
    print(f"{'='*65}")

    # overall
    print(f"\n[1] 全体 coverage (n={n})")
    print(f"  snapshot OK      : {snap_ok:3d} / {n} ({_pct(snap_ok, n)})")
    if kline_date:
        print(f"  K_1M OK          : {kline_ok:3d} / {n} ({_pct(kline_ok, n)})")
        print(f"  both OK          : {both_ok:3d}   snap_only={snap_only}  kline_only={kline_only}  both_fail={both_fail}")
    if alpaca_compare and ap_ok is not None:
        print(f"  Alpaca IEX OK    : {ap_ok:3d} / {n} ({_pct(ap_ok, n)})")

    # by cap
    print("\n[2] cap 別 coverage")
    print(f"  {'cap':8s} {'n':>4} {'snap_ok':>8} {'snap%':>7}", end="")
    if kline_date:
        print(f" {'kline_ok':>9} {'kline%':>7}", end="")
    if alpaca_compare:
        print(f" {'ap_ok':>7} {'ap%':>7}", end="")
    print()
    print("  " + "-" * (34 + (18 if kline_date else 0) + (16 if alpaca_compare else 0)))
    for cap in ["large", "mid", "small"]:
        sub = df[df["cap"] == cap]
        ns = len(sub)
        so = sub["snap_ok"].sum()
        row = f"  {cap:8s} {ns:4d} {so:8d} {_pct(so, ns):>7}"
        if kline_date:
            ko = sub["kline_ok"].sum()
            row += f" {ko:9d} {_pct(ko, ns):>7}"
        if alpaca_compare:
            ao = sub["ap_ok"].sum()
            row += f" {ao:7d} {_pct(ao, ns):>7}"
        print(row)

    # by exchange
    print("\n[3] exchange 別 coverage")
    for exch in ["NASDAQ", "NYSE"]:
        sub = df[df["exchange"] == exch]
        ns = len(sub)
        so = sub["snap_ok"].sum()
        line = f"  {exch:8s}: snap {so}/{ns} ({_pct(so, ns)})"
        if kline_date:
            ko = sub["kline_ok"].sum()
            line += f"  kline {ko}/{ns} ({_pct(ko, ns)})"
        print(line)

    # Unknown / error breakdown
    if kline_date:
        print("\n[4] K_1M エラー内訳")
        err_df = df[~df["kline_ok"]][["symbol", "cap", "kline_err"]].copy()
        if err_df.empty:
            print("  エラーなし")
        else:
            from collections import Counter
            ctr: Counter = Counter()
            for err in err_df["kline_err"].fillna("empty"):
                key = err[:60] if err else "empty"
                if "Unknown stock" in key:
                    key = "Unknown stock"
                ctr[key] += 1
            for k, v in ctr.most_common():
                print(f"  {v:4d}x  {k}")
            unknown_count = sum(1 for e in err_df["kline_err"].fillna("") if "Unknown stock" in str(e))
            print(f"\n  Unknown stock 件数: {unknown_count} / {len(err_df)} error  ({_pct(unknown_count, n)} 全体比)")

    # per-symbol table
    print("\n[5] 銘柄別詳細")
    cols = ["symbol", "cap", "exchange", "snap_ok", "kline_ok", "kline_bars", "kline_first", "verdict"]
    if alpaca_compare:
        cols += ["ap_ok", "ap_bars"]
    if kline_date:
        print(df[cols].sort_values(["cap", "verdict"]).to_string(index=False))
    else:
        snap_cols = ["symbol", "cap", "exchange", "snap_ok", "snap_open", "snap_update"]
        print(df[snap_cols].sort_values(["cap", "snap_ok"]).to_string(index=False))

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kline-date", type=str, default=None,
                   help="K_1M probe date YYYY-MM-DD (省略時は snapshot のみ)")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--alpaca-compare", action="store_true",
                   help="Alpaca IEX との比較も実施")
    p.add_argument("--alpaca-feed", default="iex")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "coverage")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    args = _parse_args()

    if not _check_opend(args.host, args.port):
        print(f"[ERROR] OpenD not reachable at {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] OpenD reachable at {args.host}:{args.port}")

    try:
        import moomoo as ft
    except ImportError:
        print("[ERROR] moomoo-api not installed", file=sys.stderr)
        sys.exit(1)

    print(f"[moomoo] SDK={ft.__version__}  basket={len(BASKET)} symbols")

    kline_date = date.fromisoformat(args.kline_date) if args.kline_date else None
    do_kline = kline_date is not None and not args.snapshot_only

    codes_mm = [f"US.{item['symbol']}" for item in BASKET]
    symbols = [item["symbol"] for item in BASKET]

    quote_ctx = ft.OpenQuoteContext(host=args.host, port=args.port)
    try:
        # snapshot
        print(f"\n[1/{'3' if do_kline else '2'}] snapshot probe ({len(BASKET)} symbols) ...")
        snap_result = probe_snapshot(quote_ctx, codes_mm)
        print(f"  snapshot OK: {len(snap_result)}/{len(BASKET)}")

        # K_1M
        kline_result: Optional[Dict] = None
        if do_kline:
            print(f"[2/3] K_1M probe date={kline_date} ...")
            kline_result = probe_kline(quote_ctx, codes_mm, kline_date)
            kline_ok = sum(1 for v in kline_result.values() if v.get("ok"))
            print(f"  K_1M OK: {kline_ok}/{len(BASKET)}")
    finally:
        quote_ctx.close()

    # Alpaca compare
    alpaca_result: Optional[Dict] = None
    if args.alpaca_compare and kline_date:
        print(f"[{'3' if do_kline else '2'}/{'3' if do_kline else '2'}] Alpaca {args.alpaca_feed} probe ...")
        alpaca_result = probe_alpaca(symbols, kline_date, feed=args.alpaca_feed)
        ap_ok = sum(1 for v in alpaca_result.values() if v.get("ok"))
        print(f"  Alpaca {args.alpaca_feed} OK: {ap_ok}/{len(BASKET)}")

    # build result df
    df = build_result_df(BASKET, snap_result, kline_result, alpaca_result)
    print_summary(df, kline_date, args.alpaca_compare and alpaca_result is not None)

    # save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{kline_date}" if kline_date else "_snapshot_only"
    out_csv = args.out_dir / f"moomoo_coverage_50{suffix}_{ts}.csv"
    df.to_csv(out_csv, index=False)

    out_json = args.out_dir / f"moomoo_coverage_50{suffix}_{ts}.json"
    payload = {
        "meta": {
            "kline_date": str(kline_date) if kline_date else None,
            "alpaca_feed": args.alpaca_feed if alpaca_result else None,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        "basket": BASKET,
        "snapshot": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in snap_result.items()},
        "kline": kline_result or {},
        "alpaca": alpaca_result or {},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
