"""
compare_moomoo_alpaca_open.py

2026-04-17 9:30-9:35 ET の 1-min bar を moomoo と Alpaca IEX/SIP で比較する。

Usage:
  # 3銘柄のみ (quick check)
  python scripts/compare_moomoo_alpaca_open.py --date 2026-04-17

  # small basket (20-50銘柄)
  python scripts/compare_moomoo_alpaca_open.py --date 2026-04-17 --basket

  # Alpaca feed を SIP に変更
  python scripts/compare_moomoo_alpaca_open.py --date 2026-04-17 --basket --feed sip
"""

from __future__ import annotations

import argparse
import socket
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlstock.data.alpaca.client import AlpacaClient  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# Small basket: common 中心、NYSE/Nasdaq 混在、低価格帯含む
# ---------------------------------------------------------------------------
CORE_3 = ["US.AAPL", "US.NVDA", "US.AAL"]

BASKET_MOOMOO = [
    # large cap Nasdaq
    "US.AAPL",
    "US.MSFT",
    "US.NVDA",
    "US.AMZN",
    "US.GOOGL",
    "US.META",
    # large cap NYSE
    "US.JPM",
    "US.BAC",
    "US.XOM",
    "US.JNJ",
    "US.WMT",
    "US.PG",
    # mid cap / momentum
    "US.AMD",
    "US.TSLA",
    "US.PLTR",
    "US.SOFI",
    "US.RIVN",
    "US.LCID",
    # low price / high vol
    "US.AAL",
    "US.CCL",
    "US.F",
    "US.SNAP",
    "US.HOOD",
    "US.MARA",
    # ETF
    "US.SPY",
    "US.QQQ",
    "US.IWM",
    "US.SQQQ",
    "US.TQQQ",
    # NYSE utilities/finance
    "US.T",
    "US.VZ",
    "US.C",
    "US.GS",
    "US.MS",
]

# Alpaca 用: "US." プレフィックスなし
BASKET_ALPACA = [s.replace("US.", "") for s in BASKET_MOOMOO]
CORE_3_ALPACA = [s.replace("US.", "") for s in CORE_3]

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


def _open_window(probe_date: date):
    start = datetime.combine(probe_date, dtime(9, 30), tzinfo=ET)
    end = datetime.combine(probe_date, dtime(9, 35), tzinfo=ET)
    return start, end


# ---------------------------------------------------------------------------
# moomoo fetch
# ---------------------------------------------------------------------------


def fetch_moomoo(codes_mm: list[str], probe_date: date, host: str = "127.0.0.1", port: int = 11111) -> pd.DataFrame:
    """Return per-symbol rows from moomoo K_1M for the open window."""
    import moomoo as ft

    if not _check_opend(host, port):
        print(f"[moomoo] ERROR: OpenD not reachable at {host}:{port}", file=sys.stderr)
        sys.exit(1)

    start, end = _open_window(probe_date)
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    quote_ctx = ft.OpenQuoteContext(host=host, port=port)
    try:
        # subscribe batch
        ret, msg = quote_ctx.subscribe(codes_mm, [ft.SubType.K_1M], subscribe_push=False)
        if ret != ft.RET_OK:
            print(f"[moomoo] subscribe error: {msg}", file=sys.stderr)

        for code in codes_mm:
            alpaca_sym = code.replace("US.", "")
            # get_history_kline で日付範囲指定
            ret2, kdata, _ = quote_ctx.request_history_kline(
                code,
                start=start_str,
                end=end_str,
                ktype=ft.KLType.K_1M,
                autype=ft.AuType.NONE,
                max_count=20,
            )
            if ret2 != ft.RET_OK:
                rows.append(
                    {
                        "symbol": alpaca_sym,
                        "source": "moomoo",
                        "bars_in_window": 0,
                        "first_time_key": None,
                        "last_time_key": None,
                        "open_available": False,
                        "volume_available": False,
                        "status": f"ERROR: {kdata}",
                    }
                )
                continue

            # filter to window [9:30, 9:35) — moomoo returns tz-naive local strings
            if not kdata.empty and "time_key" in kdata.columns:
                kdata["time_key"] = pd.to_datetime(kdata["time_key"])
                start_naive = pd.Timestamp(start.replace(tzinfo=None))
                end_naive = pd.Timestamp(end.replace(tzinfo=None))
                kdata = kdata[(kdata["time_key"] >= start_naive) & (kdata["time_key"] < end_naive)]

            n = len(kdata)
            if n == 0:
                rows.append(
                    {
                        "symbol": alpaca_sym,
                        "source": "moomoo",
                        "bars_in_window": 0,
                        "first_time_key": None,
                        "last_time_key": None,
                        "open_available": False,
                        "volume_available": False,
                        "status": "NO_BARS",
                    }
                )
            else:
                has_open = "open" in kdata.columns and kdata["open"].notna().any()
                has_vol = "volume" in kdata.columns and kdata["volume"].notna().any()
                rows.append(
                    {
                        "symbol": alpaca_sym,
                        "source": "moomoo",
                        "bars_in_window": n,
                        "first_time_key": str(kdata["time_key"].iloc[0]),
                        "last_time_key": str(kdata["time_key"].iloc[-1]),
                        "open_available": has_open,
                        "volume_available": has_vol,
                        "status": "OK",
                    }
                )
    finally:
        quote_ctx.close()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alpaca fetch
# ---------------------------------------------------------------------------


def fetch_alpaca(symbols: list[str], probe_date: date, feed: str = "iex") -> pd.DataFrame:
    start, end = _open_window(probe_date)
    client = AlpacaClient.from_env("https://data.alpaca.markets")

    # batch fetch
    resp = client.get_bars(
        symbols=symbols,
        start=_iso_utc(start),
        end=_iso_utc(end),
        timeframe="1Min",
        feed=feed,
        adjustment="raw",
    )
    bars_map: dict = resp.get("bars", {}) if isinstance(resp, dict) else {}

    rows = []
    for sym in symbols:
        sym_bars = bars_map.get(sym, [])
        if not sym_bars:
            rows.append(
                {
                    "symbol": sym,
                    "source": f"alpaca_{feed}",
                    "bars_in_window": 0,
                    "first_time_key": None,
                    "last_time_key": None,
                    "open_available": False,
                    "volume_available": False,
                    "status": "NO_BARS",
                }
            )
        else:
            n = len(sym_bars)
            first_t = sym_bars[0].get("t", "")
            last_t = sym_bars[-1].get("t", "")
            has_open = any(b.get("o") is not None for b in sym_bars)
            has_vol = any(b.get("v") is not None for b in sym_bars)
            rows.append(
                {
                    "symbol": sym,
                    "source": f"alpaca_{feed}",
                    "bars_in_window": n,
                    "first_time_key": first_t,
                    "last_time_key": last_t,
                    "open_available": has_open,
                    "volume_available": has_vol,
                    "status": "OK",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compare & summarize
# ---------------------------------------------------------------------------


def compare(mm_df: pd.DataFrame, ap_df: pd.DataFrame) -> pd.DataFrame:
    mm = mm_df.rename(
        columns={
            "bars_in_window": "mm_bars",
            "open_available": "mm_open",
            "volume_available": "mm_vol",
            "status": "mm_status",
            "first_time_key": "mm_first",
            "last_time_key": "mm_last",
        }
    ).drop(columns=["source"])

    ap = ap_df.rename(
        columns={
            "bars_in_window": "ap_bars",
            "open_available": "ap_open",
            "volume_available": "ap_vol",
            "status": "ap_status",
            "first_time_key": "ap_first",
            "last_time_key": "ap_last",
        }
    ).drop(columns=["source"])

    merged = mm.merge(ap, on="symbol", how="outer")
    merged["mm_bars"] = merged["mm_bars"].fillna(0).astype(int)
    merged["ap_bars"] = merged["ap_bars"].fillna(0).astype(int)
    merged["verdict"] = merged.apply(
        lambda r: (
            "BOTH_OK"
            if r["mm_status"] == "OK" and r["ap_status"] == "OK"
            else ("MM_ONLY" if r["mm_status"] == "OK" else ("AP_ONLY" if r["ap_status"] == "OK" else "BOTH_MISSING"))
        ),
        axis=1,
    )
    return merged


def print_summary(df_cmp: pd.DataFrame, feed: str, probe_date: date) -> None:
    n = len(df_cmp)
    mm_ok = (df_cmp["mm_status"] == "OK").sum()
    ap_ok = (df_cmp["ap_status"] == "OK").sum()
    both_ok = (df_cmp["verdict"] == "BOTH_OK").sum()
    mm_only = (df_cmp["verdict"] == "MM_ONLY").sum()
    ap_only = (df_cmp["verdict"] == "AP_ONLY").sum()
    both_miss = (df_cmp["verdict"] == "BOTH_MISSING").sum()

    def pct(a, b):
        return f"{a/b*100:.1f}%" if b else "n/a"

    print(f"\n{'='*65}")
    print(f"=== moomoo vs Alpaca {feed.upper()} | {probe_date} 9:30-9:35 ET ===")
    print(f"{'='*65}")
    print(f"  total symbols   : {n}")
    print(f"  moomoo OK       : {mm_ok:3d} ({pct(mm_ok, n)})")
    print(f"  alpaca_{feed} OK : {ap_ok:3d} ({pct(ap_ok, n)})")
    print(f"  both OK         : {both_ok:3d} ({pct(both_ok, n)})")
    print(f"  moomoo only     : {mm_only}")
    print(f"  alpaca only     : {ap_only}")
    print(f"  both missing    : {both_miss}")

    display_cols = ["symbol", "mm_bars", "ap_bars", "mm_open", "ap_open", "mm_vol", "ap_vol", "verdict"]
    print("\n--- per-symbol ---")
    print(df_cmp[display_cols].sort_values("verdict").to_string(index=False))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-04-17")
    p.add_argument("--basket", action="store_true", help="Use ~35 symbol basket instead of 3")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    probe_date = date.fromisoformat(args.date)

    if args.basket:
        codes_mm = BASKET_MOOMOO
        codes_ap = BASKET_ALPACA
        label = "basket"
    else:
        codes_mm = CORE_3
        codes_ap = CORE_3_ALPACA
        label = "core3"

    print(f"[compare] date={probe_date}  feed={args.feed}  symbols={len(codes_mm)}  mode={label}")

    try:
        import moomoo  # noqa: F401
    except ImportError:
        print("[ERROR] moomoo-api not installed: pip install moomoo-api", file=sys.stderr)
        sys.exit(1)

    print("\n[1/2] fetching moomoo K_1M ...")
    mm_df = fetch_moomoo(codes_mm, probe_date, host=args.host, port=args.port)

    print("[2/2] fetching Alpaca bars ...")
    ap_df = fetch_alpaca(codes_ap, probe_date, feed=args.feed)

    df_cmp = compare(mm_df, ap_df)
    print_summary(df_cmp, args.feed, probe_date)

    # save
    out_dir = ROOT / "artifacts" / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compare_open_{probe_date}_{args.feed}_{label}_{ts}.csv"
    df_cmp.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
