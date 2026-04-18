"""
moomoo market data 最小接続確認 (1): snapshot

前提:
  - OpenD が起動済み（デフォルト: 127.0.0.1:11111）
  - moomoo-api インストール済み: pip install moomoo-api

Usage:
  python scripts/check_moomoo_snapshot.py
  python scripts/check_moomoo_snapshot.py --host 127.0.0.1 --port 11111
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11111)
    p.add_argument("--codes", nargs="+", default=["US.AAPL", "US.NVDA", "US.AAL"])
    return p.parse_args()


def _check_opend(host: str, port: int, timeout: float = 3.0) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def main() -> None:
    args = _parse_args()

    try:
        import moomoo as ft
    except ImportError:
        print("[ERROR] moomoo-api not installed: pip install moomoo-api", file=sys.stderr)
        sys.exit(1)

    print(f"[moomoo] SDK version : {ft.__version__}")
    print(f"[moomoo] OpenD target: {args.host}:{args.port}")
    print(f"[moomoo] codes       : {args.codes}")
    print()

    if not _check_opend(args.host, args.port):
        print(f"[ERROR] OpenD not reachable at {args.host}:{args.port}")
        print("  -> OpenD を起動してから再実行してください")
        print("  -> DL: https://www.moomoo.com/us/download")
        sys.exit(1)
    print(f"[OK] OpenD reachable at {args.host}:{args.port}")

    quote_ctx = ft.OpenQuoteContext(host=args.host, port=args.port)
    try:
        ret, data = quote_ctx.get_market_snapshot(args.codes)
        if ret != ft.RET_OK:
            print(f"[ERROR] get_market_snapshot failed: {data}")
            sys.exit(1)

        print(f"[OK] snapshot rows: {len(data)}")
        print()

        # 主要カラムだけ表示
        cols = [
            "code",
            "update_time",
            "last_price",
            "open_price",
            "high_price",
            "low_price",
            "prev_close_price",
            "volume",
            "turnover",
            "amplitude",
            "price_change",
            "change_rate",
            "suspension",
            "list_time",
        ]
        available_cols = [c for c in cols if c in data.columns]
        print(data[available_cols].to_string(index=False))
        print()

        # JSON でも出力（再現性確保）
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"artifacts/coverage/moomoo_snapshot_{ts}.json"
        records = data.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"ts_utc": datetime.now(timezone.utc).isoformat(), "rows": records},
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"[saved] {out_path}")

    finally:
        quote_ctx.close()


if __name__ == "__main__":
    main()
