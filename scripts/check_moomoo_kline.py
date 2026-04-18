"""
moomoo market data 最小接続確認 (2): K_1M subscribe + 取得

前提:
  - OpenD が起動済み（デフォルト: 127.0.0.1:11111）
  - moomoo-api インストール済み: pip install moomoo-api
  - US market 時間外でも historical K_1M は取得可能

Usage:
  python scripts/check_moomoo_kline.py
  python scripts/check_moomoo_kline.py --code US.AAPL --bars 10
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
    p.add_argument("--code", default="US.AAPL")
    p.add_argument("--bars", type=int, default=10, help="取得する 1-min bar 件数")
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
    print(f"[moomoo] code        : {args.code}")
    print()

    if not _check_opend(args.host, args.port):
        print(f"[ERROR] OpenD not reachable at {args.host}:{args.port}")
        print("  -> OpenD を起動してから再実行してください")
        sys.exit(1)
    print(f"[OK] OpenD reachable at {args.host}:{args.port}")

    quote_ctx = ft.OpenQuoteContext(host=args.host, port=args.port)
    try:
        # Step 1: subscribe K_1M
        ret, msg = quote_ctx.subscribe([args.code], [ft.SubType.K_1M], subscribe_push=False)
        if ret != ft.RET_OK:
            print(f"[ERROR] subscribe K_1M failed: {msg}")
            sys.exit(1)
        print(f"[OK] subscribe K_1M: {args.code}")

        # Step 2: get_cur_kline (最新 N 本)
        ret, data = quote_ctx.get_cur_kline(args.code, num=args.bars, ktype=ft.KLType.K_1M)
        if ret != ft.RET_OK:
            print(f"[ERROR] get_cur_kline failed: {data}")
            sys.exit(1)

        print(f"[OK] get_cur_kline rows: {len(data)}")
        print()

        cols = [c for c in ["code", "time_key", "open", "high", "low", "close", "volume", "turnover"] if c in data.columns]
        print(data[cols].to_string(index=False))
        print()

        # JSON 保存
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"artifacts/coverage/moomoo_kline1m_{ts}.json"
        records = data.to_dict(orient="records")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ts_utc": datetime.now(timezone.utc).isoformat(), "code": args.code, "rows": records}, f, ensure_ascii=False, indent=2, default=str)
        print(f"[saved] {out_path}")

    finally:
        quote_ctx.close()


if __name__ == "__main__":
    main()
