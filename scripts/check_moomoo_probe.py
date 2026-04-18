"""
moomoo market data 最小接続確認 (3): small probe

AAPL / NVDA / AAL の snapshot + K_1M を一括取得し、
gap 戦略で使う open/volume/prev_close が揃うか確認する。

前提:
  - OpenD が起動済み（127.0.0.1:11111）
  - moomoo-api インストール済み

Usage:
  python scripts/check_moomoo_probe.py
  python scripts/check_moomoo_probe.py --codes US.AAPL US.NVDA US.AAL
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
    p.add_argument("--kbars", type=int, default=5, help="K_1M 取得本数")
    return p.parse_args()


def _check_gap_fields(snapshot_row: dict) -> list[str]:
    """gap 戦略で必要なフィールドが揃っているか確認。欠損フィールドを返す。"""
    required = ["last_price", "open_price", "prev_close_price", "volume"]
    return [f for f in required if not snapshot_row.get(f)]


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

    print(f"[moomoo] SDK={ft.__version__}  OpenD={args.host}:{args.port}")
    print(f"[moomoo] probe codes: {args.codes}")
    print()

    if not _check_opend(args.host, args.port):
        print(f"[ERROR] OpenD not reachable at {args.host}:{args.port}")
        print("  -> OpenD を起動してから再実行してください")
        sys.exit(1)
    print(f"[OK] OpenD reachable at {args.host}:{args.port}")
    print()

    results: dict = {"ts_utc": datetime.now(timezone.utc).isoformat(), "codes": args.codes, "snapshot": {}, "kline": {}, "errors": []}

    quote_ctx = ft.OpenQuoteContext(host=args.host, port=args.port)
    try:
        # --- snapshot ---
        ret, snap = quote_ctx.get_market_snapshot(args.codes)
        if ret != ft.RET_OK:
            msg = f"get_market_snapshot error: {snap}"
            print(f"[ERROR] {msg}")
            results["errors"].append(msg)
        else:
            print(f"[OK] snapshot: {len(snap)} rows")
            snap_cols = [c for c in ["code", "update_time", "last_price", "open_price", "prev_close_price", "volume", "suspension"] if c in snap.columns]
            print(snap[snap_cols].to_string(index=False))
            print()
            for row in snap.to_dict(orient="records"):
                code = row.get("code", "")
                missing = _check_gap_fields(row)
                status = "OK" if not missing else f"MISSING: {missing}"
                print(f"  gap fields check [{code}]: {status}")
                results["snapshot"][code] = {"fields_ok": not missing, "missing": missing, "data": row}
            print()

        # --- subscribe + K_1M ---
        ret, msg = quote_ctx.subscribe(args.codes, [ft.SubType.K_1M], subscribe_push=False)
        if ret != ft.RET_OK:
            err = f"subscribe K_1M error: {msg}"
            print(f"[ERROR] {err}")
            results["errors"].append(err)
        else:
            print("[OK] subscribe K_1M")
            for code in args.codes:
                ret2, kdata = quote_ctx.get_cur_kline(code, num=args.kbars, ktype=ft.KLType.K_1M)
                if ret2 != ft.RET_OK:
                    err = f"get_cur_kline [{code}] error: {kdata}"
                    print(f"  [ERROR] {err}")
                    results["errors"].append(err)
                    results["kline"][code] = {"rows": 0, "error": str(kdata)}
                else:
                    print(f"  [OK] K_1M [{code}]: {len(kdata)} rows")
                    kcols = [c for c in ["time_key", "open", "close", "volume"] if c in kdata.columns]
                    if not kdata.empty:
                        print(kdata[kcols].tail(3).to_string(index=False))
                    results["kline"][code] = {"rows": len(kdata), "data": kdata.to_dict(orient="records")}
            print()

    finally:
        quote_ctx.close()

    # サマリ
    print("=" * 50)
    print("=== probe summary ===")
    print(f"  errors    : {len(results['errors'])}")
    for e in results["errors"]:
        print(f"    - {e}")
    print(f"  snapshot  : {len(results['snapshot'])} codes")
    for code, r in results["snapshot"].items():
        print(f"    {code}: {'OK' if r['fields_ok'] else 'MISSING ' + str(r['missing'])}")
    print(f"  kline 1m  : {len(results['kline'])} codes")
    for code, r in results["kline"].items():
        print(f"    {code}: {r.get('rows', 0)} bars")

    # JSON 保存
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"artifacts/coverage/moomoo_probe_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out_path}")

    if results["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
