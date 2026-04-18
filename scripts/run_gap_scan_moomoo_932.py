"""
run_gap_scan_moomoo_932.py  —  moomoo 9:32 ET 実験版 gap scanner 実行スクリプト

【注意】
  - 実験実装。現行 Alpaca scanner の置換ではない。
  - live/order/unlock_trade は一切触らない。scan-only。
  - OpenD が起動済みであること (127.0.0.1:11111)

Usage:
  # 2026-04-17 replay (seed 2000銘柄)
  python scripts/run_gap_scan_moomoo_932.py --date 2026-04-17

  # small basket で動作確認
  python scripts/run_gap_scan_moomoo_932.py --date 2026-04-17 --symbols AAPL NVDA AAL TSLA AMD META

  # Alpaca feed を SIP に変更
  python scripts/run_gap_scan_moomoo_932.py --date 2026-04-17 --alpaca-feed sip
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from mlstock.config.loader import load_config  # noqa: E402
from mlstock.data.alpaca.client import AlpacaClient  # noqa: E402
from mlstock.jobs.gap_scanner_moomoo_932 import check_opend, scan_gap_candidates_moomoo_932  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="moomoo 9:32 ET 実験版 gap scanner")
    p.add_argument("--date", type=str, default=None, help="対象日 YYYY-MM-DD (デフォルト: 今日 ET)")
    p.add_argument("--symbols", nargs="*", default=None, help="銘柄リスト (省略時は seed 2000銘柄)")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--gap-config", type=Path, default=ROOT / "config" / "gap_config.yaml")
    p.add_argument("--opend-host", default="127.0.0.1")
    p.add_argument("--opend-port", type=int, default=11111)
    p.add_argument(
        "--alpaca-feed", default=None, help="Alpaca feed override (iex/sip). 省略時は config.yaml の bars.feed"
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "gap_scan_moomoo")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- OpenD preflight ---
    if not check_opend(args.opend_host, args.opend_port):
        print(f"[ERROR] OpenD not reachable at {args.opend_host}:{args.opend_port}", file=sys.stderr)
        print("  -> moomoo app から OpenD を有効化してください", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] OpenD reachable at {args.opend_host}:{args.opend_port}")

    # --- config load ---
    cfg = load_config(args.config)
    with open(args.gap_config, encoding="utf-8") as f:
        gap_cfg: dict = yaml.safe_load(f)

    # feed override
    if args.alpaca_feed:
        cfg.bars.feed = args.alpaca_feed  # type: ignore[attr-defined]

    # --- trade_date ---
    et = ZoneInfo("America/New_York")
    if args.date:
        trade_date = date.fromisoformat(args.date)
    else:
        trade_date = datetime.now(et).date()

    print(f"[scan] trade_date={trade_date}  feed={cfg.bars.feed}  symbols={args.symbols or 'seed'}")
    print()

    # --- Alpaca client ---
    alpaca_client = AlpacaClient.from_env(cfg.alpaca.data_base_url)

    # --- run scan ---
    candidates, diag = scan_gap_candidates_moomoo_932(
        cfg=cfg,
        gap_cfg=gap_cfg,
        alpaca_client=alpaca_client,
        trade_date=trade_date,
        opend_host=args.opend_host,
        opend_port=args.opend_port,
        symbols=args.symbols,
    )

    # --- print diagnostics ---
    print("=" * 60)
    print("=== moomoo 9:32 scan diagnostics ===")
    print(f"  trade_date              : {diag.trade_date}")
    print(f"  universe_count          : {diag.universe_count}")
    print(f"  daily_stats_count       : {diag.daily_stats_count}")
    print(f"  snapshot_ok_count       : {diag.snapshot_ok_count}")
    print(f"  kline_0931_ok_count     : {diag.kline_0931_ok_count}")
    print(f"  missing_snapshot_count  : {diag.missing_snapshot_count}")
    print(f"  missing_kline_count     : {diag.missing_kline_count}")
    print(f"  price_filter_drop       : {diag.price_filter_drop_count}")
    print(f"  gap_filter_drop         : {diag.gap_filter_drop_count}")
    print(f"  volume_filter_drop      : {diag.volume_filter_drop_count}")
    print(f"  market_cap_drop         : {diag.market_cap_drop_count}")
    print(f"  final_candidate_count   : {diag.final_candidate_count}")
    print()

    if not candidates:
        print("[RESULT] candidates: 0")
    else:
        print(f"[RESULT] candidates: {len(candidates)}")
        print()
        header = f"{'symbol':8s} {'open':>8} {'prev_cls':>8} {'gap%':>7} {'vol_0931':>12} {'vol_ratio':>10} {'pace_x':>8} {'mcap_m':>10} {'upd_time'}"
        print(header)
        print("-" * len(header))
        for c in candidates:
            print(
                f"{c.symbol:8s} {c.open_price:8.3f} {c.prev_close_price:8.3f} "
                f"{c.gap_pct:7.2f} {c.bar_0931_volume:12,.0f} "
                f"{c.volume_ratio:10.3f} {c.volume_pace_annualized:8.2f} "
                f"{c.market_cap_m:10.1f} {c.snapshot_update_time}"
            )
    print()

    # --- save ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"moomoo_932_scan_{trade_date}_{ts}.json"
    payload = {
        "meta": {
            "scanner": "moomoo_932_experimental",
            "trade_date": trade_date.isoformat(),
            "alpaca_feed": cfg.bars.feed,
            "opend": f"{args.opend_host}:{args.opend_port}",
            "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        },
        "diagnostics": diag.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
