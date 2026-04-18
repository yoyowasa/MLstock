"""
compare_gap_scan_alpaca_vs_moomoo.py

同日の既存 Alpaca scanner 結果 と moomoo 9:32 試作 scanner 結果を並べて比較する。

既存 scanner の結果は artifacts/logs/ 以下の JSONL ログから取得。
moomoo 側は artifacts/gap_scan_moomoo/ 以下の JSON から取得。

Usage:
  # 直近のログを自動検出して比較
  python scripts/compare_gap_scan_alpaca_vs_moomoo.py --date 2026-04-17

  # ファイルを直接指定
  python scripts/compare_gap_scan_alpaca_vs_moomoo.py \\
    --alpaca-log artifacts/logs/gap_trade_20260417_093200.jsonl \\
    --moomoo-json artifacts/gap_scan_moomoo/moomoo_932_scan_2026-04-17_....json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Alpaca log loader
# ---------------------------------------------------------------------------

def _load_alpaca_candidates(log_path: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (diagnostics, candidates) from a gap_trade JSONL log."""
    diag: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(obj.get("event") or obj.get("message") or "")
            if "scanner_diagnostics" in event:
                diag = obj
            elif event == "scanner_complete":
                # candidates embedded in scanner_complete
                for c in obj.get("candidates", []):
                    if isinstance(c, dict):
                        candidates.append(c)
            elif "gap_candidate" in event or obj.get("gap_pct") is not None:
                candidates.append(obj)
    return diag, candidates


def _find_alpaca_log(date_str: str) -> Optional[Path]:
    logs_dir = ROOT / "artifacts" / "logs"
    date_compact = date_str.replace("-", "")
    candidates = sorted(logs_dir.glob(f"gap_trade_{date_compact}*.jsonl"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# moomoo JSON loader
# ---------------------------------------------------------------------------

def _load_moomoo_result(json_path: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    diag = data.get("diagnostics")
    candidates = data.get("candidates", [])
    return diag, candidates


def _find_moomoo_json(date_str: str) -> Optional[Path]:
    scan_dir = ROOT / "artifacts" / "gap_scan_moomoo"
    candidates = sorted(scan_dir.glob(f"moomoo_932_scan_{date_str}_*.json"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _sym_set(candidates: List[Dict[str, Any]], key: str = "symbol") -> set:
    return {str(c.get(key, "")).upper() for c in candidates}


def print_comparison(
    date_str: str,
    alpaca_diag: Optional[Dict[str, Any]],
    alpaca_cands: List[Dict[str, Any]],
    moomoo_diag: Optional[Dict[str, Any]],
    moomoo_cands: List[Dict[str, Any]],
) -> None:
    ap_syms = _sym_set(alpaca_cands)
    mm_syms = _sym_set(moomoo_cands)
    both = ap_syms & mm_syms
    ap_only = ap_syms - mm_syms
    mm_only = mm_syms - ap_syms

    print(f"\n{'='*65}")
    print(f"=== Alpaca vs moomoo-9:32 gap scan comparison | {date_str} ===")
    print(f"{'='*65}")

    # --- diagnostics ---
    print("\n[1] funnel 比較")
    print(f"  {'指標':35s} {'Alpaca':>10} {'moomoo-9:32':>12}")
    print("  " + "-" * 60)

    def ap(k): return str(alpaca_diag.get(k, "n/a")) if alpaca_diag else "n/a"
    def mm(k): return str(moomoo_diag.get(k, "n/a")) if moomoo_diag else "n/a"

    rows = [
        ("universe_count",          "universe_count",          "universe_count"),
        ("daily_stats",             "daily_count",             "daily_stats_count"),
        ("snapshot/open OK",        "open_count",              "snapshot_ok_count"),
        ("kline OK",                "n/a",                     "kline_0931_ok_count"),
        ("missing open/snapshot",   "missing_open_count",      "missing_snapshot_count"),
        ("price_filter_drop",       "price_filter_drop_count", "price_filter_drop_count"),
        ("gap_filter_drop",         "gap_filter_drop_count",   "gap_filter_drop_count"),
        ("volume_filter_drop",      "pace_filter_drop_count",  "volume_filter_drop_count"),
        ("market_cap_drop",         "market_cap_drop_count",   "market_cap_drop_count"),
        ("final_candidates",        "final_candidate_count",   "final_candidate_count"),
    ]
    for label, ak, mk in rows:
        av = ap(ak) if ak != "n/a" else "n/a"
        mv = mm(mk) if mk != "n/a" else "n/a"
        print(f"  {label:35s} {av:>10} {mv:>12}")

    # --- candidate overlap ---
    print(f"\n[2] 候補一致")
    print(f"  Alpaca candidates : {sorted(ap_syms)}")
    print(f"  moomoo candidates : {sorted(mm_syms)}")
    print(f"  both              : {sorted(both)}")
    print(f"  alpaca only       : {sorted(ap_only)}")
    print(f"  moomoo only       : {sorted(mm_only)}")

    # --- per-symbol detail ---
    all_syms = sorted(ap_syms | mm_syms)
    if all_syms:
        print(f"\n[3] 銘柄別詳細")
        hdr = f"  {'sym':8s} {'ap_gap%':>8} {'mm_gap%':>8} {'ap_pace':>8} {'mm_pace_x':>10} {'in_both':>8}"
        print(hdr)
        print("  " + "-" * 55)
        ap_map = {str(c.get("symbol","")).upper(): c for c in alpaca_cands}
        mm_map = {str(c.get("symbol","")).upper(): c for c in moomoo_cands}
        for sym in all_syms:
            ac = ap_map.get(sym, {})
            mc = mm_map.get(sym, {})
            ap_gap = f"{float(ac['gap_pct']):.2f}" if ac.get("gap_pct") is not None else "-"
            mm_gap = f"{float(mc['gap_pct']):.2f}" if mc.get("gap_pct") is not None else "-"
            ap_pace = f"{float(ac.get('volume_pace_ratio', ac.get('daily_volume_pace', 0))):.2f}" if ac else "-"
            mm_pace = f"{float(mc['volume_pace_annualized']):.2f}" if mc.get("volume_pace_annualized") is not None else "-"
            in_both = "✓" if sym in both else ""
            print(f"  {sym:8s} {ap_gap:>8} {mm_gap:>8} {ap_pace:>8} {mm_pace:>10} {in_both:>8}")

    print(f"\n[4] 判定")
    overlap_rate = len(both) / max(len(ap_syms | mm_syms), 1) * 100
    print(f"  候補一致率: {len(both)}/{len(ap_syms | mm_syms)} ({overlap_rate:.1f}%)")
    if overlap_rate >= 70:
        print("  → 高い一致。moomoo 9:32版は Alpaca 結果とほぼ同等の候補を検出。")
    elif overlap_rate >= 30:
        print("  → 部分一致。銘柄差異あり。データ源の違いまたは timing 差を要確認。")
    else:
        print("  → 低一致。設定・データ源・timing 差が大きい可能性。")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2026-04-17")
    p.add_argument("--alpaca-log", type=Path, default=None)
    p.add_argument("--moomoo-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    args = _parse_args()

    # alpaca log
    ap_path = args.alpaca_log or _find_alpaca_log(args.date)
    if ap_path is None or not ap_path.exists():
        print(f"[WARN] Alpaca log not found for {args.date}. Run Alpaca scanner first.", file=sys.stderr)
        alpaca_diag, alpaca_cands = None, []
    else:
        print(f"[alpaca] {ap_path}")
        alpaca_diag, alpaca_cands = _load_alpaca_candidates(ap_path)

    # moomoo json
    mm_path = args.moomoo_json or _find_moomoo_json(args.date)
    if mm_path is None or not mm_path.exists():
        print(f"[WARN] moomoo scan JSON not found for {args.date}. Run run_gap_scan_moomoo_932.py first.", file=sys.stderr)
        moomoo_diag, moomoo_cands = None, []
    else:
        print(f"[moomoo] {mm_path}")
        moomoo_diag, moomoo_cands = _load_moomoo_result(mm_path)

    print_comparison(args.date, alpaca_diag, alpaca_cands, moomoo_diag, moomoo_cands)


if __name__ == "__main__":
    main()
