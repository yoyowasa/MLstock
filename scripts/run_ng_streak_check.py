from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_dir
from mlstock.data.storage.state import read_state


def _parse_date(value: object) -> datetime.date | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _week_bounds(as_of: datetime.date) -> Tuple[datetime.date, datetime.date]:
    monday = as_of - timedelta(days=as_of.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _read_orders(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    if path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_weekly_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _find_file(bundle_dir: Path, pattern: str) -> Path | None:
    matches = sorted(bundle_dir.glob(pattern))
    return matches[-1] if matches else None


def _check_bundle(bundle_dir: Path, max_positions: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    selection_path = _find_file(bundle_dir, "selection_*.json")
    orders_path = _find_file(bundle_dir, "orders_*.csv")
    weekly_log_path = _find_file(bundle_dir, "weekly_*.jsonl")
    portfolio_path = bundle_dir / "portfolio.json"

    if selection_path is None or not selection_path.exists():
        errors.append("selection_missing")
        return False, errors
    if orders_path is None or not orders_path.exists():
        errors.append("orders_missing")
        return False, errors
    if weekly_log_path is None or not weekly_log_path.exists():
        errors.append("weekly_log_missing")
        return False, errors
    if not portfolio_path.exists():
        errors.append("portfolio_missing")
        return False, errors

    selection = read_state(selection_path)
    portfolio = read_state(portfolio_path)
    orders = _read_orders(orders_path)
    weekly_log = _read_weekly_log(weekly_log_path)

    buy = [s for s in selection.get("buy_symbols") or [] if s]
    sell = [s for s in selection.get("sell_symbols") or [] if s]
    keep = [s for s in selection.get("keep_symbols") or [] if s]
    n_selected = int(selection.get("n_selected") or 0)

    stamp = None
    name = selection_path.name
    if name.startswith("selection_") and name.endswith(".json"):
        stamp = name.replace("selection_", "").replace(".json", "")
        if stamp and len(stamp) == 8 and stamp.isdigit():
            preferred_orders = bundle_dir / f"orders_{stamp}.csv"
            if preferred_orders.exists():
                orders_path = preferred_orders
            preferred_logs = sorted(bundle_dir.glob(f"weekly_{stamp}_*.jsonl"))
            if preferred_logs:
                weekly_log_path = preferred_logs[-1]
    if orders_path and "orders_candidates" in orders_path.name:
        candidates = sorted(
            path
            for path in bundle_dir.glob("orders_*.csv")
            if "orders_candidates" not in path.name
        )
        if candidates:
            orders_path = candidates[-1]
    as_of_date = _parse_date(selection.get("as_of"))
    week_start_date = _parse_date(selection.get("week_start"))
    stamp_date = None
    if stamp and len(stamp) == 8 and stamp.isdigit():
        stamp_date = datetime.strptime(stamp, "%Y%m%d").date()
    if not (as_of_date and stamp_date and as_of_date == stamp_date):
        errors.append("selection_as_of_stamp_mismatch")

    if as_of_date and week_start_date:
        monday, sunday = _week_bounds(as_of_date)
        if not (monday <= week_start_date <= sunday):
            errors.append("selection_week_start_not_same_week")
    else:
        errors.append("selection_week_start_missing")

    features_date = _parse_date(selection.get("data_max_features_date"))
    labels_date = _parse_date(selection.get("data_max_labels_date"))
    week_map_date = _parse_date(selection.get("data_max_week_map_date"))
    if week_start_date and features_date and week_map_date:
        labels_ok = False
        if labels_date:
            labels_ok = labels_date >= (week_start_date - timedelta(days=7))
        if not (features_date >= week_start_date and week_map_date >= week_start_date and labels_ok):
            errors.append("selection_data_max_old")
    else:
        errors.append("selection_data_max_missing")

    if n_selected != len(buy):
        errors.append("selection_n_selected_mismatch")

    overlap = set(buy) & (set(sell) | set(keep))
    overlap |= set(sell) & set(keep)
    if overlap:
        errors.append("selection_overlap")

    if not orders:
        if not (len(buy) == 0 and len(sell) == 0):
            errors.append("orders_empty_mismatch")
    else:
        for row in orders:
            side = (row.get("side") or "").lower()
            symbol = (row.get("symbol") or "").upper()
            qty_raw = row.get("qty")
            try:
                qty_val = int(qty_raw)
            except (TypeError, ValueError):
                qty_val = 0
            if side not in ("buy", "sell"):
                errors.append("orders_side_invalid")
            if qty_val < 1:
                errors.append("orders_qty_invalid")
            if side == "buy" and symbol not in buy:
                errors.append("orders_buy_symbol_mismatch")
            if side == "sell" and symbol not in sell:
                errors.append("orders_sell_symbol_mismatch")

    if not (
        portfolio.get("as_of")
        and portfolio.get("week_start")
        and selection.get("as_of")
        and selection.get("week_start")
        and portfolio.get("as_of") == selection.get("as_of")
        and portfolio.get("week_start") == selection.get("week_start")
    ):
        errors.append("portfolio_date_mismatch")

    cash_usd = portfolio.get("cash_usd")
    try:
        cash_ok = float(cash_usd) >= 0
    except (TypeError, ValueError):
        cash_ok = False
    if not cash_ok:
        errors.append("portfolio_cash_invalid")

    positions = portfolio.get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}
    pos_symbols = list(positions.keys())
    if len(pos_symbols) > max_positions:
        errors.append("portfolio_positions_over_max")
    for symbol, qty in positions.items():
        try:
            qty_val = int(qty)
        except (TypeError, ValueError):
            qty_val = 0
        if qty_val < 1:
            errors.append("portfolio_qty_invalid")
            break

    expected = sorted(set(keep + buy))
    actual = sorted(set(pos_symbols))
    if expected != actual:
        errors.append("compare_positions_mismatch")
    sell_in_portfolio = [s for s in sell if s in actual]
    if sell_in_portfolio:
        errors.append("compare_sell_in_portfolio")

    cash_after_exec = selection.get("cash_after_exec")
    try:
        cash_diff = abs(float(cash_after_exec) - float(cash_usd))
    except (TypeError, ValueError):
        cash_diff = None
    if cash_diff is None or cash_diff > 0.01:
        errors.append("compare_cash_mismatch")

    has_error = False
    has_complete = False
    has_validate = False
    for entry in weekly_log:
        if entry.get("level") == "ERROR":
            has_error = True
        if entry.get("logger") == "weekly" and entry.get("message") == "complete":
            has_complete = True
        message = str(entry.get("message") or "")
        if "validate" in message or "validation_failed" in message:
            has_validate = True
    if has_error or not has_complete or has_validate:
        errors.append("weekly_log_invalid")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=15)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--exit-nonzero", action="store_true", default=False)
    args = parser.parse_args()

    cfg = load_config()
    bundle_root = Path(args.bundle_root) if args.bundle_root else artifacts_dir(cfg) / "weekly_bundle"
    if not bundle_root.exists():
        print(f"bundle_root not found: {bundle_root}")
        sys.exit(1)

    bundle_dirs = [
        d for d in bundle_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    ]
    bundle_dirs = sorted(bundle_dirs, key=lambda p: p.name)
    if args.limit and args.limit > 0:
        bundle_dirs = bundle_dirs[-args.limit :]

    if not bundle_dirs:
        print("no bundles found")
        sys.exit(1)

    results = []
    for bundle_dir in bundle_dirs:
        ok, errors = _check_bundle(bundle_dir, args.max_positions)
        results.append({"bundle": bundle_dir.name, "ok": ok, "errors": errors})

    ng_streak = 0
    for item in reversed(results):
        if item["ok"]:
            break
        ng_streak += 1
    trigger = ng_streak >= max(1, args.threshold)

    for item in results:
        status = "OK" if item["ok"] else "NG"
        line = f"{item['bundle']}: {status}"
        if not item["ok"] and item["errors"]:
            line = f"{line} ({', '.join(sorted(set(item['errors'])))})"
        print(line)

    print(f"ng_streak={ng_streak} threshold={args.threshold} trigger_liquidation={trigger}")

    if args.exit_nonzero and trigger:
        sys.exit(1)


if __name__ == "__main__":
    main()
