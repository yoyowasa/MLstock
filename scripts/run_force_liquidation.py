from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_orders_dir, artifacts_state_dir
from mlstock.data.storage.state import read_state, write_state


DEFAULT_COLUMNS = [
    "side",
    "symbol",
    "qty",
    "type",
    "time_in_force",
    "priority",
    "est_price",
    "est_cost",
    "required_est",
]


def _normalize_positions(state: Dict[str, object]) -> Dict[str, int]:
    positions = state.get("positions")
    if not isinstance(positions, dict):
        return {}
    normalized: Dict[str, int] = {}
    for symbol, qty in positions.items():
        try:
            qty_int = int(qty)
        except (TypeError, ValueError):
            continue
        if qty_int <= 0:
            continue
        symbol_key = str(symbol).upper()
        normalized[symbol_key] = normalized.get(symbol_key, 0) + qty_int
    return normalized


def _resolve_stamp(stamp: str | None, tz_name: str) -> str:
    if stamp:
        if len(stamp) != 8 or not stamp.isdigit():
            raise ValueError("stamp must be YYYYMMDD")
        return stamp
    today = datetime.now(ZoneInfo(tz_name)).date()
    return today.strftime("%Y%m%d")


def _parse_date(value: str) -> datetime.date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise ValueError("as_of must be YYYY-MM-DD") from exc


def _week_start(value: datetime.date) -> datetime.date:
    return value - timedelta(days=value.weekday())


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument("--time-in-force", type=str, default="day")
    parser.add_argument("--update-portfolio", action="store_true", default=False)
    parser.add_argument("--cash-usd", type=float, default=None)
    parser.add_argument("--as-of", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    portfolio_path = Path(args.portfolio) if args.portfolio else artifacts_state_dir(cfg) / "portfolio.json"
    state = read_state(portfolio_path)
    positions = _normalize_positions(state)

    stamp = _resolve_stamp(args.stamp, cfg.project.timezone)
    orders_dir = artifacts_orders_dir(cfg)
    output_path = Path(args.output) if args.output else orders_dir / f"orders_force_{stamp}.csv"

    rows: list[dict[str, object]] = []
    for symbol in sorted(positions.keys()):
        rows.append(
            {
                "side": "sell",
                "symbol": symbol,
                "qty": positions[symbol],
                "type": "market",
                "time_in_force": args.time_in_force,
                "priority": "",
                "est_price": "",
                "est_cost": "",
                "required_est": "",
            }
        )

    _write_csv(rows, output_path)

    if args.update_portfolio:
        if args.as_of:
            as_of_date = _parse_date(args.as_of)
        elif args.stamp:
            as_of_date = datetime.strptime(stamp, "%Y%m%d").date()
        else:
            as_of_date = datetime.now(ZoneInfo(cfg.project.timezone)).date()
        week_start = _week_start(as_of_date)
        cash_usd = state.get("cash_usd")
        if args.cash_usd is not None:
            cash_usd = float(args.cash_usd)
        if cash_usd is None:
            cash_usd = 0.0
        next_state = {
            "as_of": as_of_date.isoformat(),
            "week_start": week_start.isoformat(),
            "cash_usd": float(cash_usd),
            "positions": {},
        }
        write_state(next_state, portfolio_path)
        print(f"updated: {portfolio_path}")

    if rows:
        print(f"saved: {output_path} (sell_orders={len(rows)})")
    else:
        print(f"saved: {output_path} (no positions)")


if __name__ == "__main__":
    main()
