from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRATEGY_SRC = Path(__file__).resolve().parents[1] / "src"
for path in [str(ROOT / "src"), str(STRATEGY_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from gap_d1_0935.reclaim_executor import (  # noqa: E402
    SUPPORTED_BRANCHES,
    replay_reclaim_executor_period,
    run_reclaim_executor,
)


def _parse_trade_date_from_scanner_csv(path: Path) -> date:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"Trade date not found in scanner csv filename: {path}")
    tail = digits[-8:]
    return date.fromisoformat(f"{tail[:4]}-{tail[4:6]}-{tail[6:8]}")


def _scanner_csv_map(from_scanner_csv: str) -> dict[date, Path]:
    path = Path(from_scanner_csv)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.glob("gap_0935_candidates_*.csv"))
    else:
        candidates = sorted(path.parent.glob(path.name))
    mapping: dict[date, Path] = {}
    for item in candidates:
        mapping[_parse_trade_date_from_scanner_csv(item)] = item
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", type=date.fromisoformat, default=None)
    parser.add_argument("--branch", choices=sorted(SUPPORTED_BRANCHES | {"all"}), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-scanner-csv", default=None)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--fee-bps-round-trip", type=float, default=2.0)
    args = parser.parse_args()

    branches = sorted(SUPPORTED_BRANCHES) if args.branch == "all" else [args.branch]
    if args.months or args.end_date or args.branch == "all":
        actual_end_date = args.end_date or args.trade_date
        if actual_end_date is None:
            raise ValueError("trade-date or end-date is required for period replay")
        result = replay_reclaim_executor_period(
            branches=branches,
            months=args.months or 12,
            end_date=actual_end_date,
            scanner_csv_map=_scanner_csv_map(args.from_scanner_csv) if args.from_scanner_csv else None,
            slippage_bps_per_side=args.slippage_bps_per_side,
            fee_bps_round_trip=args.fee_bps_round_trip,
        )
        print(f"period={result.start_date}..{result.end_date} trade_days={result.trade_days}")
        print(f"branches={','.join(branches)}")
        print(f"compare={result.compare_path}")
        print(f"daily_report={result.daily_report_path}")
        print(f"summary={result.summary_path}")
        return 0

    if args.trade_date is None:
        raise ValueError("--trade-date is required for single-day mode")
    result = run_reclaim_executor(
        trade_date=args.trade_date,
        branch=branches[0],
        dry_run=args.dry_run or True,
        from_scanner_csv=args.from_scanner_csv,
        slippage_bps_per_side=args.slippage_bps_per_side,
        fee_bps_round_trip=args.fee_bps_round_trip,
    )
    print(f"trade_date={result.trade_date}")
    print(f"log={result.log_path}")
    print(f"daily_report={result.daily_report_path}")
    print(f"branch_compare={result.branch_compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
