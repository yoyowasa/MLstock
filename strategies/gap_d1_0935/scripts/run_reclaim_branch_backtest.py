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

from gap_d1_0935.reclaim_branch_backtest import backtest_reclaim_branch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--fee-bps-round-trip", type=float, default=2.0)
    args = parser.parse_args()
    result = backtest_reclaim_branch(
        months=args.months,
        end_date=args.end_date,
        slippage_bps_per_side=args.slippage_bps_per_side,
        fee_bps_round_trip=args.fee_bps_round_trip,
    )
    print(f"period={result.start_date}..{result.end_date} trade_days={result.trade_days}")
    print(f"trades={result.trades_path}")
    print(f"summary={result.summary_path}")
    print(f"compare={result.compare_path}")
    print(f"decomposition={result.decomposition_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
