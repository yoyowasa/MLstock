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

from gap_d1_0935.analysis_phase1 import analyze_phase1_population  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    result = analyze_phase1_population(months=args.months, end_date=args.end_date)
    print(f"period={result.start_date}..{result.end_date} trade_days={result.trade_days}")
    print(f"summary={result.summary_path}")
    print(f"daily_counts={result.daily_counts_path}")
    print(f"drop_counts={result.daily_drop_counts_path}")
    print(f"symbol_detail={result.symbol_detail_path}")
    print(f"missing_first5_detail={result.missing_first5_detail_path}")
    print(f"missing_first5_daily={result.missing_first5_daily_path}")
    print(f"missing_first5_symbol={result.missing_first5_symbol_path}")
    print(f"coverage_type={result.coverage_by_type_path}")
    print(f"coverage_exchange={result.coverage_by_exchange_path}")
    print(f"coverage_market_cap={result.coverage_by_market_cap_path}")
    print(f"compare={result.compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
