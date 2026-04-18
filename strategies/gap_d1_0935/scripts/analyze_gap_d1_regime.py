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

from gap_d1_0935.analysis_regime import analyze_watchlist_regime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    result = analyze_watchlist_regime(months=args.months, end_date=args.end_date)
    print(f"period={result.start_date}..{result.end_date} trade_days={result.trade_days}")
    print(f"detail={result.detail_path}")
    print(f"summary={result.summary_path}")
    print(f"branch_compare={result.branch_compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
