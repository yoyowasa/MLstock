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

from gap_d1_0935.replay_gap_0935_watchlist import replay_gap_0935_watchlist  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date", type=date.fromisoformat, required=True)
    parser.add_argument("--rebuild-watchlist", action="store_true")
    args = parser.parse_args()
    result = replay_gap_0935_watchlist(trade_date=args.trade_date, rebuild_watchlist=args.rebuild_watchlist)
    print(f"trade_date={result.trade_date} candidates={result.final_candidate_count} csv={result.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
