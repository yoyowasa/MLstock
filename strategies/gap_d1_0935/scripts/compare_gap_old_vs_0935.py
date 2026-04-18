from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRATEGY_SRC = Path(__file__).resolve().parents[1] / "src"
for path in [str(ROOT / "src"), str(STRATEGY_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from gap_d1_0935.compare_gap_old_vs_0935 import compare_gap_old_vs_0935  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-logs-dir", type=Path, default=ROOT / "artifacts" / "logs")
    args = parser.parse_args()
    result = compare_gap_old_vs_0935(root_logs_dir=args.root_logs_dir)
    print(f"rows={result.rows} report={result.report_path} summary={result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
