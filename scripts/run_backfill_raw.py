from __future__ import annotations

import argparse
import re
from typing import List, Optional

from mlstock.config.loader import load_config
from mlstock.jobs import ingest_bars, ingest_corp_actions


def _parse_symbols(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    return parts or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None, help="Comma or space separated symbols")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)

    cfg = load_config()
    ingest_bars.backfill(cfg, symbols=symbols)
    ingest_corp_actions.backfill(cfg, symbols=symbols)


if __name__ == "__main__":
    main()
