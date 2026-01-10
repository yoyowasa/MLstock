from __future__ import annotations

from mlstock.config.loader import load_config
from mlstock.jobs import build_snapshots, ingest_bars, ingest_calendar, ingest_corp_actions, weekly


def main() -> None:
    cfg = load_config()
    ingest_calendar.run(cfg)
    ingest_bars.incremental_update(cfg)
    ingest_corp_actions.incremental_update(cfg)
    build_snapshots.run(cfg)
    weekly.run(cfg)


if __name__ == "__main__":
    main()
