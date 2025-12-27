from __future__ import annotations

from mlstock.config.loader import load_config
from mlstock.jobs import ingest_calendar


def main() -> None:
    cfg = load_config()
    ingest_calendar.run(cfg)


if __name__ == "__main__":
    main()
