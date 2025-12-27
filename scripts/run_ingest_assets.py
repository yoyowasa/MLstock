from __future__ import annotations

from mlstock.config.loader import load_config
from mlstock.jobs import ingest_assets


def main() -> None:
    cfg = load_config()
    ingest_assets.run(cfg)


if __name__ == "__main__":
    main()
