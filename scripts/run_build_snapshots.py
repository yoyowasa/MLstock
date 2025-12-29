from __future__ import annotations

from mlstock.config.loader import load_config
from mlstock.jobs import build_snapshots


def main() -> None:
    cfg = load_config()
    build_snapshots.run(cfg)


if __name__ == "__main__":
    main()
