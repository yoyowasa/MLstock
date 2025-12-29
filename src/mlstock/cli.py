from __future__ import annotations

import argparse

from mlstock.config.loader import load_config
from mlstock.jobs import seed_symbols


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlstock")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("make-seed", help="Create seed symbol universe")
    seed_parser.add_argument("--n-seed", type=int, default=None, help="Number of seed symbols to keep")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config()

    if args.command == "make-seed":
        seed_symbols.run(cfg, n_seed=args.n_seed)


if __name__ == "__main__":
    main()
