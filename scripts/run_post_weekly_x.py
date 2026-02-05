from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_orders_dir
from mlstock.data.storage.state import read_state


DEFAULT_HEADER = "MLStock weekly screening"


def _latest_selection_file(orders_dir: Path) -> Path:
    selection_files = sorted(orders_dir.glob("selection_*.json"))
    if not selection_files:
        raise FileNotFoundError(f"selection_*.json not found in {orders_dir}")
    return selection_files[-1]


def _ensure_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _format_symbols(label: str, symbols: List[str], max_count: int, include_empty: bool) -> Optional[str]:
    total = len(symbols)
    if total == 0:
        return f"{label}(0): -" if include_empty else None
    shown = symbols[: max(0, max_count)]
    remain = total - len(shown)
    text = f"{label}({total}): {', '.join(shown)}"
    if remain > 0:
        text += f" +{remain}"
    return text


def _build_text(
    payload: dict,
    header: str,
    footer: str,
    max_buy: int,
    max_sell: int,
    max_keep: int,
    max_chars: int,
    include_empty: bool,
) -> Tuple[str, int]:
    buy_symbols = _ensure_list(payload.get("buy_symbols"))
    sell_symbols = _ensure_list(payload.get("sell_symbols"))
    keep_symbols = _ensure_list(payload.get("keep_symbols"))

    week_start = payload.get("week_start")
    as_of = payload.get("as_of")

    drop_header = False
    drop_footer = False
    drop_date = False

    while True:
        lines: List[str] = []
        if header and not drop_header:
            lines.append(header)
        if not drop_date and (week_start or as_of):
            parts = []
            if week_start:
                parts.append(f"week_start: {week_start}")
            if as_of:
                parts.append(f"as_of: {as_of}")
            lines.append(" / ".join(parts))

        for line in [
            _format_symbols("BUY", buy_symbols, max_buy, include_empty),
            _format_symbols("SELL", sell_symbols, max_sell, include_empty),
            _format_symbols("KEEP", keep_symbols, max_keep, include_empty),
        ]:
            if line:
                lines.append(line)

        if footer and not drop_footer:
            lines.append(footer)

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text, len(text)

        reduce_targets = [
            ("buy", max_buy, len(buy_symbols)),
            ("sell", max_sell, len(sell_symbols)),
            ("keep", max_keep, len(keep_symbols)),
        ]
        reduce_targets = [item for item in reduce_targets if item[1] > 0 and item[2] > 0]
        if reduce_targets:
            reduce_targets.sort(key=lambda item: (min(item[1], item[2]), item[2]), reverse=True)
            target = reduce_targets[0][0]
            if target == "buy":
                max_buy -= 1
            elif target == "sell":
                max_sell -= 1
            else:
                max_keep -= 1
            continue

        if footer and not drop_footer:
            drop_footer = True
            continue
        if header and not drop_header:
            drop_header = True
            continue
        if (week_start or as_of) and not drop_date:
            drop_date = True
            continue

        truncated = text[: max(0, max_chars - 3)]
        return truncated + "...", len(truncated) + 3


def _read_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing env: {name}")
    return value


def _post_to_x(text: str) -> object:
    import tweepy

    client = tweepy.Client(
        consumer_key=_read_env("X_API_KEY"),
        consumer_secret=_read_env("X_API_KEY_SECRET"),
        access_token=_read_env("X_ACCESS_TOKEN"),
        access_token_secret=_read_env("X_ACCESS_TOKEN_SECRET"),
        wait_on_rate_limit=True,
    )
    return client.create_tweet(text=text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", type=str, default=None)
    parser.add_argument("--orders-dir", type=str, default=None)
    parser.add_argument("--max-buy", type=int, default=10)
    parser.add_argument("--max-sell", type=int, default=10)
    parser.add_argument("--max-keep", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument("--header", type=str, default=DEFAULT_HEADER)
    parser.add_argument("--footer", type=str, default="")
    parser.add_argument("--include-empty", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    load_dotenv()

    cfg = load_config()
    orders_dir = Path(args.orders_dir) if args.orders_dir else artifacts_orders_dir(cfg)
    selection_path = Path(args.selection_file) if args.selection_file else _latest_selection_file(orders_dir)

    payload = read_state(selection_path)
    if not payload:
        raise SystemExit(f"Selection file is empty: {selection_path}")

    text, length = _build_text(
        payload=payload,
        header=args.header,
        footer=args.footer,
        max_buy=args.max_buy,
        max_sell=args.max_sell,
        max_keep=args.max_keep,
        max_chars=args.max_chars,
        include_empty=args.include_empty,
    )

    print(f"selection: {selection_path}")
    print(f"length: {length}")
    print(text)

    if args.dry_run:
        return

    response = _post_to_x(text)
    print(f"posted: {response.data}")


if __name__ == "__main__":
    main()
