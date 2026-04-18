from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import reference_seed_symbols_path


ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')


def load_root_config():
    return load_config()


def load_alpaca_client() -> tuple[Any, AlpacaClient]:
    cfg = load_root_config()
    return cfg, AlpacaClient.from_env(base_url=cfg.alpaca.data_base_url)


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def fetch_bars_batch(
    client: AlpacaClient,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: str,
    adjustment: str,
    asof: Optional[str] = None,
    limit: int = 10000,
) -> Dict[str, List[Dict[str, Any]]]:
    collected: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in symbols}
    page_token: Optional[str] = None
    while True:
        response = client.get_bars(
            symbols=list(symbols),
            start=iso_utc(start),
            end=iso_utc(end),
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            asof=asof,
            limit=limit,
            page_token=page_token,
        )
        bars = response.get('bars') if isinstance(response, dict) else None
        if isinstance(bars, dict):
            for symbol, items in bars.items():
                key = str(symbol).upper()
                if key in collected:
                    collected[key].extend([item for item in items if isinstance(item, dict)])
        elif isinstance(bars, list) and len(symbols) == 1:
            collected[str(symbols[0]).upper()].extend([item for item in bars if isinstance(item, dict)])
        page_token = response.get('next_page_token') if isinstance(response, dict) else None
        if not page_token:
            break
    return collected


def load_seed_symbols() -> List[str]:
    cfg = load_root_config()
    seed_path = reference_seed_symbols_path(cfg)
    seed_df = read_parquet(seed_path)
    symbols = seed_df['symbol'].dropna().astype(str).str.strip().str.upper().tolist()
    return sorted(set(symbols))


def get_trading_days(client: AlpacaClient, center_date: date, days_before: int = 10, days_after: int = 10) -> List[date]:
    start = center_date - timedelta(days=days_before)
    end = center_date + timedelta(days=days_after)
    try:
        calendar = client.get_calendar(start=start.isoformat(), end=end.isoformat())
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
        cfg = load_root_config()
        trading_client = AlpacaClient.from_env(base_url=cfg.alpaca.trading_base_url)
        calendar = trading_client.get_calendar(start=start.isoformat(), end=end.isoformat())
    dates: List[date] = []
    for item in calendar if isinstance(calendar, list) else []:
        raw = item.get('date')
        if raw:
            dates.append(date.fromisoformat(str(raw)))
    return sorted(set(dates))


def get_previous_trading_day(client: AlpacaClient, trade_date: date) -> date:
    dates = [d for d in get_trading_days(client, trade_date, days_before=15, days_after=0) if d < trade_date]
    if not dates:
        raise ValueError(f'No previous trading day found for {trade_date}')
    return dates[-1]


def get_next_trading_day(client: AlpacaClient, anchor_date: date) -> date:
    dates = [d for d in get_trading_days(client, anchor_date, days_before=0, days_after=15) if d > anchor_date]
    if not dates:
        raise ValueError(f'No next trading day found after {anchor_date}')
    return dates[0]


def infer_trade_date(client: AlpacaClient, requested: date | None = None) -> date:
    if requested is not None:
        return requested
    now_et = datetime.now(ET).date()
    return get_next_trading_day(client, now_et)


def to_local_ts(raw: Any) -> pd.Timestamp:
    return pd.to_datetime(raw, utc=True).tz_convert(ET)
