from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import raw_bars_path, reference_seed_symbols_path
from mlstock.logging.logger import build_log_path, log_event, setup_logger


BAR_COLUMNS = [
    "symbol",
    "timestamp_utc",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
]


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _normalize_bars(records: List[dict], symbol: str, tz: ZoneInfo) -> pd.DataFrame:
    rows = []
    for item in records:
        ts_raw = item.get("t")
        if not ts_raw:
            continue
        ts = pd.to_datetime(ts_raw, utc=True)
        ts_local = ts.tz_convert(tz)
        rows.append(
            {
                "symbol": symbol,
                "timestamp_utc": ts.to_pydatetime(),
                "date": ts_local.date(),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "trade_count": item.get("n"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _merge_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    for column in BAR_COLUMNS:
        if column not in merged.columns:
            merged[column] = None
    return merged[BAR_COLUMNS]


def _fetch_bars_batch(
    client: AlpacaClient,
    symbols: List[str],
    start: str,
    end: str,
    cfg: AppConfig,
) -> Dict[str, List[dict]]:
    collected: Dict[str, List[dict]] = {symbol: [] for symbol in symbols}
    page_token: Optional[str] = None
    while True:
        response = client.get_bars(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=cfg.bars.timeframe,
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
            page_token=page_token,
        )
        bars = response.get("bars") if isinstance(response, dict) else None
        if isinstance(bars, list):
            target = symbols[0] if symbols else None
            if target:
                collected[target].extend(bars)
        elif isinstance(bars, dict):
            for symbol, items in bars.items():
                if symbol not in collected:
                    collected[symbol] = []
                if items:
                    collected[symbol].extend(items)
        page_token = response.get("next_page_token") if isinstance(response, dict) else None
        if not page_token:
            break
    return collected


def _ingest_batch(
    cfg: AppConfig,
    symbols: List[str],
    start: date,
    end: date,
    incremental: bool,
    tz: ZoneInfo,
) -> Dict[str, int]:
    client = AlpacaClient.from_env(cfg.alpaca.data_base_url)
    collected = _fetch_bars_batch(
        client,
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        cfg=cfg,
    )
    counts: Dict[str, int] = {}
    for symbol, items in collected.items():
        df_new = _normalize_bars(items, symbol, tz)
        if df_new.empty and incremental:
            continue
        path = raw_bars_path(cfg, symbol)
        if incremental and path.exists():
            existing = read_parquet(path)
            df_new = _merge_bars(existing, df_new)
        else:
            for column in BAR_COLUMNS:
                if column not in df_new.columns:
                    df_new[column] = None
            df_new = df_new[BAR_COLUMNS]
        if df_new.empty:
            continue
        write_parquet_atomic(df_new, path)
        counts[symbol] = int(len(df_new))
    return counts


def _load_seed_symbols(cfg: AppConfig) -> List[str]:
    seed_path = reference_seed_symbols_path(cfg)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed symbols not found: {seed_path}. Run: python -m mlstock make-seed")
    seed_df = read_parquet(seed_path)
    symbols = seed_df["symbol"].dropna().astype(str).tolist()
    return symbols


def backfill(cfg: AppConfig, symbols: Optional[List[str]] = None) -> None:
    log_path = build_log_path(cfg, "ingest_bars_backfill")
    logger = setup_logger("ingest_bars_backfill", log_path, cfg.logging.level)
    log_event(logger, "start")

    tz = ZoneInfo(cfg.project.timezone)
    start = date.fromisoformat(cfg.bars.backfill_start)
    end = datetime.now(tz).date()
    symbols = symbols or _load_seed_symbols(cfg)

    if not symbols:
        raise ValueError("No seed symbols available for bars backfill")

    batch_size = cfg.bars.batch_size if cfg.bars.mode == "multi_symbol" else 1
    max_workers = max(int(cfg.bars.max_workers), 1)

    total_rows = 0
    if max_workers > 1 and len(symbols) > batch_size:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_ingest_batch, cfg, batch, start, end, False, tz)
                for batch in _chunk(symbols, batch_size)
            ]
            for future in as_completed(futures):
                counts = future.result()
                batch_rows = sum(counts.values())
                total_rows += batch_rows
                log_event(logger, "batch_complete", rows=batch_rows)
    else:
        for batch in _chunk(symbols, batch_size):
            counts = _ingest_batch(cfg, batch, start, end, False, tz)
            batch_rows = sum(counts.values())
            total_rows += batch_rows
            log_event(logger, "batch_complete", rows=batch_rows)

    log_event(logger, "complete", symbols=len(symbols), rows=total_rows)


def incremental_update(cfg: AppConfig, symbols: Optional[List[str]] = None) -> None:
    log_path = build_log_path(cfg, "ingest_bars_incremental")
    logger = setup_logger("ingest_bars_incremental", log_path, cfg.logging.level)
    log_event(logger, "start")

    tz = ZoneInfo(cfg.project.timezone)
    end = datetime.now(tz).date()
    start = end - timedelta(days=int(cfg.bars.lookback_days))
    symbols = symbols or _load_seed_symbols(cfg)

    if not symbols:
        raise ValueError("No seed symbols available for bars incremental update")

    batch_size = cfg.bars.batch_size if cfg.bars.mode == "multi_symbol" else 1
    max_workers = max(int(cfg.bars.max_workers), 1)

    total_rows = 0
    if max_workers > 1 and len(symbols) > batch_size:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_ingest_batch, cfg, batch, start, end, True, tz)
                for batch in _chunk(symbols, batch_size)
            ]
            for future in as_completed(futures):
                counts = future.result()
                batch_rows = sum(counts.values())
                total_rows += batch_rows
                log_event(logger, "batch_complete", rows=batch_rows)
    else:
        for batch in _chunk(symbols, batch_size):
            counts = _ingest_batch(cfg, batch, start, end, True, tz)
            batch_rows = sum(counts.values())
            total_rows += batch_rows
            log_event(logger, "batch_complete", rows=batch_rows)

    log_event(logger, "complete", symbols=len(symbols), rows=total_rows)
