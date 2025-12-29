from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import raw_corp_actions_path, reference_seed_symbols_path
from mlstock.logging.logger import build_log_path, log_event, setup_logger


SPLIT_TYPES = {"forward_split", "reverse_split", "unit_split"}

ACTION_COLUMNS = [
    "symbol",
    "action_type",
    "ex_date",
    "old_rate",
    "new_rate",
    "ratio",
    "raw_json",
    "fetched_at_utc",
]


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _parse_ratio(ratio: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not ratio:
        return None, None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*$", ratio)
    if not match:
        return None, None
    new_rate = float(match.group(1))
    old_rate = float(match.group(2))
    return new_rate, old_rate


def _parse_rate(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _normalize_actions(records: List[dict], fetched_at: datetime) -> pd.DataFrame:
    rows = []
    for item in records:
        raw_type = item.get("type") or item.get("action_type")
        if raw_type not in SPLIT_TYPES:
            continue
        ratio = item.get("ratio")
        new_rate, old_rate = _parse_ratio(ratio)
        old_rate = _parse_rate(item.get("old_rate")) or old_rate
        new_rate = _parse_rate(item.get("new_rate")) or new_rate
        ex_date = item.get("ex_date") or item.get("exDate") or item.get("effective_date")
        rows.append(
            {
                "symbol": item.get("symbol"),
                "action_type": "split",
                "ex_date": date.fromisoformat(ex_date) if ex_date else None,
                "old_rate": old_rate,
                "new_rate": new_rate,
                "ratio": ratio,
                "raw_json": json.dumps(item, separators=(",", ":"), sort_keys=True),
                "fetched_at_utc": fetched_at,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["symbol", "action_type", "ex_date", "old_rate", "new_rate"]).reset_index(drop=True)
    return df


def _extract_actions(response: object) -> List[dict]:
    if not isinstance(response, dict):
        return []
    actions = response.get("corporate_actions") or response.get("actions")
    if actions is None:
        return []
    if isinstance(actions, list):
        return actions
    if isinstance(actions, dict):
        flattened: List[dict] = []
        for symbol, items in actions.items():
            if not items:
                continue
            for item in items:
                if "symbol" not in item:
                    item = dict(item)
                    item["symbol"] = symbol
                flattened.append(item)
        return flattened
    return []


def _fetch_actions_batch(
    client: AlpacaClient,
    symbols: List[str],
    start: str,
    end: str,
) -> List[dict]:
    collected: List[dict] = []
    page_token: Optional[str] = None
    while True:
        response = client.get_corporate_actions(
            symbols=symbols,
            types=",".join(sorted(SPLIT_TYPES)),
            start=start,
            end=end,
            page_token=page_token,
        )
        collected.extend(_extract_actions(response))
        page_token = response.get("next_page_token") if isinstance(response, dict) else None
        if not page_token:
            break
    return collected


def _ingest_batch(cfg: AppConfig, symbols: List[str], start: date, end: date) -> pd.DataFrame:
    client = AlpacaClient.from_env(cfg.alpaca.data_base_url)
    raw_actions = _fetch_actions_batch(client, symbols, start.isoformat(), end.isoformat())
    return _normalize_actions(raw_actions, datetime.now(timezone.utc))


def _load_seed_symbols(cfg: AppConfig) -> List[str]:
    seed_path = reference_seed_symbols_path(cfg)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed symbols not found: {seed_path}. Run: python -m mlstock make-seed")
    seed_df = read_parquet(seed_path)
    return seed_df["symbol"].dropna().astype(str).tolist()


def backfill(cfg: AppConfig, symbols: Optional[List[str]] = None) -> None:
    log_path = build_log_path(cfg, "ingest_corp_actions_backfill")
    logger = setup_logger("ingest_corp_actions_backfill", log_path, cfg.logging.level)
    log_event(logger, "start")

    tz = ZoneInfo(cfg.project.timezone)
    start = date.fromisoformat(cfg.corp_actions.backfill_start)
    end = datetime.now(tz).date()
    symbols = symbols or _load_seed_symbols(cfg)

    if not symbols:
        raise ValueError("No seed symbols available for corporate actions backfill")

    batch_size = min(cfg.corp_actions.batch_size, 200)
    max_workers = max(int(cfg.corp_actions.max_workers), 1)

    frames: List[pd.DataFrame] = []
    if max_workers > 1 and len(symbols) > batch_size:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_ingest_batch, cfg, batch, start, end) for batch in _chunk(symbols, batch_size)]
            for future in as_completed(futures):
                df = future.result()
                frames.append(df)
                log_event(logger, "batch_complete", rows=int(len(df)))
    else:
        for batch in _chunk(symbols, batch_size):
            df = _ingest_batch(cfg, batch, start, end)
            frames.append(df)
            log_event(logger, "batch_complete", rows=int(len(df)))

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ACTION_COLUMNS)
    for column in ACTION_COLUMNS:
        if column not in df_all.columns:
            df_all[column] = None
    df_all = df_all[ACTION_COLUMNS].sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    if not df_all.empty:
        df_all["ex_date"] = pd.to_datetime(df_all["ex_date"]).dt.date

    path = raw_corp_actions_path(cfg)
    write_parquet_atomic(df_all, path)

    log_event(logger, "complete", rows=int(len(df_all)), path=str(path))


def incremental_update(cfg: AppConfig, symbols: Optional[List[str]] = None) -> None:
    log_path = build_log_path(cfg, "ingest_corp_actions_incremental")
    logger = setup_logger("ingest_corp_actions_incremental", log_path, cfg.logging.level)
    log_event(logger, "start")

    tz = ZoneInfo(cfg.project.timezone)
    end = datetime.now(tz).date()
    start = end - timedelta(days=int(cfg.corp_actions.lookback_days))
    symbols = symbols or _load_seed_symbols(cfg)

    if not symbols:
        raise ValueError("No seed symbols available for corporate actions incremental update")

    batch_size = min(cfg.corp_actions.batch_size, 200)
    max_workers = max(int(cfg.corp_actions.max_workers), 1)

    frames: List[pd.DataFrame] = []
    if max_workers > 1 and len(symbols) > batch_size:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_ingest_batch, cfg, batch, start, end) for batch in _chunk(symbols, batch_size)]
            for future in as_completed(futures):
                df = future.result()
                frames.append(df)
                log_event(logger, "batch_complete", rows=int(len(df)))
    else:
        for batch in _chunk(symbols, batch_size):
            df = _ingest_batch(cfg, batch, start, end)
            frames.append(df)
            log_event(logger, "batch_complete", rows=int(len(df)))

    df_new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ACTION_COLUMNS)
    for column in ACTION_COLUMNS:
        if column not in df_new.columns:
            df_new[column] = None
    df_new = df_new[ACTION_COLUMNS]
    if not df_new.empty:
        df_new["ex_date"] = pd.to_datetime(df_new["ex_date"]).dt.date

    path = raw_corp_actions_path(cfg)
    if path.exists():
        existing = read_parquet(path)
        if not existing.empty:
            existing["ex_date"] = pd.to_datetime(existing["ex_date"]).dt.date
            existing = existing[existing["ex_date"] < start]
        df_all = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all = df_all.drop_duplicates(subset=["symbol", "action_type", "ex_date", "old_rate", "new_rate"])
    df_all = df_all.sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    write_parquet_atomic(df_all, path)

    log_event(logger, "complete", rows=int(len(df_all)), path=str(path))
