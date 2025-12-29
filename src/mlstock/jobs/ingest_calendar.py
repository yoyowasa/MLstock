from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import write_parquet_atomic
from mlstock.data.storage.paths import reference_calendar_path
from mlstock.logging.logger import build_log_path, log_event, setup_logger
from mlstock.validate.reference import validate_calendar_df


CALENDAR_COLUMNS = [
    "date",
    "open_time_local",
    "close_time_local",
    "open_ts_utc",
    "close_ts_utc",
    "fetched_at_utc",
]


def _parse_datetime_local(date_str: str, time_str: str, tz: ZoneInfo) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)


def _normalize_calendar(
    records: List[dict],
    tz: ZoneInfo,
    fetched_at: datetime,
) -> pd.DataFrame:
    rows = []
    for item in records:
        date_str = item.get("date")
        open_str = item.get("open")
        close_str = item.get("close")
        open_ts_utc = None
        close_ts_utc = None
        if date_str and open_str:
            open_ts_utc = _parse_datetime_local(date_str, open_str, tz).astimezone(timezone.utc)
        if date_str and close_str:
            close_ts_utc = _parse_datetime_local(date_str, close_str, tz).astimezone(timezone.utc)
        rows.append(
            {
                "date": date.fromisoformat(date_str) if date_str else None,
                "open_time_local": open_str,
                "close_time_local": close_str,
                "open_ts_utc": open_ts_utc,
                "close_ts_utc": close_ts_utc,
                "fetched_at_utc": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: AppConfig) -> pd.DataFrame:
    log_path = build_log_path(cfg, "ingest_calendar")
    logger = setup_logger("ingest_calendar", log_path, cfg.logging.level)

    log_event(logger, "start")

    tz = ZoneInfo(cfg.project.timezone)
    start_date = date.fromisoformat(cfg.project.start_date)
    end_date = datetime.now(tz).date()

    client = AlpacaClient.from_env(cfg.alpaca.trading_base_url)
    records = client.get_calendar(start=start_date.isoformat(), end=end_date.isoformat())
    log_event(logger, "fetched", count=len(records), start=str(start_date), end=str(end_date))

    fetched_at = datetime.now(timezone.utc)
    df = _normalize_calendar(records, tz, fetched_at)

    for column in CALENDAR_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[CALENDAR_COLUMNS].sort_values("date").reset_index(drop=True)

    report = validate_calendar_df(df, start_date=start_date, end_date=end_date)
    if not report["pass"]:
        log_event(logger, "validation_failed", report=report)
        raise ValueError("Calendar validation failed")

    calendar_path = reference_calendar_path(cfg)
    write_parquet_atomic(df, calendar_path)

    log_event(logger, "complete", rows=int(len(df)), path=str(calendar_path))
    return df
