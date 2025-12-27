from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import write_parquet_atomic
from mlstock.data.storage.paths import reference_assets_path
from mlstock.logging.logger import build_log_path, log_event, setup_logger
from mlstock.validate.reference import validate_assets_df


ASSET_COLUMNS = [
    "symbol",
    "name",
    "exchange",
    "asset_class",
    "status",
    "tradable",
    "marginable",
    "shortable",
    "easy_to_borrow",
    "fractionable",
    "raw_json",
    "fetched_at_utc",
]


def _normalize_assets(records: List[dict], fetched_at: datetime) -> pd.DataFrame:
    rows = []
    for item in records:
        rows.append(
            {
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "exchange": item.get("exchange"),
                "asset_class": item.get("asset_class"),
                "status": item.get("status"),
                "tradable": item.get("tradable"),
                "marginable": item.get("marginable"),
                "shortable": item.get("shortable"),
                "easy_to_borrow": item.get("easy_to_borrow"),
                "fractionable": item.get("fractionable"),
                "raw_json": json.dumps(item, separators=(",", ":"), sort_keys=True),
                "fetched_at_utc": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: AppConfig) -> pd.DataFrame:
    log_path = build_log_path(cfg, "ingest_assets")
    logger = setup_logger("ingest_assets", log_path, cfg.logging.level)

    log_event(logger, "start")

    client = AlpacaClient.from_env(cfg.alpaca.data_base_url)
    fetched_at = datetime.now(timezone.utc)

    all_frames = []
    for exchange in ("NYSE", "NASDAQ"):
        records = client.get_assets(
            status="active",
            asset_class="us_equity",
            exchange=exchange,
        )
        log_event(logger, "fetched", exchange=exchange, count=len(records))
        all_frames.append(_normalize_assets(records, fetched_at))

    if not all_frames:
        raise RuntimeError("No assets data returned")

    df = pd.concat(all_frames, ignore_index=True)

    for column in ASSET_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[
        (df["status"] == "active")
        & (df["asset_class"] == "us_equity")
        & (df["tradable"].fillna(False))
    ]

    if "symbol" in df.columns:
        df = df[df["symbol"].notna()]

    duplicate_count = df["symbol"].duplicated().sum()
    if duplicate_count:
        log_event(logger, "duplicate_symbols", count=int(duplicate_count))
        df = df.drop_duplicates("symbol", keep="first")

    df = df[ASSET_COLUMNS].sort_values("symbol").reset_index(drop=True)

    report = validate_assets_df(df)
    if not report["pass"]:
        log_event(logger, "validation_failed", report=report)
        raise ValueError("Assets validation failed")

    assets_path = reference_assets_path(cfg)
    write_parquet_atomic(df, assets_path)

    log_event(logger, "complete", rows=int(len(df)), path=str(assets_path))
    return df
