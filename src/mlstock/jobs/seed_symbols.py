from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import reference_assets_path, reference_seed_symbols_path
from mlstock.logging.logger import build_log_path, log_event, setup_logger


SEED_COLUMNS = [
    "symbol",
    "name",
    "exchange",
    "seed_rank",
    "seeded_at_utc",
]


def run(cfg: AppConfig, n_seed: Optional[int] = None) -> pd.DataFrame:
    log_path = build_log_path(cfg, "seed_symbols")
    logger = setup_logger("seed_symbols", log_path, cfg.logging.level)

    log_event(logger, "start")

    assets_path = reference_assets_path(cfg)
    assets_df = read_parquet(assets_path)

    df = assets_df.copy()
    if "tradable" in df.columns and df["tradable"].notna().any():
        df = df[df["tradable"].fillna(False)]
    else:
        log_event(logger, "missing_tradable_column")
    if "status" in df.columns and df["status"].notna().any():
        df = df[df["status"] == "active"]
    else:
        log_event(logger, "missing_status_column")
    if "asset_class" in df.columns and df["asset_class"].notna().any():
        df = df[df["asset_class"] == "us_equity"]
    else:
        log_event(logger, "missing_asset_class_column")

    df = df[df["symbol"].notna()]
    df = df.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    df_all = df

    target = n_seed or cfg.seed.n_seed
    if target:
        df = df.head(int(target)).copy()

    spy_symbol = cfg.risk.regime_gate.spy_symbol
    if spy_symbol and spy_symbol not in df["symbol"].astype(str).tolist():
        spy_row = df_all[df_all["symbol"] == spy_symbol]
        if spy_row.empty:
            spy_row = pd.DataFrame([{"symbol": spy_symbol, "name": None, "exchange": None}])
        if target and len(df) > 0:
            df = pd.concat([df.iloc[:-1], spy_row], ignore_index=True)
        else:
            df = pd.concat([df, spy_row], ignore_index=True)
        df = df.drop_duplicates("symbol", keep="last").reset_index(drop=True)

    df["seed_rank"] = range(1, len(df) + 1)
    df["seeded_at_utc"] = datetime.now(timezone.utc)

    for column in SEED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[SEED_COLUMNS]

    seed_path = reference_seed_symbols_path(cfg)
    write_parquet_atomic(df, seed_path)

    log_event(logger, "complete", rows=int(len(df)), path=str(seed_path))
    return df
