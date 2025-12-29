from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import (
    raw_bars_path,
    raw_corp_actions_path,
    reference_calendar_path,
    reference_seed_symbols_path,
    snapshots_features_path,
    snapshots_labels_path,
    snapshots_universe_path,
    snapshots_week_map_path,
)
from mlstock.logging.logger import build_log_path, log_event, setup_logger
from mlstock.model.features import FEATURE_COLUMNS


def _build_week_map(calendar_df: pd.DataFrame) -> pd.DataFrame:
    df = calendar_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["week_start"] = df["date"].apply(lambda value: value - timedelta(days=value.weekday()))
    grouped = df.groupby("week_start", as_index=False).agg(anchor_date=("date", "min"), week_end=("date", "max"))
    grouped["next_week_start"] = grouped["week_start"].apply(lambda value: value + timedelta(days=7))
    anchor_lookup = dict(zip(grouped["week_start"], grouped["anchor_date"]))
    grouped["next_anchor_date"] = grouped["next_week_start"].map(anchor_lookup)
    return grouped


def _apply_splits(bars_df: pd.DataFrame, actions_df: pd.DataFrame) -> pd.DataFrame:
    if actions_df.empty:
        return bars_df
    df = bars_df.copy()
    df["adj_factor"] = 1.0
    actions = actions_df.sort_values("ex_date")
    for _, action in actions.iterrows():
        old_rate = action.get("old_rate")
        new_rate = action.get("new_rate")
        ex_date = action.get("ex_date")
        if not old_rate or not new_rate or not ex_date:
            continue
        factor = float(old_rate) / float(new_rate)
        df.loc[df["date"] < ex_date, "adj_factor"] *= factor
    for column in ("open", "high", "low", "close", "vwap"):
        if column in df.columns:
            df[f"adj_{column}"] = df[column] * df["adj_factor"]
    return df


def _compute_daily_metrics(bars_df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    df = bars_df.sort_values("date").reset_index(drop=True)
    df["dollar_vol"] = df["close"] * df["volume"]
    window = int(cfg.snapshots.feature_lookback_days)
    min_days = int(cfg.snapshots.min_trading_days)
    df["avg_dollar_vol_20d"] = df["dollar_vol"].rolling(window, min_periods=min_days).mean()
    return df


def _build_weekly_table(
    bars_df: pd.DataFrame,
    week_map: pd.DataFrame,
) -> pd.DataFrame:
    price_col = "adj_open" if "adj_open" in bars_df.columns else "open"
    weekly = bars_df.merge(week_map, left_on="date", right_on="anchor_date", how="inner")
    next_open = bars_df[["date", price_col]].rename(columns={"date": "next_anchor_date", price_col: "next_open"})
    weekly = weekly.merge(next_open, on="next_anchor_date", how="left")

    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["price"] = weekly[price_col]
    weekly["ret_1w"] = weekly["price"].pct_change()
    weekly["ret_4w"] = weekly["price"].pct_change(4)
    weekly["vol_4w"] = weekly["ret_1w"].rolling(4, min_periods=4).std()
    weekly["label_return"] = weekly["next_open"] / weekly["price"] - 1
    return weekly


def _load_seed_symbols(cfg: AppConfig) -> List[str]:
    seed_path = reference_seed_symbols_path(cfg)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed symbols not found: {seed_path}. Run: python -m mlstock make-seed")
    seed_df = read_parquet(seed_path)
    return seed_df["symbol"].dropna().astype(str).tolist()


def run(cfg: AppConfig, symbols: Optional[List[str]] = None) -> Dict[str, int]:
    log_path = build_log_path(cfg, "build_snapshots")
    logger = setup_logger("build_snapshots", log_path, cfg.logging.level)
    log_event(logger, "start")

    exclude_symbols = {symbol.upper() for symbol in cfg.snapshots.exclude_symbols}

    calendar_df = read_parquet(reference_calendar_path(cfg))
    if calendar_df.empty:
        raise ValueError("Calendar reference data is empty")

    week_map = _build_week_map(calendar_df)
    write_parquet_atomic(week_map, snapshots_week_map_path(cfg))

    symbols = symbols or _load_seed_symbols(cfg)
    if not symbols:
        raise ValueError("No seed symbols available for snapshots")

    actions_path = raw_corp_actions_path(cfg)
    actions_df = read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()

    features_frames: List[pd.DataFrame] = []
    labels_frames: List[pd.DataFrame] = []
    universe_frames: List[pd.DataFrame] = []

    for symbol in symbols:
        if exclude_symbols and str(symbol).upper() in exclude_symbols:
            continue
        bars_path = raw_bars_path(cfg, symbol)
        if not bars_path.exists():
            continue
        bars_df = read_parquet(bars_path)
        if bars_df.empty:
            continue
        bars_df["date"] = pd.to_datetime(bars_df["date"]).dt.date
        bars_df = bars_df.sort_values("date").reset_index(drop=True)

        symbol_actions = actions_df[actions_df["symbol"] == symbol] if not actions_df.empty else pd.DataFrame()
        if not symbol_actions.empty:
            symbol_actions = symbol_actions[symbol_actions["action_type"] == "split"]
        bars_df = _apply_splits(bars_df, symbol_actions)
        bars_df = _compute_daily_metrics(bars_df, cfg)

        weekly = _build_weekly_table(bars_df, week_map)
        if weekly.empty:
            continue
        weekly["symbol"] = symbol

        features_cols = ["week_start", "symbol", "price", "avg_dollar_vol_20d"] + list(FEATURE_COLUMNS)
        for column in features_cols:
            if column not in weekly.columns:
                weekly[column] = None
        features_frames.append(weekly[features_cols])

        labels = weekly[["week_start", "symbol", "label_return"]].dropna(subset=["label_return"])
        labels_frames.append(labels)

        universe = weekly[["week_start", "symbol", "price", "avg_dollar_vol_20d"]]
        universe_frames.append(universe)

    features_df = pd.concat(features_frames, ignore_index=True) if features_frames else pd.DataFrame()
    labels_df = pd.concat(labels_frames, ignore_index=True) if labels_frames else pd.DataFrame()
    universe_df = pd.concat(universe_frames, ignore_index=True) if universe_frames else pd.DataFrame()

    if exclude_symbols:
        if not features_df.empty:
            features_df = features_df[~features_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
        if not labels_df.empty:
            labels_df = labels_df[~labels_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
        if not universe_df.empty:
            universe_df = universe_df[~universe_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]

    if not features_df.empty:
        features_df = features_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
        write_parquet_atomic(features_df, snapshots_features_path(cfg))
    else:
        write_parquet_atomic(features_df, snapshots_features_path(cfg))

    if not labels_df.empty:
        labels_df = labels_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
        write_parquet_atomic(labels_df, snapshots_labels_path(cfg))
    else:
        write_parquet_atomic(labels_df, snapshots_labels_path(cfg))

    if not universe_df.empty:
        universe_df = universe_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
        universe_df = universe_df[
            universe_df["avg_dollar_vol_20d"].fillna(0) >= float(cfg.snapshots.min_avg_dollar_vol_20d)
        ]
        write_parquet_atomic(universe_df, snapshots_universe_path(cfg))
    else:
        write_parquet_atomic(universe_df, snapshots_universe_path(cfg))

    counts = {
        "weeks": int(len(week_map)),
        "features": int(len(features_df)),
        "labels": int(len(labels_df)),
        "universe": int(len(universe_df)),
    }
    log_event(logger, "complete", **counts)
    return counts
