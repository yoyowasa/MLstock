from __future__ import annotations

from datetime import date, timedelta
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
    spy_return_by_week: Optional[Dict[date, float]] = None,
) -> pd.DataFrame:
    price_col = "adj_open" if "adj_open" in bars_df.columns else "open"
    weekly = bars_df.merge(week_map, left_on="date", right_on="anchor_date", how="inner")
    next_open = bars_df[["date", price_col]].rename(columns={"date": "next_anchor_date", price_col: "next_open"})
    weekly = weekly.merge(next_open, on="next_anchor_date", how="left")

    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly["price"] = weekly[price_col]
    # This function is called per symbol, so pct_change/rolling never crosses symbols.
    # Initial NaNs are intentional warm-up periods and are filtered downstream when needed.
    weekly["ret_1w"] = weekly["price"].pct_change()
    weekly["ret_4w"] = weekly["price"].pct_change(4)
    weekly["ret_8w"] = weekly["price"].pct_change(8)
    weekly["ret_13w"] = weekly["price"].pct_change(13)
    weekly["ret_26w"] = weekly["price"].pct_change(26)

    weekly["vol_4w"] = weekly["ret_1w"].rolling(4, min_periods=4).std()
    weekly["vol_8w"] = weekly["ret_1w"].rolling(8, min_periods=8).std()
    weekly["vol_13w"] = weekly["ret_1w"].rolling(13, min_periods=13).std()

    weekly["ma_ratio_10w"] = weekly["price"] / weekly["price"].rolling(10, min_periods=10).mean() - 1.0
    weekly["ma_ratio_20w"] = weekly["price"] / weekly["price"].rolling(20, min_periods=20).mean() - 1.0

    # 週次 high/low の結合は未実装のため、簡易的に price の rolling max/min で代用する。
    weekly["high_low_range_4w"] = (
        weekly["price"].rolling(4).max() - weekly["price"].rolling(4).min()
    ) / weekly["price"]

    if "volume" in weekly.columns:
        vol_1w = weekly["volume"]
        vol_4w_avg = weekly["volume"].rolling(4, min_periods=4).mean()
        weekly["volume_ratio_4w"] = vol_1w / vol_4w_avg
    else:
        weekly["volume_ratio_4w"] = None

    weekly["label_return_raw"] = weekly["next_open"] / weekly["price"] - 1
    if spy_return_by_week is not None:
        weekly["spy_return"] = weekly["week_start"].map(spy_return_by_week)
        weekly["label_return"] = weekly["label_return_raw"] - weekly["spy_return"].fillna(0.0)
    else:
        weekly["label_return"] = weekly["label_return_raw"]
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

    actions_path = raw_corp_actions_path(cfg)
    actions_df = read_parquet(actions_path) if actions_path.exists() else pd.DataFrame()

    benchmark_symbol = str(cfg.weekly.labels.benchmark_symbol).upper()
    use_excess_label = bool(cfg.weekly.labels.use_excess)
    spy_return_by_week: Dict[date, float] = {}
    spy_features_by_week: Dict[date, Dict[str, float]] = {}
    if use_excess_label and benchmark_symbol:
        spy_bars_path = raw_bars_path(cfg, benchmark_symbol)
        if spy_bars_path.exists():
            spy_bars_df = read_parquet(spy_bars_path)
            if not spy_bars_df.empty:
                spy_bars_df["date"] = pd.to_datetime(spy_bars_df["date"]).dt.date
                spy_bars_df = spy_bars_df.sort_values("date").reset_index(drop=True)
                spy_actions = (
                    actions_df[actions_df["symbol"] == benchmark_symbol] if not actions_df.empty else pd.DataFrame()
                )
                if not spy_actions.empty:
                    spy_actions = spy_actions[spy_actions["action_type"] == "split"]
                spy_bars_df = _apply_splits(spy_bars_df, spy_actions)
                spy_bars_df = _compute_daily_metrics(spy_bars_df, cfg)
                spy_weekly = _build_weekly_table(spy_bars_df, week_map)
                if not spy_weekly.empty and "label_return_raw" in spy_weekly.columns:
                    spy_labels = spy_weekly[["week_start", "label_return_raw"]].dropna(subset=["label_return_raw"])
                    for row in spy_labels.itertuples(index=False):
                        spy_return_by_week[row.week_start] = float(row.label_return_raw)
                # Build SPY market context features for each week
                if not spy_weekly.empty:
                    for row in spy_weekly.itertuples(index=False):
                        feats: Dict[str, float] = {}
                        feats["spy_ret_1w"] = float(row.ret_1w) if pd.notna(row.ret_1w) else float("nan")
                        feats["spy_ret_4w"] = float(row.ret_4w) if pd.notna(row.ret_4w) else float("nan")
                        feats["spy_vol_4w"] = float(row.vol_4w) if pd.notna(row.vol_4w) else float("nan")
                        spy_features_by_week[row.week_start] = feats

    symbols = symbols or _load_seed_symbols(cfg)
    if not symbols:
        raise ValueError("No seed symbols available for snapshots")

    features_frames: List[pd.DataFrame] = []
    labels_frames: List[pd.DataFrame] = []
    universe_frames: List[pd.DataFrame] = []
    feature_output_cols = ["week_start", "symbol", "price", "avg_dollar_vol_20d"] + list(FEATURE_COLUMNS)
    label_output_cols = ["week_start", "symbol", "label_return"]
    universe_output_cols = ["week_start", "symbol", "price", "avg_dollar_vol_20d"]

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

        weekly = _build_weekly_table(
            bars_df,
            week_map,
            spy_return_by_week=spy_return_by_week if use_excess_label else None,
        )
        if weekly.empty:
            continue
        weekly["symbol"] = symbol

        for column in feature_output_cols:
            if column not in weekly.columns:
                weekly[column] = None
        features_frames.append(weekly[feature_output_cols])

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

    # ret_1w_rank は同一週の銘柄間で計算するクロスセクショナル特徴量。
    if not features_df.empty and "ret_1w" in features_df.columns:
        features_df["ret_1w_rank"] = features_df.groupby("week_start")["ret_1w"].rank(pct=True, method="average")

    # マーケットコンテキスト特徴量: SPY系 + market_breadth
    if not features_df.empty:
        # SPY features: spy_ret_1w, spy_ret_4w, spy_vol_4w
        if spy_features_by_week:
            for col in ("spy_ret_1w", "spy_ret_4w", "spy_vol_4w"):
                features_df[col] = features_df["week_start"].map(
                    lambda ws, c=col: spy_features_by_week.get(ws, {}).get(c, float("nan"))
                )
        else:
            for col in ("spy_ret_1w", "spy_ret_4w", "spy_vol_4w"):
                features_df[col] = float("nan")
        # market_breadth: 同一週で ret_1w > 0 の銘柄割合
        breadth = features_df.groupby("week_start")["ret_1w"].apply(
            lambda s: (s > 0).sum() / max(len(s), 1)
        ).rename("market_breadth")
        features_df = features_df.merge(
            breadth.reset_index(), on="week_start", how="left", suffixes=("", "_calc")
        )
        if "market_breadth_calc" in features_df.columns:
            features_df["market_breadth"] = features_df["market_breadth_calc"]
            features_df = features_df.drop(columns=["market_breadth_calc"])

    if features_df.empty:
        features_df = pd.DataFrame(columns=feature_output_cols)
    else:
        for column in feature_output_cols:
            if column not in features_df.columns:
                features_df[column] = None
        features_df = features_df[feature_output_cols]
        features_df = features_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
    write_parquet_atomic(features_df, snapshots_features_path(cfg))

    if labels_df.empty:
        labels_df = pd.DataFrame(columns=label_output_cols)
    else:
        for column in label_output_cols:
            if column not in labels_df.columns:
                labels_df[column] = None
        labels_df = labels_df[label_output_cols]
        labels_df = labels_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
    write_parquet_atomic(labels_df, snapshots_labels_path(cfg))

    if universe_df.empty:
        universe_df = pd.DataFrame(columns=universe_output_cols)
    else:
        for column in universe_output_cols:
            if column not in universe_df.columns:
                universe_df[column] = None
        universe_df = universe_df[universe_output_cols]
        universe_df = universe_df.sort_values(["week_start", "symbol"]).reset_index(drop=True)
        universe_df = universe_df[
            universe_df["avg_dollar_vol_20d"].fillna(0) >= float(cfg.snapshots.min_avg_dollar_vol_20d)
        ]
    write_parquet_atomic(universe_df, snapshots_universe_path(cfg))

    counts = {
        "weeks": int(len(week_map)),
        "features": int(len(features_df)),
        "labels": int(len(labels_df)),
        "universe": int(len(universe_df)),
    }
    log_event(logger, "complete", **counts)
    return counts
