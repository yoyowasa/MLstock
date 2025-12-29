from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Dict, Optional, Tuple

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import raw_bars_path


@dataclass(frozen=True)
class RegimeGateResult:
    open_by_week: Dict[date, bool]
    source: str
    ma_days: int
    missing_weeks: int


def _parse_ma_days(rule: str, default_days: int) -> int:
    match = re.search(r"ma(\d+)", rule)
    if match:
        return int(match.group(1))
    return default_days


def _normalize_week_map(week_map_df: pd.DataFrame) -> pd.DataFrame:
    df = week_map_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    if "week_end" in df.columns:
        df["week_end"] = pd.to_datetime(df["week_end"]).dt.date
    else:
        df["week_end"] = df["week_start"]
    return df


def _build_daily_gate(
    week_map_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    ma_days: int,
) -> Optional[Tuple[Dict[date, bool], int]]:
    close_col = "adj_close" if "adj_close" in bars_df.columns else "close"
    if close_col not in bars_df.columns or "date" not in bars_df.columns:
        return None

    daily = bars_df[["date", close_col]].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily.dropna(subset=["date", close_col]).sort_values("date")
    if daily.empty:
        return None

    daily["ma"] = daily[close_col].rolling(ma_days, min_periods=ma_days).mean()
    daily["date_dt"] = pd.to_datetime(daily["date"])

    week_map = _normalize_week_map(week_map_df).sort_values("week_end")
    week_map["week_end_dt"] = pd.to_datetime(week_map["week_end"])

    merged = pd.merge_asof(
        week_map,
        daily[["date_dt", close_col, "ma"]],
        left_on="week_end_dt",
        right_on="date_dt",
        direction="backward",
    )

    valid = merged[close_col].notna() & merged["ma"].notna()
    gate_open = (merged[close_col] >= merged["ma"]) | ~valid
    missing_weeks = int((~valid).sum())

    open_by_week = {
        row.week_start: bool(open_flag)
        for row, open_flag in zip(merged.itertuples(index=False), gate_open.tolist())
    }
    return open_by_week, missing_weeks


def _build_weekly_gate(
    week_map_df: pd.DataFrame,
    spy_weekly_df: pd.DataFrame,
    ma_days: int,
) -> Optional[Tuple[Dict[date, bool], int]]:
    if "week_start" not in spy_weekly_df.columns or "price" not in spy_weekly_df.columns:
        return None

    week_map = _normalize_week_map(week_map_df)
    spy_weekly = spy_weekly_df[["week_start", "price"]].copy()
    spy_weekly["week_start"] = pd.to_datetime(spy_weekly["week_start"]).dt.date
    spy_weekly = spy_weekly.dropna(subset=["week_start", "price"]).sort_values("week_start")
    if spy_weekly.empty:
        return None

    ma_weeks = max(1, int(round(ma_days / 5)))
    spy_weekly["ma"] = spy_weekly["price"].rolling(ma_weeks, min_periods=ma_weeks).mean()

    merged = week_map.merge(spy_weekly, on="week_start", how="left")
    valid = merged["price"].notna() & merged["ma"].notna()
    gate_open = (merged["price"] >= merged["ma"]) | ~valid
    missing_weeks = int((~valid).sum())

    open_by_week = {
        row.week_start: bool(open_flag)
        for row, open_flag in zip(merged.itertuples(index=False), gate_open.tolist())
    }
    return open_by_week, missing_weeks


def build_spy_regime_gate(
    cfg: AppConfig,
    week_map_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
) -> RegimeGateResult:
    if week_map_df.empty or "week_start" not in week_map_df.columns:
        return RegimeGateResult({}, "missing_week_map", cfg.risk.regime_gate.ma_days, 0)

    week_map_norm = _normalize_week_map(week_map_df)
    week_starts = week_map_norm["week_start"].tolist()

    rule = cfg.risk.regime_gate.rule
    ma_days = _parse_ma_days(rule, cfg.risk.regime_gate.ma_days)

    if not rule.startswith("spy_close_above_ma"):
        open_by_week = {week: True for week in week_starts}
        return RegimeGateResult(open_by_week, "unknown_rule", ma_days, len(open_by_week))

    spy_symbol = cfg.risk.regime_gate.spy_symbol
    bars_path = raw_bars_path(cfg, spy_symbol)
    if bars_path.exists():
        bars_df = read_parquet(bars_path)
        if not bars_df.empty:
            daily_gate = _build_daily_gate(week_map_norm, bars_df, ma_days)
            if daily_gate is not None:
                open_by_week, missing_weeks = daily_gate
                return RegimeGateResult(open_by_week, "daily", ma_days, missing_weeks)

    if features_df is not None and not features_df.empty:
        spy_weekly = features_df[features_df["symbol"] == spy_symbol]
        if not spy_weekly.empty:
            weekly_gate = _build_weekly_gate(week_map_norm, spy_weekly, ma_days)
            if weekly_gate is not None:
                open_by_week, missing_weeks = weekly_gate
                return RegimeGateResult(open_by_week, "weekly_fallback", ma_days, missing_weeks)

    open_by_week = {week: True for week in week_starts}
    return RegimeGateResult(open_by_week, "missing_spy_data", ma_days, len(open_by_week))
