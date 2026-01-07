from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class VolCapStats:
    candidates: int
    excluded: int
    missing: int

    @property
    def excluded_rate(self) -> Optional[float]:
        if self.candidates <= 0:
            return None
        return self.excluded / self.candidates


@dataclass(frozen=True)
class VolCapPenaltyStats:
    candidates: int
    penalized: int
    missing: int

    @property
    def penalized_rate(self) -> Optional[float]:
        if self.candidates <= 0:
            return None
        return self.penalized / self.candidates


def apply_vol_cap(
    df: pd.DataFrame,
    *,
    feature_name: str,
    rank_threshold: float,
    hold_symbols: Optional[Iterable[str]] = None,
    hold_threshold: Optional[float] = None,
    group_by: Optional[str] = None,
    enabled: bool = True,
) -> Tuple[pd.DataFrame, VolCapStats]:
    if not enabled:
        return df, VolCapStats(candidates=0, excluded=0, missing=0)
    if df.empty:
        return df, VolCapStats(candidates=0, excluded=0, missing=0)
    if feature_name not in df.columns:
        raise ValueError(f"Vol cap feature not found: {feature_name}")
    if not (0.0 < rank_threshold <= 1.0):
        raise ValueError(f"Vol cap rank_threshold must be in (0, 1]: {rank_threshold}")

    series = pd.to_numeric(df[feature_name], errors="coerce")
    missing_mask = series.isna()
    if group_by and group_by in df.columns:
        rank = series.groupby(df[group_by]).rank(pct=True, method="max")
    else:
        rank = series.rank(pct=True, method="max")

    hold_mask = pd.Series(False, index=df.index)
    if hold_symbols is not None and hold_threshold is not None:
        hold_threshold = max(float(rank_threshold), float(hold_threshold))
        hold_set: Set[str] = {str(symbol).upper() for symbol in hold_symbols}
        if hold_set and "symbol" in df.columns:
            symbol_series = df["symbol"].astype(str).str.upper()
            hold_mask = symbol_series.isin(hold_set) & (rank <= hold_threshold)

    keep_mask = (~missing_mask) & ((rank <= rank_threshold) | hold_mask)

    filtered = df.loc[keep_mask].copy()
    candidates = int(len(df))
    excluded = candidates - int(len(filtered))
    missing = int(missing_mask.sum())

    return filtered, VolCapStats(candidates=candidates, excluded=excluded, missing=missing)


def apply_vol_penalty(
    df: pd.DataFrame,
    *,
    feature_name: str,
    rank_threshold: float,
    penalty_min: float,
    group_by: Optional[str] = None,
    enabled: bool = True,
) -> Tuple[pd.DataFrame, VolCapPenaltyStats]:
    if not enabled:
        return df, VolCapPenaltyStats(candidates=0, penalized=0, missing=0)
    if df.empty:
        return df, VolCapPenaltyStats(candidates=0, penalized=0, missing=0)
    if feature_name not in df.columns:
        raise ValueError(f"Vol cap feature not found: {feature_name}")
    if not (0.0 < rank_threshold <= 1.0):
        raise ValueError(f"Vol cap rank_threshold must be in (0, 1]: {rank_threshold}")
    if not (0.0 < penalty_min <= 1.0):
        raise ValueError(f"Vol cap penalty_min must be in (0, 1]: {penalty_min}")

    series = pd.to_numeric(df[feature_name], errors="coerce")
    missing_mask = series.isna()
    if group_by and group_by in df.columns:
        rank = series.groupby(df[group_by]).rank(pct=True, method="max")
    else:
        rank = series.rank(pct=True, method="max")

    penalty = pd.Series(1.0, index=df.index, dtype=float)
    if rank_threshold < 1.0:
        slope = (1.0 - penalty_min) / (1.0 - rank_threshold)
        above = rank > rank_threshold
        penalty = penalty.where(~above, 1.0 - (rank - rank_threshold) * slope)
    penalty = penalty.clip(lower=penalty_min, upper=1.0)
    penalty = penalty.where(~missing_mask)

    candidates = int(len(df))
    penalized = int(((rank > rank_threshold) & ~missing_mask).sum())
    missing = int(missing_mask.sum())

    filtered = df.assign(vol_cap_rank=rank, vol_cap_penalty=penalty)
    filtered = filtered[filtered["vol_cap_penalty"].notna()].copy()

    return filtered, VolCapPenaltyStats(candidates=candidates, penalized=penalized, missing=missing)
