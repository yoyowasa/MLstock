from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


ASSETS_REQUIRED_COLUMNS = [
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

CALENDAR_REQUIRED_COLUMNS = [
    "date",
    "open_time_local",
    "close_time_local",
    "open_ts_utc",
    "close_ts_utc",
    "fetched_at_utc",
]


def _missing_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def validate_assets_df(df: pd.DataFrame) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    missing = _missing_columns(df, ASSETS_REQUIRED_COLUMNS)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("Assets dataframe is empty")

    if "symbol" in df.columns:
        if df["symbol"].isna().any() or (df["symbol"].astype(str).str.len() == 0).any():
            errors.append("Empty symbol found")
        dupes = df["symbol"].duplicated().sum()
        if dupes:
            errors.append(f"Duplicate symbols: {dupes}")

    if "tradable" in df.columns:
        non_bool = df["tradable"].dropna().apply(lambda value: isinstance(value, (bool, np.bool_)))
        if not non_bool.all():
            errors.append("tradable contains non-bool values")

    if len(df) < 1000:
        warnings.append("Assets row count below expected threshold (<1000)")

    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "rows": int(len(df)),
        },
    }


def validate_calendar_df(
    df: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    missing = _missing_columns(df, CALENDAR_REQUIRED_COLUMNS)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("Calendar dataframe is empty")

    if "date" in df.columns:
        dupes = df["date"].duplicated().sum()
        if dupes:
            errors.append(f"Duplicate dates: {dupes}")
        if not df["date"].is_monotonic_increasing:
            errors.append("Calendar dates are not monotonic increasing")

        if start_date and df["date"].max() < start_date:
            errors.append("Calendar does not cover requested start_date")
        if start_date and end_date:
            if df["date"].max() < start_date or df["date"].min() > end_date:
                errors.append("Calendar does not cover requested range")

    if "open_ts_utc" in df.columns and "close_ts_utc" in df.columns:
        mask = df["open_ts_utc"].notna() & df["close_ts_utc"].notna()
        invalid = (df.loc[mask, "open_ts_utc"] >= df.loc[mask, "close_ts_utc"]).sum()
        if invalid:
            errors.append("open_ts_utc must be before close_ts_utc")

    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "rows": int(len(df)),
        },
    }


def build_reference_report(
    assets_report: Dict[str, Any],
    calendar_report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "pass": assets_report.get("pass") and calendar_report.get("pass"),
        "assets": assets_report,
        "calendar": calendar_report,
    }


def validate_assets_file(path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    return validate_assets_df(df)


def validate_calendar_file(
    path: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    return validate_calendar_df(df, start_date=start_date, end_date=end_date)
