from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def select_training_weeks(
    available_weeks: Sequence[date],
    current_week: date,
    train_window_years: float,
    min_train_weeks: int,
) -> List[date]:
    if train_window_years <= 0:
        raise ValueError("train_window_years must be positive")
    if min_train_weeks <= 0:
        raise ValueError("min_train_weeks must be positive")

    if not available_weeks:
        return []

    weeks_sorted = sorted(available_weeks)
    eligible = [week for week in weeks_sorted if week < current_week]
    if not eligible:
        return []

    max_weeks = max(int(train_window_years * 52), 1)
    selected = eligible[-max_weeks:]

    if len(selected) < min_train_weeks:
        return []

    return selected


def train_linear_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
) -> Optional[Dict[str, object]]:
    if train_df.empty:
        return None
    X = train_df.loc[:, feature_cols].to_numpy(dtype=float, copy=False)
    y = train_df.loc[:, label_col].to_numpy(dtype=float, copy=False)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if X.size == 0:
        return None
    X_design = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return {
        "intercept": float(coef[0]),
        "coef": [float(value) for value in coef[1:]],
        "features": list(feature_cols),
    }


def predict_linear_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    X = features_df.loc[:, feature_cols].to_numpy(dtype=float, copy=False)
    coef = np.array(model.get("coef", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))
    return intercept + X.dot(coef)
