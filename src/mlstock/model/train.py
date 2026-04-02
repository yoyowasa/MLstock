from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except Exception:
    lgb = None  # type: ignore[assignment]


def _prepare_feature_frame(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in feature_cols:
        if column not in prepared.columns:
            prepared[column] = 0.0
    return prepared.loc[:, list(feature_cols)]


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
    return train_ridge_model(train_df, feature_cols, label_col)


def train_ridge_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    alpha: float = 1.0,
) -> Optional[Dict[str, object]]:
    if Ridge is None:
        raise ImportError("scikit-learn is required for Ridge model. Install with: pip install scikit-learn>=1.3")
    if train_df.empty:
        return None
    X = _prepare_feature_frame(train_df, feature_cols).to_numpy(dtype=float, copy=False)
    y = train_df.loc[:, label_col].to_numpy(dtype=float, copy=False)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if len(X) < 1:
        return None

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)

    return {
        "type": "ridge",
        "intercept": float(model.intercept_),
        "coef": [float(value) for value in model.coef_],
        "features": list(feature_cols),
        "alpha": float(alpha),
    }


def train_lgb_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    if lgb is None:
        raise ImportError("lightgbm is required for LGBM model. Install with: pip install lightgbm>=4.0")
    if train_df.empty:
        return None
    X = _prepare_feature_frame(train_df, feature_cols).astype(float)
    y = train_df.loc[:, label_col].to_numpy(dtype=float, copy=False)
    mask = np.isfinite(X.to_numpy(dtype=float, copy=False)).all(axis=1) & np.isfinite(y)
    X = X.loc[mask]
    y = y[mask]
    if len(X) < 1:
        return None

    params = {
        "objective": "regression",
        "metric": "mse",
        "n_estimators": n_estimators if n_estimators is not None else 200,
        "max_depth": max_depth if max_depth is not None else 4,
        "learning_rate": learning_rate if learning_rate is not None else 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return {
        "type": "lgbm",
        "model_obj": model,
        "features": list(feature_cols),
    }


def train_ensemble_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    ridge_alpha: float = 1.0,
    weight_ridge: float = 0.5,
    lgbm_n_estimators: Optional[int] = None,
    lgbm_max_depth: Optional[int] = None,
    lgbm_learning_rate: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    """Train both Ridge and LGBM, return ensemble model dict."""
    ridge_model = train_ridge_model(train_df, feature_cols, label_col, alpha=ridge_alpha)
    lgbm_model = train_lgb_model(
        train_df,
        feature_cols,
        label_col,
        n_estimators=lgbm_n_estimators,
        max_depth=lgbm_max_depth,
        learning_rate=lgbm_learning_rate,
    )
    if ridge_model is None and lgbm_model is None:
        return None
    if ridge_model is not None and lgbm_model is None:
        return ridge_model
    if lgbm_model is not None and ridge_model is None:
        return lgbm_model

    return {
        "type": "ensemble",
        "weight_ridge": float(weight_ridge),
        "ridge_model": ridge_model,
        "lgbm_model": lgbm_model,
        "features": list(feature_cols),
    }


def train_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    model_type: str = "ridge",
    ridge_alpha: float = 1.0,
    ensemble_weight_ridge: float = 0.5,
    lgbm_n_estimators: Optional[int] = None,
    lgbm_max_depth: Optional[int] = None,
    lgbm_learning_rate: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    model_type = str(model_type).strip().lower()
    if model_type in ("ridge", "linear"):
        return train_ridge_model(train_df, feature_cols, label_col, alpha=ridge_alpha)
    if model_type in ("lgbm", "lightgbm"):
        return train_lgb_model(
            train_df,
            feature_cols,
            label_col,
            n_estimators=lgbm_n_estimators,
            max_depth=lgbm_max_depth,
            learning_rate=lgbm_learning_rate,
        )
    if model_type == "ensemble":
        return train_ensemble_model(
            train_df,
            feature_cols,
            label_col,
            ridge_alpha=ridge_alpha,
            weight_ridge=ensemble_weight_ridge,
            lgbm_n_estimators=lgbm_n_estimators,
            lgbm_max_depth=lgbm_max_depth,
            lgbm_learning_rate=lgbm_learning_rate,
        )
    raise ValueError(f"Unsupported training.model_type: {model_type}")


def predict_linear_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    return predict_ridge_model(model, features_df, feature_cols)


def predict_ridge_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    X = _prepare_feature_frame(features_df, feature_cols).to_numpy(dtype=float, copy=False)
    coef = np.array(model.get("coef", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))
    if X.ndim == 2 and coef.size != X.shape[1]:
        raise ValueError("Ridge model coefficients and feature columns length mismatch")
    return intercept + X.dot(coef)


def predict_lgb_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    model_obj = model.get("model_obj")
    if model_obj is None:
        raise ValueError("LGBM model object is missing in model payload")
    X = _prepare_feature_frame(features_df, feature_cols).astype(float)
    return np.asarray(model_obj.predict(X), dtype=float)


def predict_ensemble_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    """Blend Ridge and LGBM predictions with configurable weight."""
    weight_ridge = float(model.get("weight_ridge", 0.5))
    ridge_model = model.get("ridge_model")
    lgbm_model = model.get("lgbm_model")

    if ridge_model is None and lgbm_model is None:
        raise ValueError("Ensemble model has no sub-models")

    if ridge_model is not None and lgbm_model is not None:
        pred_ridge = predict_ridge_model(ridge_model, features_df, feature_cols)
        pred_lgbm = predict_lgb_model(lgbm_model, features_df, feature_cols)
        return weight_ridge * pred_ridge + (1.0 - weight_ridge) * pred_lgbm

    if ridge_model is not None:
        return predict_ridge_model(ridge_model, features_df, feature_cols)
    return predict_lgb_model(lgbm_model, features_df, feature_cols)


def predict_model(
    model: Dict[str, object],
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    model_type = str(model.get("type", "ridge")).strip().lower()
    if model_type in ("ridge", "linear"):
        return predict_ridge_model(model, features_df, feature_cols)
    if model_type in ("lgbm", "lightgbm"):
        return predict_lgb_model(model, features_df, feature_cols)
    if model_type == "ensemble":
        return predict_ensemble_model(model, features_df, feature_cols)
    raise ValueError(f"Unsupported model type in payload: {model_type}")
