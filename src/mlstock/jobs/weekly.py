from __future__ import annotations

from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import (
    artifacts_models_dir,
    artifacts_orders_dir,
    artifacts_state_dir,
    snapshots_features_path,
    snapshots_labels_path,
    snapshots_week_map_path,
)
from mlstock.data.storage.state import read_state, write_state
from mlstock.logging.logger import build_log_path, log_event, setup_logger
from mlstock.model.features import FEATURE_COLUMNS
from mlstock.model.train import predict_linear_model, select_training_weeks, train_linear_model
from mlstock.risk.regime import build_spy_regime_gate


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def _normalize_portfolio_state(state: Dict[str, object], cfg: AppConfig) -> Dict[str, object]:
    cash = state.get("cash_usd")
    if cash is None:
        cash = float(cfg.selection.cash_start_usd)

    positions: Dict[str, int] = {}
    raw_positions = state.get("positions")
    if isinstance(raw_positions, dict):
        for symbol, qty in raw_positions.items():
            try:
                qty_int = int(qty)
            except (TypeError, ValueError):
                continue
            if qty_int:
                positions[str(symbol)] = qty_int
    else:
        symbol = state.get("symbol")
        qty = state.get("qty", 0)
        if symbol and qty:
            positions[str(symbol)] = int(qty)

    return {
        "cash_usd": float(cash),
        "positions": positions,
    }


def run(cfg: AppConfig) -> Dict[str, object]:
    log_path = build_log_path(cfg, "weekly")
    logger = setup_logger("weekly", log_path, cfg.logging.level)
    log_event(logger, "start")

    features_df = read_parquet(snapshots_features_path(cfg))
    labels_df = read_parquet(snapshots_labels_path(cfg))
    if features_df.empty:
        raise ValueError("Snapshots features are empty")

    features_df["week_start"] = _to_date(features_df["week_start"])
    if not labels_df.empty:
        labels_df["week_start"] = _to_date(labels_df["week_start"])

    exclude_symbols = {symbol.upper() for symbol in cfg.snapshots.exclude_symbols}
    if exclude_symbols:
        features_df = features_df[~features_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
        if not labels_df.empty:
            labels_df = labels_df[~labels_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]

    current_week = max(features_df["week_start"])
    available_weeks = sorted(labels_df["week_start"].unique().tolist()) if not labels_df.empty else []

    gate_cfg = cfg.risk.regime_gate
    gate_action = gate_cfg.action
    if gate_action not in ("no_trade", "raise_threshold"):
        log_event(logger, "unknown_regime_gate_action", value=gate_action)
        gate_action = "no_trade"

    gate_result = None
    gate_open = True
    if gate_cfg.enabled:
        week_map_df = read_parquet(snapshots_week_map_path(cfg))
        gate_result = build_spy_regime_gate(cfg, week_map_df, features_df)
        gate_open = gate_result.open_by_week.get(current_week, True)
        log_event(
            logger,
            "regime_gate_status",
            week_start=current_week.isoformat(),
            open=gate_open,
            rule=gate_cfg.rule,
            action=gate_action,
            source=gate_result.source,
            ma_days=gate_result.ma_days,
            missing_weeks=gate_result.missing_weeks,
        )

    train_weeks = select_training_weeks(
        available_weeks,
        current_week,
        train_window_years=cfg.training.train_window_years,
        min_train_weeks=cfg.training.min_train_weeks,
    )
    if not train_weeks:
        raise ValueError("Not enough training weeks for weekly run")

    train_df = features_df.merge(labels_df, on=["week_start", "symbol"], how="inner")
    train_df = train_df[train_df["week_start"].isin(train_weeks)]

    model = train_linear_model(train_df, FEATURE_COLUMNS, "label_return")
    if model is None:
        raise ValueError("Training failed for weekly run")

    week_all = features_df[features_df["week_start"] == current_week].copy()
    if week_all.empty:
        raise ValueError("No current-week features available")

    week_features = week_all[
        week_all["avg_dollar_vol_20d"].fillna(0) >= float(cfg.snapshots.min_avg_dollar_vol_20d)
    ]
    week_features = week_features.dropna(subset=FEATURE_COLUMNS)
    week_features = week_features[week_features["price"].notna()]
    week_features = week_features[week_features["price"] <= float(cfg.selection.price_cap)]

    if week_features.empty:
        raise ValueError("No eligible symbols for weekly selection")

    preds = predict_linear_model(model, week_features, FEATURE_COLUMNS)
    week_features = week_features.assign(pred_return=preds)
    week_features = week_features.sort_values("pred_return", ascending=False)

    gate_pred_floor = None
    if gate_cfg.enabled and not gate_open and gate_action == "raise_threshold":
        gate_pred_floor = float(gate_cfg.pred_return_floor)
        week_features = week_features[week_features["pred_return"] >= gate_pred_floor]

    tz = ZoneInfo(cfg.project.timezone)
    today = datetime.now(tz).date()
    stamp = today.strftime("%Y%m%d")

    models_dir = artifacts_models_dir(cfg)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"model_{stamp}.joblib"
    joblib.dump(model, model_path)

    pred_path = models_dir / f"pred_{stamp}.parquet"
    write_parquet_atomic(week_features, pred_path)

    orders_dir = artifacts_orders_dir(cfg)
    orders_dir.mkdir(parents=True, exist_ok=True)
    state_dir = artifacts_state_dir(cfg)
    state_dir.mkdir(parents=True, exist_ok=True)

    portfolio_path = state_dir / "portfolio.json"
    raw_state = read_state(portfolio_path)
    state = _normalize_portfolio_state(raw_state, cfg)
    cash_start = float(state["cash_usd"])
    positions = state["positions"]

    price_lookup = dict(zip(week_all["symbol"], week_all["price"]))
    cost_rate = float(cfg.cost_model.bps_per_side) / 10000.0
    reserve = float(cfg.selection.cash_reserve_usd)
    buffer_bps = float(cfg.selection.estimate_entry_buffer_bps)
    buy_fill_policy = cfg.selection.buy_fill_policy
    if buy_fill_policy not in ("ranked_partial", "ranked_strict"):
        log_event(logger, "unknown_buy_fill_policy", value=buy_fill_policy)
        buy_fill_policy = "ranked_partial"

    sell_orders: List[Dict[str, object]] = []
    unsold_positions: Dict[str, int] = {}
    est_proceeds = 0.0
    missing_prices = 0
    for symbol, qty in positions.items():
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price):
            missing_prices += 1
            unsold_positions[symbol] = qty
            continue
        est_price = float(price)
        sell_cost = est_price * qty * cost_rate
        est_proceeds += est_price * qty - sell_cost
        sell_orders.append(
            {
                "side": "sell",
                "symbol": symbol,
                "qty": int(qty),
                "type": "market",
                "time_in_force": "day",
            }
        )

    cash_est = cash_start + est_proceeds

    buy_orders: List[Dict[str, object]] = []
    candidates: List[Dict[str, object]] = []
    selected_symbols: List[str] = []
    selected_count = 0
    skipped_buys = 0
    max_positions = max(int(cfg.selection.max_positions), 1)
    if gate_cfg.enabled and not gate_open and gate_action == "no_trade":
        max_positions = 0

    for row in week_features.itertuples(index=False):
        est_price = float(row.price) * (1.0 + buffer_bps / 10000.0)
        est_cost = est_price * cost_rate
        required = est_price + est_cost
        budget_ok = cash_est - reserve >= required
        selected = False

        if selected_count < max_positions and budget_ok:
            selected = True
            selected_count += 1
            cash_est -= required
            selected_symbols.append(str(row.symbol))
            buy_orders.append(
                {
                    "side": "buy",
                    "symbol": row.symbol,
                    "qty": 1,
                    "type": "market",
                    "time_in_force": "day",
                    "priority": selected_count,
                    "est_price": est_price,
                    "est_cost": est_cost,
                    "required_est": required,
                }
            )
        else:
            if not budget_ok:
                skipped_buys += 1
                if buy_fill_policy == "ranked_strict":
                    candidates.append(
                        {
                            "rank": len(candidates) + 1,
                            "symbol": row.symbol,
                            "pred_return": float(row.pred_return),
                            "est_price": est_price,
                            "est_cost": est_cost,
                            "required_est": required,
                            "budget_ok": budget_ok,
                            "selected": False,
                        }
                    )
                    break

        candidates.append(
            {
                "rank": len(candidates) + 1,
                "symbol": row.symbol,
                "pred_return": float(row.pred_return),
                "est_price": est_price,
                "est_cost": est_cost,
                "required_est": required,
                "budget_ok": budget_ok,
                "selected": selected,
            }
        )

    orders = sell_orders + buy_orders
    orders_path = orders_dir / f"orders_{stamp}.csv"
    pd.DataFrame(orders).to_csv(orders_path, index=False)

    candidates_path = orders_dir / f"orders_candidates_{stamp}.csv"
    pd.DataFrame(candidates).to_csv(candidates_path, index=False)

    selection_payload = {
        "as_of": today.isoformat(),
        "week_start": current_week.isoformat(),
        "symbols": selected_symbols,
        "n_selected": selected_count,
        "cash_start_usd": cash_start,
        "cash_reserve_usd": reserve,
        "cash_est_before_buys": cash_start + est_proceeds,
        "cash_est_after_buys": cash_est,
        "skipped_buys_insufficient_cash": skipped_buys,
        "buy_fill_policy": buy_fill_policy,
        "estimate_entry_buffer_bps": buffer_bps,
        "missing_sell_prices": missing_prices,
        "regime_gate": {
            "enabled": gate_cfg.enabled,
            "open": gate_open,
            "rule": gate_cfg.rule,
            "action": gate_action,
            "source": gate_result.source if gate_result else None,
            "ma_days": gate_result.ma_days if gate_result else None,
            "pred_return_floor": gate_pred_floor,
        },
    }
    selection_path = orders_dir / f"selection_{stamp}.json"
    write_state(selection_payload, selection_path)

    next_positions = dict(unsold_positions)
    for symbol in selected_symbols:
        next_positions[symbol] = int(next_positions.get(symbol, 0)) + 1

    next_state = {
        "as_of": today.isoformat(),
        "week_start": current_week.isoformat(),
        "cash_usd": cash_est,
        "positions": next_positions,
    }
    write_state(next_state, portfolio_path)

    log_event(logger, "complete", buys=selected_count, sells=len(sell_orders), orders=len(orders))
    return selection_payload
