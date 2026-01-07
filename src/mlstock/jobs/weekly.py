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
from mlstock.risk.vol_cap import apply_vol_cap, apply_vol_penalty


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

    data_max_features_date = max(features_df["week_start"]) if not features_df.empty else None
    data_max_labels_date = max(labels_df["week_start"]) if not labels_df.empty else None

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
    gate_active = False
    data_max_week_map_date = None
    week_map_df = None
    if gate_cfg.enabled:
        week_map_df = read_parquet(snapshots_week_map_path(cfg))
        if not week_map_df.empty:
            week_map_df["week_start"] = _to_date(week_map_df["week_start"])
            data_max_week_map_date = max(week_map_df["week_start"])
        gate_result = build_spy_regime_gate(cfg, week_map_df, features_df)
        gate_open = gate_result.open_by_week.get(current_week, True)
        gate_active = not gate_open
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
    else:
        week_map_path = snapshots_week_map_path(cfg)
        if week_map_path.exists():
            try:
                week_map_df = pd.read_parquet(week_map_path, columns=["week_start"])
                if not week_map_df.empty:
                    week_map_df["week_start"] = _to_date(week_map_df["week_start"])
                    data_max_week_map_date = max(week_map_df["week_start"])
            except Exception as exc:
                log_event(logger, "week_map_read_error", error=str(exc))

    if data_max_features_date or data_max_labels_date or data_max_week_map_date:
        log_event(
            logger,
            "data_max_dates",
            features=data_max_features_date.isoformat() if data_max_features_date else None,
            labels=data_max_labels_date.isoformat() if data_max_labels_date else None,
            week_map=data_max_week_map_date.isoformat() if data_max_week_map_date else None,
        )

    vol_cap_cfg = cfg.risk.vol_cap
    vol_cap_enabled = vol_cap_cfg.enabled
    vol_cap_stage = vol_cap_cfg.apply_stage.strip().lower()
    vol_cap_mode = vol_cap_cfg.mode.strip().lower()
    vol_cap_penalty_min = float(vol_cap_cfg.penalty_min)
    vol_cap_soft = vol_cap_mode in ("soft", "soft_penalty", "penalty")
    vol_cap_apply_to_selection = False
    vol_cap_apply_to_training = False
    vol_cap_hold_buffer = 0.0
    vol_cap_hold_threshold = min(1.0, float(vol_cap_cfg.rank_threshold) + vol_cap_hold_buffer)
    if vol_cap_enabled:
        if vol_cap_mode not in ("hard", "soft", "soft_penalty", "penalty"):
            log_event(logger, "unknown_vol_cap_mode", value=vol_cap_cfg.mode)
            vol_cap_enabled = False
        if vol_cap_stage == "selection":
            vol_cap_apply_to_selection = True
        elif vol_cap_stage == "training":
            vol_cap_apply_to_training = True
        elif vol_cap_stage in ("training+selection", "training_and_selection", "training_selection", "both"):
            vol_cap_apply_to_training = True
            vol_cap_apply_to_selection = True
        else:
            log_event(logger, "unknown_vol_cap_stage", value=vol_cap_cfg.apply_stage)
            vol_cap_enabled = False
    if vol_cap_enabled and not (0.0 < vol_cap_penalty_min <= 1.0):
        log_event(logger, "vol_cap_penalty_min_invalid", value=vol_cap_penalty_min)
        vol_cap_penalty_min = 1.0

    state_dir = artifacts_state_dir(cfg)
    state_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = state_dir / "portfolio.json"
    raw_state = read_state(portfolio_path)
    state = _normalize_portfolio_state(raw_state, cfg)
    positions = state["positions"]
    hold_symbols = {str(symbol) for symbol in positions.keys()}

    guard_cfg = cfg.risk.exposure_guard
    guard_enabled = guard_cfg.enabled
    guard_active = False
    guard_base_scale = 1.0
    guard_cap_enabled = False
    gross_cap = None
    if guard_enabled:
        if guard_cfg.trigger == "vol_cap_enabled":
            guard_active = vol_cap_enabled
        elif guard_cfg.trigger == "always":
            guard_active = True
        else:
            log_event(logger, "unknown_exposure_guard_trigger", value=guard_cfg.trigger)
            guard_enabled = False
    if guard_enabled and guard_active:
        if guard_cfg.mode not in ("daily", "weekly_aligned"):
            log_event(logger, "unknown_exposure_guard_mode", value=guard_cfg.mode)
            guard_enabled = False
            guard_active = False
        if guard_cfg.base_source not in ("off_avg_on_avg", "fixed", "none"):
            log_event(logger, "unknown_exposure_guard_base_source", value=guard_cfg.base_source)
            guard_enabled = False
            guard_active = False
        if guard_cfg.cap_source not in ("off_p95", "off_avg", "fixed", "none"):
            log_event(logger, "unknown_exposure_guard_cap_source", value=guard_cfg.cap_source)
            guard_enabled = False
            guard_active = False
        if guard_active:
            if guard_cfg.base_source == "none":
                guard_base_scale = 1.0
            elif guard_cfg.base_scale is None:
                log_event(logger, "exposure_guard_base_scale_missing", source=guard_cfg.base_source)
                guard_base_scale = 1.0
            else:
                guard_base_scale = float(guard_cfg.base_scale)
                if guard_base_scale <= 0:
                    log_event(logger, "exposure_guard_base_scale_invalid", value=guard_base_scale)
                    guard_base_scale = 1.0
                elif guard_base_scale > 1.0:
                    log_event(logger, "exposure_guard_base_scale_capped", value=guard_base_scale)
                    guard_base_scale = 1.0

            if guard_cfg.cap_source == "none":
                guard_cap_enabled = False
            elif guard_cfg.cap_value is None:
                log_event(logger, "exposure_guard_cap_missing", source=guard_cfg.cap_source)
                guard_cap_enabled = False
            else:
                cap_value = float(guard_cfg.cap_value)
                if cap_value <= 0:
                    log_event(logger, "exposure_guard_cap_invalid", value=cap_value)
                    guard_cap_enabled = False
                else:
                    gross_cap = cap_value + float(guard_cfg.cap_buffer)
                    if gross_cap <= 0:
                        log_event(logger, "exposure_guard_cap_invalid", value=gross_cap)
                        guard_cap_enabled = False
                    else:
                        guard_cap_enabled = True

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
    if vol_cap_enabled and vol_cap_apply_to_training and not vol_cap_soft:
        try:
            train_df, _ = apply_vol_cap(
                train_df,
                feature_name=vol_cap_cfg.feature_name,
                rank_threshold=vol_cap_cfg.rank_threshold,
                group_by="week_start",
                enabled=vol_cap_enabled,
            )
        except ValueError as exc:
            log_event(logger, "vol_cap_error", error=str(exc))
            raise
        if train_df.empty:
            raise ValueError("No training data after vol cap filtering")

    model = train_linear_model(train_df, FEATURE_COLUMNS, "label_return")
    if model is None:
        raise ValueError("Training failed for weekly run")

    week_all = features_df[features_df["week_start"] == current_week].copy()
    if week_all.empty:
        raise ValueError("No current-week features available")

    week_features = week_all[week_all["avg_dollar_vol_20d"].fillna(0) >= float(cfg.snapshots.min_avg_dollar_vol_20d)]
    week_features = week_features.dropna(subset=FEATURE_COLUMNS)
    week_features = week_features[week_features["price"].notna()]
    week_features = week_features[week_features["price"] <= float(cfg.selection.price_cap)]

    if week_features.empty:
        raise ValueError("No eligible symbols for weekly selection")

    vol_cap_candidates = None
    vol_cap_excluded = None
    vol_cap_missing = None
    vol_cap_penalized = None
    use_vol_penalty = vol_cap_enabled and vol_cap_apply_to_selection and vol_cap_soft
    if vol_cap_enabled and vol_cap_apply_to_selection and not vol_cap_soft:
        try:
            week_features, vol_stats = apply_vol_cap(
                week_features,
                feature_name=vol_cap_cfg.feature_name,
                rank_threshold=vol_cap_cfg.rank_threshold,
                hold_symbols=hold_symbols,
                hold_threshold=vol_cap_hold_threshold,
                enabled=vol_cap_enabled,
            )
        except ValueError as exc:
            log_event(logger, "vol_cap_error", error=str(exc))
            raise
        vol_cap_candidates = vol_stats.candidates
        vol_cap_excluded = vol_stats.excluded
        vol_cap_missing = vol_stats.missing
        if week_features.empty:
            raise ValueError("No eligible symbols after vol cap filtering")

    preds = predict_linear_model(model, week_features, FEATURE_COLUMNS)
    week_features = week_features.assign(pred_return=preds)

    if use_vol_penalty:
        try:
            week_features, vol_stats = apply_vol_penalty(
                week_features,
                feature_name=vol_cap_cfg.feature_name,
                rank_threshold=vol_cap_cfg.rank_threshold,
                penalty_min=vol_cap_penalty_min,
                enabled=vol_cap_enabled,
            )
        except ValueError as exc:
            log_event(logger, "vol_cap_penalty_error", error=str(exc))
            raise
        vol_cap_candidates = vol_stats.candidates
        vol_cap_missing = vol_stats.missing
        vol_cap_penalized = vol_stats.penalized
        if week_features.empty:
            raise ValueError("No eligible symbols after vol cap penalty")
        week_features = week_features.assign(pred_return_raw=week_features["pred_return"])
        week_features = week_features.assign(
            pred_return=week_features["pred_return"] * week_features["vol_cap_penalty"]
        )

    min_buy = float(cfg.selection.min_proba_buy)
    min_keep = float(cfg.selection.min_proba_keep)
    if not (0.0 <= min_buy <= 1.0) or not (0.0 <= min_keep <= 1.0):
        raise ValueError("min_proba_buy/min_proba_keep must be within [0, 1]")
    if min_buy > 0.0 or min_keep > 0.0:
        week_features = week_features.assign(pred_rank=week_features["pred_return"].rank(pct=True, method="max"))
        primary = week_features
        if min_buy > 0.0:
            primary = week_features[week_features["pred_rank"] >= min_buy]
        if min_keep > 0.0 and min_keep < min_buy:
            secondary = week_features[(week_features["pred_rank"] >= min_keep) & (week_features["pred_rank"] < min_buy)]
            week_features = pd.concat([primary, secondary], ignore_index=False)
        else:
            week_features = primary
        if week_features.empty:
            raise ValueError("No eligible symbols after min_proba thresholds")

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
    max_positions = max(int(cfg.selection.max_positions), 1)
    if gate_cfg.enabled and not gate_open and gate_action == "no_trade":
        max_positions = 0
    deadband_abs = max(0.0, float(cfg.selection.deadband_abs))
    deadband_rel = max(0.0, float(cfg.selection.deadband_rel))
    min_trade_notional = max(0.0, float(cfg.selection.min_trade_notional))
    deadband_v2_enabled = bool(cfg.execution.deadband_v2.enabled)
    if not deadband_v2_enabled:
        deadband_abs = 0.0
        deadband_rel = 0.0
        min_trade_notional = 0.0
    nav_est = cash_start
    for symbol, qty in positions.items():
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price):
            continue
        nav_est += float(price) * int(qty)

    sell_orders: List[Dict[str, object]] = []
    unsold_positions: Dict[str, int] = {}
    kept_positions: Dict[str, int] = {}
    optional_keeps: List[tuple[str, int, float]] = []
    est_proceeds = 0.0
    missing_prices = 0
    ranked_symbols = [str(row.symbol) for row in week_features.itertuples(index=False)]
    target_symbols = ranked_symbols[:max_positions] if max_positions > 0 else []
    target_set = set(target_symbols)
    held_symbols = set(positions.keys())
    sum_abs_dw_raw = 0.0
    sum_abs_dw_filtered = 0.0
    buy_dw_filtered = 0.0
    sell_dw_filtered = 0.0
    trade_count_raw = 0
    trade_count_filtered = 0
    eps = 1e-12
    if nav_est > 0:
        if deadband_v2_enabled:
            for symbol, qty in positions.items():
                price = price_lookup.get(symbol)
                if price is None or pd.isna(price):
                    continue
                w_cur = float(price) * int(qty) / nav_est
                w_tgt = w_cur if symbol in target_set else 0.0
                dw = w_tgt - w_cur
                if dw != 0.0:
                    sum_abs_dw_raw += abs(dw)
                    if abs(dw) > eps:
                        trade_count_raw += 1
                    band = max(deadband_abs, deadband_rel * abs(w_tgt))
                    dw_filtered = dw
                    if abs(dw_filtered) < band:
                        dw_filtered = 0.0
                    if (
                        min_trade_notional > 0.0
                        and w_cur != 0.0
                        and w_tgt != 0.0
                        and abs(dw_filtered) < min_trade_notional
                    ):
                        dw_filtered = 0.0
                    sum_abs_dw_filtered += abs(dw_filtered)
                    if abs(dw_filtered) > eps:
                        trade_count_filtered += 1
                    if dw_filtered > 0.0:
                        buy_dw_filtered += dw_filtered
                    elif dw_filtered < 0.0:
                        sell_dw_filtered += abs(dw_filtered)

            for symbol in target_symbols:
                if symbol in held_symbols:
                    continue
                row = week_features.loc[week_features["symbol"] == symbol].head(1)
                if row.empty:
                    continue
                price = float(row.iloc[0].price)
                w_tgt = price / nav_est
                if w_tgt == 0.0:
                    continue
                sum_abs_dw_raw += abs(w_tgt)
                if abs(w_tgt) > eps:
                    trade_count_raw += 1
                band = max(deadband_abs, deadband_rel * abs(w_tgt))
                dw_filtered = w_tgt if abs(w_tgt) >= band else 0.0
                sum_abs_dw_filtered += abs(dw_filtered)
                if abs(dw_filtered) > eps:
                    trade_count_filtered += 1
                if dw_filtered > 0.0:
                    buy_dw_filtered += dw_filtered
        else:
            for symbol, qty in positions.items():
                price = price_lookup.get(symbol)
                if price is None or pd.isna(price):
                    continue
                w_cur = float(price) * int(qty) / nav_est
                w_tgt = w_cur if symbol in target_set else 0.0
                dw = w_tgt - w_cur
                if dw != 0.0:
                    sum_abs_dw_raw += abs(dw)
                    sum_abs_dw_filtered += abs(dw)
                    if abs(dw) > eps:
                        trade_count_raw += 1
                        trade_count_filtered += 1
                    if dw > 0.0:
                        buy_dw_filtered += dw
                    elif dw < 0.0:
                        sell_dw_filtered += abs(dw)

            for symbol in target_symbols:
                if symbol in held_symbols:
                    continue
                row = week_features.loc[week_features["symbol"] == symbol].head(1)
                if row.empty:
                    continue
                price = float(row.iloc[0].price)
                w_tgt = price / nav_est
                if w_tgt == 0.0:
                    continue
                sum_abs_dw_raw += abs(w_tgt)
                sum_abs_dw_filtered += abs(w_tgt)
                if abs(w_tgt) > eps:
                    trade_count_raw += 1
                    trade_count_filtered += 1
                if w_tgt > 0.0:
                    buy_dw_filtered += w_tgt
    for symbol, qty in positions.items():
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price):
            missing_prices += 1
            unsold_positions[symbol] = qty
            continue
        qty_int = int(qty)
        est_price = float(price)
        w_cur = est_price * qty_int / nav_est if nav_est > 0 else 0.0
        w_tgt = w_cur if symbol in target_set else 0.0
        delta = w_tgt - w_cur
        keep = False
        if symbol in target_set:
            keep = True
        elif max_positions == 0:
            keep = False
        else:
            if deadband_v2_enabled:
                band = max(deadband_abs, deadband_rel * abs(w_tgt))
                if abs(delta) < band:
                    keep = True
                elif (
                    min_trade_notional > 0.0
                    and w_cur != 0.0
                    and w_tgt != 0.0
                    and abs(delta) < min_trade_notional
                ):
                    keep = True
        if keep:
            kept_positions[symbol] = qty_int
            if symbol not in target_set:
                optional_keeps.append((symbol, qty_int, est_price))
            continue
        sell_cost = est_price * qty_int * cost_rate
        est_proceeds += est_price * qty_int - sell_cost
        sell_orders.append(
            {
                "side": "sell",
                "symbol": symbol,
                "qty": qty_int,
                "type": "market",
                "time_in_force": "day",
            }
        )

    held_count = len(kept_positions) + len(unsold_positions)
    if max_positions > 0 and held_count > max_positions and optional_keeps:
        excess = held_count - max_positions
        optional_keeps.sort(key=lambda item: item[2] * item[1], reverse=True)
        for symbol, qty_int, est_price in optional_keeps[:excess]:
            if symbol not in kept_positions:
                continue
            del kept_positions[symbol]
            sell_cost = est_price * qty_int * cost_rate
            est_proceeds += est_price * qty_int - sell_cost
            sell_orders.append(
                {
                    "side": "sell",
                    "symbol": symbol,
                    "qty": qty_int,
                    "type": "market",
                    "time_in_force": "day",
                }
            )
            held_count -= 1

    cash_est_before_buys = cash_start + est_proceeds
    cash_est = cash_est_before_buys

    buy_orders: List[Dict[str, object]] = []
    candidates: List[Dict[str, object]] = []
    selected_symbols: List[str] = []
    selected_count = 0
    skipped_buys = 0
    total_required = 0.0
    positions_value_est = 0.0
    for symbol, qty in kept_positions.items():
        price = price_lookup.get(symbol)
        if price is None or pd.isna(price):
            continue
        positions_value_est += float(price) * int(qty)

    feature_by_symbol = {str(row.symbol): row for row in week_features.itertuples(index=False)}
    for rank, symbol in enumerate(target_symbols, start=1):
        row = feature_by_symbol.get(symbol)
        if row is None:
            continue
        est_price = float(row.price) * (1.0 + buffer_bps / 10000.0)
        est_cost = est_price * cost_rate
        required = est_price + est_cost
        target_weight = float(row.price) / nav_est if nav_est > 0 else 0.0
        within_min_trade = False
        if deadband_v2_enabled:
            band = max(deadband_abs, deadband_rel * abs(target_weight))
            within_band = abs(target_weight) < band
        else:
            within_band = False
        budget_ok = cash_est - reserve >= required
        selected = False

        if not within_band and not within_min_trade:
            if held_count + selected_count < max_positions and budget_ok:
                if symbol not in kept_positions and symbol not in unsold_positions:
                    selected = True
                    selected_count += 1
                    cash_est -= required
                    total_required += required
                    positions_value_est += est_price
                    selected_symbols.append(symbol)
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
            elif not budget_ok:
                skipped_buys += 1
                if buy_fill_policy == "ranked_strict":
                    candidates.append(
                        {
                            "rank": rank,
                            "symbol": symbol,
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
                "rank": rank,
                "symbol": symbol,
                "pred_return": float(row.pred_return),
                "est_price": est_price,
                "est_cost": est_cost,
                "required_est": required,
                "budget_ok": budget_ok,
                "selected": selected,
            }
        )

    total_positions = held_count + selected_count
    exposure_guard_scale = None
    gross_exposure_raw = None
    gross_exposure_guarded = None
    cash_est_guarded = None
    positions_value_guarded = None
    if guard_enabled and guard_active and positions_value_est > 0.0:
        total_required_scaled = total_required * guard_base_scale
        positions_value_scaled = positions_value_est * guard_base_scale
        cash_est_scaled = cash_est_before_buys - total_required_scaled
        nav_est = cash_est_scaled + positions_value_scaled
        if nav_est > 0.0:
            gross_exposure_raw = positions_value_scaled / nav_est
            cap_scale = 1.0
            if guard_cap_enabled and gross_cap is not None and gross_exposure_raw > gross_cap:
                cap_scale = gross_cap / gross_exposure_raw
            exposure_guard_scale = guard_base_scale * cap_scale
            positions_value_guarded = positions_value_est * exposure_guard_scale
            cash_est_guarded = cash_est_before_buys - total_required * exposure_guard_scale
            nav_guarded = cash_est_guarded + positions_value_guarded
            if nav_guarded > 0.0:
                gross_exposure_guarded = positions_value_guarded / nav_guarded

    filtered_trade_fraction_notional = 0.0
    if sum_abs_dw_raw > 0.0:
        filtered_trade_fraction_notional = 1.0 - (sum_abs_dw_filtered / sum_abs_dw_raw)
        filtered_trade_fraction_notional = max(0.0, min(1.0, filtered_trade_fraction_notional))
    filtered_trade_fraction = filtered_trade_fraction_notional
    deadband_notional_reduction = filtered_trade_fraction_notional
    filtered_trade_fraction_count = 0.0
    if trade_count_raw > 0:
        filtered_trade_fraction_count = (trade_count_raw - trade_count_filtered) / trade_count_raw
        filtered_trade_fraction_count = max(0.0, min(1.0, filtered_trade_fraction_count))
    turnover_ratio_std = buy_dw_filtered
    turnover_ratio_buy = buy_dw_filtered
    turnover_ratio_sell = sell_dw_filtered
    turnover_ratio_total_abs = turnover_ratio_buy + turnover_ratio_sell
    turnover_ratio_total_half = 0.5 * turnover_ratio_total_abs
    cash_after_exec = cash_est

    orders = sell_orders + buy_orders
    orders_path = orders_dir / f"orders_{stamp}.csv"
    pd.DataFrame(orders).to_csv(orders_path, index=False)

    candidates_path = orders_dir / f"orders_candidates_{stamp}.csv"
    pd.DataFrame(candidates).to_csv(candidates_path, index=False)

    sell_symbols = [str(order["symbol"]) for order in sell_orders if order.get("symbol")]
    keep_symbols = [str(symbol) for symbol in kept_positions.keys()] + [
        str(symbol) for symbol in unsold_positions.keys()
    ]

    selection_payload = {
        "as_of": today.isoformat(),
        "week_start": current_week.isoformat(),
        "symbols": selected_symbols,
        "buy_symbols": selected_symbols,
        "sell_symbols": sell_symbols,
        "keep_symbols": keep_symbols,
        "target_symbols": target_symbols,
        "n_selected": selected_count,
        "cash_start_usd": cash_start,
        "cash_reserve_usd": reserve,
        "cash_est_before_buys": cash_est_before_buys,
        "cash_est_after_buys": cash_est,
        "skipped_buys_insufficient_cash": skipped_buys,
        "buy_fill_policy": buy_fill_policy,
        "estimate_entry_buffer_bps": buffer_bps,
        "missing_sell_prices": missing_prices,
        "deadband_v2_enabled": deadband_v2_enabled,
        "deadband_abs": deadband_abs,
        "deadband_rel": deadband_rel,
        "min_trade_notional": min_trade_notional,
        "sum_abs_dw_raw": sum_abs_dw_raw,
        "sum_abs_dw_filtered": sum_abs_dw_filtered,
        "deadband_notional_reduction": deadband_notional_reduction,
        "filtered_trade_fraction_notional": filtered_trade_fraction_notional,
        "filtered_trade_fraction": filtered_trade_fraction,
        "filtered_trade_fraction_count": filtered_trade_fraction_count,
        "trade_count_raw": trade_count_raw,
        "trade_count_filtered": trade_count_filtered,
        "turnover_ratio_std": turnover_ratio_std,
        "turnover_ratio_buy": turnover_ratio_buy,
        "turnover_ratio_sell": turnover_ratio_sell,
        "turnover_ratio_total_abs": turnover_ratio_total_abs,
        "turnover_ratio_total_half": turnover_ratio_total_half,
        "cash_after_exec": cash_after_exec,
        "kept_positions": len(kept_positions),
        "held_positions": total_positions,
        "data_max_features_date": data_max_features_date.isoformat() if data_max_features_date else None,
        "data_max_labels_date": data_max_labels_date.isoformat() if data_max_labels_date else None,
        "data_max_week_map_date": data_max_week_map_date.isoformat() if data_max_week_map_date else None,
        "regime_gate": {
            "enabled": gate_cfg.enabled,
            "active": gate_active,
            "open": gate_open,
            "rule": gate_cfg.rule,
            "action": gate_action,
            "source": gate_result.source if gate_result else None,
            "ma_days": gate_result.ma_days if gate_result else None,
            "pred_return_floor": gate_pred_floor,
        },
        "vol_cap": {
            "enabled": vol_cap_enabled,
            "mode": vol_cap_cfg.mode if vol_cap_enabled else None,
            "penalty_min": float(vol_cap_cfg.penalty_min) if vol_cap_enabled else None,
            "apply_stage": vol_cap_cfg.apply_stage if vol_cap_enabled else None,
            "apply_to_training": vol_cap_apply_to_training if vol_cap_enabled else None,
            "apply_to_selection": vol_cap_apply_to_selection if vol_cap_enabled else None,
            "feature_name": vol_cap_cfg.feature_name if vol_cap_enabled else None,
            "rank_threshold": float(vol_cap_cfg.rank_threshold) if vol_cap_enabled else None,
            "candidates": vol_cap_candidates,
            "excluded": vol_cap_excluded,
            "missing": vol_cap_missing,
            "penalized": vol_cap_penalized,
        },
        "exposure_guard": {
            "enabled": guard_enabled,
            "active": guard_active,
            "trigger": guard_cfg.trigger if guard_enabled else None,
            "mode": guard_cfg.mode if guard_enabled else None,
            "base_source": guard_cfg.base_source if guard_enabled else None,
            "base_scale": guard_cfg.base_scale if guard_enabled else None,
            "cap_source": guard_cfg.cap_source if guard_enabled else None,
            "cap_value": guard_cfg.cap_value if guard_enabled else None,
            "cap_buffer": float(guard_cfg.cap_buffer) if guard_enabled else None,
            "cap": gross_cap,
            "cap_enabled": guard_cap_enabled if guard_enabled else None,
            "gross_exposure_raw": gross_exposure_raw,
            "scale": exposure_guard_scale,
            "gross_exposure_guarded": gross_exposure_guarded,
            "cash_est_after_buys_guarded": cash_est_guarded,
        },
    }
    selection_path = orders_dir / f"selection_{stamp}.json"
    write_state(selection_payload, selection_path)

    next_positions = dict(unsold_positions)
    for symbol, qty in kept_positions.items():
        next_positions[symbol] = int(next_positions.get(symbol, 0)) + int(qty)
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
