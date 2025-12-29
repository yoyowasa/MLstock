from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic
from mlstock.data.storage.paths import (
    artifacts_backtest_dir,
    snapshots_features_path,
    snapshots_labels_path,
    snapshots_week_map_path,
)
from mlstock.data.storage.state import write_state
from mlstock.logging.logger import build_log_path, log_event, setup_logger
from mlstock.model.features import FEATURE_COLUMNS
from mlstock.model.train import predict_linear_model, select_training_weeks, train_linear_model
from mlstock.risk.regime import build_spy_regime_gate


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def _format_week(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _maybe_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)



def _maybe_int(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _with_prev_fields(nav_df: pd.DataFrame) -> pd.DataFrame:
    if nav_df.empty:
        return nav_df.copy()
    nav_df = nav_df.copy()
    nav_df["nav_prev"] = nav_df["nav"].shift(1)
    nav_df["cash_prev"] = nav_df["cash_usd"].shift(1)
    nav_df["positions_value_prev"] = nav_df["positions_value"].shift(1)
    nav_df["weekly_pnl"] = nav_df["nav"] - nav_df["nav_prev"]
    nav_prev = pd.to_numeric(nav_df["nav_prev"], errors="coerce").mask(lambda s: s == 0)
    nav_df["pct_change"] = nav_df["weekly_pnl"] / nav_prev
    return nav_df


def _select_top_contributors(
    trades_df: pd.DataFrame,
    week_start: object,
    *,
    mode: str,
    top_n: int = 5,
) -> List[Dict[str, object]]:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return []
    trades_week = trades_df[trades_df["week_start"] == week_start]
    if trades_week.empty:
        return []
    pnl_by_symbol = trades_week.groupby("symbol", as_index=False)["pnl"].sum()
    if pnl_by_symbol.empty:
        return []
    if mode == "negative":
        pnl_by_symbol = pnl_by_symbol.sort_values("pnl", ascending=True)
    elif mode == "positive":
        pnl_by_symbol = pnl_by_symbol.sort_values("pnl", ascending=False)
    else:
        pnl_by_symbol["abs_pnl"] = pnl_by_symbol["pnl"].abs()
        pnl_by_symbol = pnl_by_symbol.sort_values("abs_pnl", ascending=False)
    contributors: List[Dict[str, object]] = []
    for _, item in pnl_by_symbol.head(top_n).iterrows():
        contributors.append({"symbol": item["symbol"], "pnl": float(item["pnl"])})
    return contributors


def _build_jump_audit(
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    idx: int,
    *,
    contributor_mode: str,
) -> Optional[Dict[str, object]]:
    if nav_df.empty or idx not in nav_df.index:
        return None
    row = nav_df.loc[idx]
    week_start = row["week_start"]
    week_str = _format_week(week_start)

    nav_prev = _maybe_float(row.get("nav_prev"))
    nav = _maybe_float(row.get("nav"))
    cash_prev = _maybe_float(row.get("cash_prev"))
    cash = _maybe_float(row.get("cash_usd"))
    positions_value_prev = _maybe_float(row.get("positions_value_prev"))
    positions_value = _maybe_float(row.get("positions_value"))
    n_positions = _maybe_int(row.get("n_positions"))
    weekly_pnl = _maybe_float(row.get("weekly_pnl"))
    pct_change = _maybe_float(row.get("pct_change"))
    abs_pct_change = abs(pct_change) if pct_change is not None else None

    cash_ratio = cash / nav if nav not in (None, 0.0) and cash is not None else None

    trades_week = trades_df[trades_df["week_start"] == week_start] if not trades_df.empty else pd.DataFrame()
    trades_count = int(len(trades_week)) if not trades_week.empty else 0
    contributors = _select_top_contributors(trades_df, week_start, mode=contributor_mode, top_n=5)

    return {
        "week_start": week_str,
        "nav_prev": nav_prev,
        "cash_prev": cash_prev,
        "positions_value_prev": positions_value_prev,
        "weekly_pnl": weekly_pnl,
        "pct_change": pct_change,
        "abs_pct_change": abs_pct_change,
        "nav": nav,
        "cash_usd": cash,
        "positions_value": positions_value,
        "cash_ratio": cash_ratio,
        "n_positions": n_positions,
        "trades_count": trades_count,
        "top_contributors": contributors,
    }


def _build_max_jump_audit(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if nav_df.empty or len(nav_df) < 2 or "pct_change" not in nav_df.columns:
        return None
    pct = pd.to_numeric(nav_df["pct_change"], errors="coerce")
    pct = pct.dropna()
    if pct.empty:
        return None
    idx = pct.abs().idxmax()
    return _build_jump_audit(nav_df, trades_df, idx, contributor_mode="abs")


def _build_min_jump_audit(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if nav_df.empty or len(nav_df) < 2 or "pct_change" not in nav_df.columns:
        return None
    pct = pd.to_numeric(nav_df["pct_change"], errors="coerce")
    pct = pct.dropna()
    if pct.empty:
        return None
    idx = pct.idxmin()
    return _build_jump_audit(nav_df, trades_df, idx, contributor_mode="negative")


def _build_max_drawdown_audit(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if nav_df.empty or len(nav_df) < 2:
        return None
    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce")
    if nav_series.dropna().empty:
        return None
    running_peak = nav_series.cummax()
    drawdown = nav_series / running_peak - 1.0
    drawdown = drawdown.dropna()
    if drawdown.empty:
        return None
    trough_idx = drawdown.idxmin()
    if pd.isna(trough_idx):
        return None
    peak_slice = nav_series.loc[:trough_idx]
    if peak_slice.dropna().empty:
        return None
    peak_idx = peak_slice.idxmax()

    dd_peak_nav = _maybe_float(nav_df.loc[peak_idx, "nav"])
    dd_trough_nav = _maybe_float(nav_df.loc[trough_idx, "nav"])
    if dd_peak_nav is None or dd_trough_nav is None:
        return None

    audit = _build_jump_audit(nav_df, trades_df, int(trough_idx), contributor_mode="negative")
    if audit is None:
        return None

    try:
        dd_weeks = int(trough_idx) - int(peak_idx)
    except (TypeError, ValueError):
        dd_weeks = None

    audit.update(
        {
            "dd_peak_week": _format_week(nav_df.loc[peak_idx, "week_start"]),
            "dd_trough_week": _format_week(nav_df.loc[trough_idx, "week_start"]),
            "dd_peak_nav": dd_peak_nav,
            "dd_trough_nav": dd_trough_nav,
            "max_drawdown_pct": (dd_trough_nav / dd_peak_nav - 1.0) if dd_peak_nav else None,
            "dd_weeks": dd_weeks,
        }
    )
    return audit


def _quantile_map(series: pd.Series, quantiles: List[float]) -> Dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {}
    return {f"p{int(q * 100)}": float(series.quantile(q)) for q in quantiles}


def _histogram(series: pd.Series, bins: List[float]) -> List[Dict[str, object]]:
    series = series.dropna()
    if series.empty:
        return []
    series = series.clip(lower=0.0, upper=1.0)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    cut = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
    counts = cut.value_counts().reindex(labels, fill_value=0)
    return [{"bin": label, "count": int(counts[label])} for label in labels]


def _build_top_share_profile(
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    min_abs_weekly_pnl: float = 1.0,
    top_n: int = 5,
) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
    if nav_df.empty:
        return None, pd.DataFrame()

    pnl_by_symbol = pd.DataFrame(columns=["week_start", "symbol", "pnl"])
    if not trades_df.empty and "pnl" in trades_df.columns:
        pnl_by_symbol = trades_df.groupby(["week_start", "symbol"], as_index=False)["pnl"].sum()

    week_groups = {
        week: group["pnl"] for week, group in pnl_by_symbol.groupby("week_start")
    } if not pnl_by_symbol.empty else {}

    rows: List[Dict[str, object]] = []
    for row in nav_df.itertuples(index=False):
        week_start = getattr(row, "week_start")
        week_str = _format_week(week_start)
        weekly_pnl = _maybe_float(getattr(row, "weekly_pnl", None))
        nav = _maybe_float(getattr(row, "nav", None))
        cash = _maybe_float(getattr(row, "cash_usd", None))
        cash_ratio = cash / nav if nav not in (None, 0.0) and cash is not None else None
        n_positions = _maybe_int(getattr(row, "n_positions", None))

        week_pnls = week_groups.get(week_start)
        gross_pnl = 0.0
        top1_share_gross = None
        top5_share_gross = None
        top5_share_net = None
        if week_pnls is not None and not week_pnls.empty:
            abs_pnl = week_pnls.abs().sort_values(ascending=False)
            gross_pnl = float(abs_pnl.sum())
            if gross_pnl > 0:
                top1_share_gross = float(abs_pnl.iloc[0] / gross_pnl)
                top5_share_gross = float(abs_pnl.head(top_n).sum() / gross_pnl)
                if weekly_pnl is not None and abs(weekly_pnl) >= min_abs_weekly_pnl:
                    top5_indices = abs_pnl.head(top_n).index
                    top5_pnl_sum = float(week_pnls.loc[top5_indices].sum())
                    if weekly_pnl != 0:
                        top5_share_net = float(top5_pnl_sum / weekly_pnl)

        rows.append(
            {
                "week_start": week_str,
                "weekly_pnl": weekly_pnl,
                "gross_pnl": gross_pnl,
                "top1_share_gross": top1_share_gross,
                "top5_share_gross": top5_share_gross,
                "others_share_gross": 1.0 - top5_share_gross if top5_share_gross is not None else None,
                "top5_share_net": top5_share_net,
                "cash_ratio": cash_ratio,
                "n_positions": n_positions,
            }
        )

    by_week = pd.DataFrame(rows)
    if by_week.empty:
        return None, by_week

    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    bins = [i / 10.0 for i in range(11)]

    top1_series = by_week["top1_share_gross"]
    top5_series = by_week["top5_share_gross"]

    ranked = (
        by_week.dropna(subset=["top5_share_gross"])
        .sort_values("top5_share_gross", ascending=False)
        .head(10)
    )
    top_weeks: List[Dict[str, object]] = []
    for row in ranked.itertuples(index=False):
        top_weeks.append(
            {
                "week_start": row.week_start,
                "top5_share_gross": row.top5_share_gross,
                "gross_pnl": row.gross_pnl,
                "weekly_pnl": row.weekly_pnl,
                "cash_ratio": row.cash_ratio,
                "n_positions": row.n_positions,
            }
        )

    profile = {
        "weeks_total": int(len(by_week)),
        "weeks_with_gross_pnl": int((by_week["gross_pnl"] > 0).sum()),
        "weeks_with_net_share": int(by_week["top5_share_net"].notna().sum()),
        "min_abs_weekly_pnl_for_net_share": float(min_abs_weekly_pnl),
        "quantiles": {
            "top1_share_gross": _quantile_map(top1_series, quantiles),
            "top5_share_gross": _quantile_map(top5_series, quantiles),
        },
        "histograms": {
            "top1_share_gross": _histogram(top1_series, bins),
            "top5_share_gross": _histogram(top5_series, bins),
        },
        "top5_share_gross_top_weeks": top_weeks,
    }
    return profile, by_week


def run(cfg: AppConfig, start: Optional[date] = None, end: Optional[date] = None) -> Dict[str, object]:
    log_path = build_log_path(cfg, "backtest")
    logger = setup_logger("backtest", log_path, cfg.logging.level)
    log_event(logger, "start")

    features_df = read_parquet(snapshots_features_path(cfg))
    labels_df = read_parquet(snapshots_labels_path(cfg))
    if features_df.empty or labels_df.empty:
        raise ValueError("Snapshots features/labels are empty")

    features_df["week_start"] = _to_date(features_df["week_start"])
    labels_df["week_start"] = _to_date(labels_df["week_start"])

    exclude_symbols = {symbol.upper() for symbol in cfg.snapshots.exclude_symbols}
    if exclude_symbols:
        features_df = features_df[~features_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]
        labels_df = labels_df[~labels_df["symbol"].astype(str).str.upper().isin(exclude_symbols)]

    start = start or date.fromisoformat(cfg.backtest.start_date)
    end = end or date.fromisoformat(cfg.backtest.end_date)

    labels_df = labels_df[(labels_df["week_start"] >= start) & (labels_df["week_start"] <= end)]
    full_df = features_df.merge(labels_df, on=["week_start", "symbol"], how="inner")
    weeks = sorted(full_df["week_start"].unique().tolist())

    gate_cfg = cfg.risk.regime_gate
    gate_action = gate_cfg.action
    if gate_action not in ("no_trade", "raise_threshold"):
        log_event(logger, "unknown_regime_gate_action", value=gate_action)
        gate_action = "no_trade"

    gate_open_by_week: Dict[date, bool] = {}
    gate_closed_weeks = 0
    gate_result = None
    if gate_cfg.enabled:
        week_map_df = read_parquet(snapshots_week_map_path(cfg))
        gate_result = build_spy_regime_gate(cfg, week_map_df, features_df)
        gate_open_by_week = gate_result.open_by_week
        log_event(
            logger,
            "regime_gate_ready",
            rule=gate_cfg.rule,
            action=gate_action,
            source=gate_result.source,
            ma_days=gate_result.ma_days,
            missing_weeks=gate_result.missing_weeks,
        )

    start_cash = float(cfg.backtest.initial_cash_usd)
    if float(cfg.selection.cash_start_usd) != start_cash:
        log_event(
            logger,
            "cash_start_mismatch",
            backtest=start_cash,
            selection=float(cfg.selection.cash_start_usd),
        )
    cash = start_cash
    nav = cash
    reserve = float(cfg.selection.cash_reserve_usd)

    trades: List[Dict[str, object]] = []
    nav_rows: List[Dict[str, object]] = []
    skipped_buys = 0
    reserve_violations = 0
    cash_ratios: List[float] = []

    cost_rate = float(cfg.cost_model.bps_per_side) / 10000.0
    max_positions = max(int(cfg.selection.max_positions), 1)
    buy_fill_policy = cfg.selection.buy_fill_policy
    if buy_fill_policy not in ("ranked_partial", "ranked_strict"):
        log_event(logger, "unknown_buy_fill_policy", value=buy_fill_policy)
        buy_fill_policy = "ranked_partial"

    for week in weeks:
        gate_open = True
        if gate_cfg.enabled:
            gate_open = gate_open_by_week.get(week, True)
            if not gate_open:
                gate_closed_weeks += 1
                if gate_action == "no_trade":
                    cash_minus_reserve = cash - reserve
                    nav_rows.append(
                        {
                            "week_start": week,
                            "nav": cash,
                            "cash_usd": cash,
                            "positions_value": 0.0,
                            "reserve_usd": reserve,
                            "cash_minus_reserve": cash_minus_reserve,
                            "n_positions": 0,
                        }
                    )
                    continue

        train_weeks = select_training_weeks(
            weeks,
            week,
            train_window_years=cfg.training.train_window_years,
            min_train_weeks=cfg.training.min_train_weeks,
        )
        if not train_weeks:
            cash_minus_reserve = cash - reserve
            nav_rows.append(
                {
                    "week_start": week,
                    "nav": cash,
                    "cash_usd": cash,
                    "positions_value": 0.0,
                    "reserve_usd": reserve,
                    "cash_minus_reserve": cash_minus_reserve,
                    "n_positions": 0,
                }
            )
            continue

        train_df = full_df[full_df["week_start"].isin(train_weeks)]
        model = train_linear_model(train_df, FEATURE_COLUMNS, "label_return")
        if model is None:
            cash_minus_reserve = cash - reserve
            nav_rows.append(
                {
                    "week_start": week,
                    "nav": cash,
                    "cash_usd": cash,
                    "positions_value": 0.0,
                    "reserve_usd": reserve,
                    "cash_minus_reserve": cash_minus_reserve,
                    "n_positions": 0,
                }
            )
            continue

        week_features = full_df[full_df["week_start"] == week].copy()
        if week_features.empty:
            cash_minus_reserve = cash - reserve
            nav_rows.append(
                {
                    "week_start": week,
                    "nav": cash,
                    "cash_usd": cash,
                    "positions_value": 0.0,
                    "reserve_usd": reserve,
                    "cash_minus_reserve": cash_minus_reserve,
                    "n_positions": 0,
                }
            )
            continue

        week_features = week_features[
            week_features["avg_dollar_vol_20d"].fillna(0) >= float(cfg.snapshots.min_avg_dollar_vol_20d)
        ]
        week_features = week_features[week_features["price"].notna()]
        week_features = week_features[week_features["price"] <= float(cfg.selection.price_cap)]
        week_features = week_features.dropna(subset=list(FEATURE_COLUMNS) + ["label_return"])

        if week_features.empty:
            cash_minus_reserve = cash - reserve
            nav_rows.append(
                {
                    "week_start": week,
                    "nav": cash,
                    "cash_usd": cash,
                    "positions_value": 0.0,
                    "reserve_usd": reserve,
                    "cash_minus_reserve": cash_minus_reserve,
                    "n_positions": 0,
                }
            )
            continue

        preds = predict_linear_model(model, week_features, FEATURE_COLUMNS)
        week_features = week_features.assign(pred_return=preds)
        week_features = week_features.sort_values("pred_return", ascending=False)

        if gate_cfg.enabled and not gate_open and gate_action == "raise_threshold":
            pred_floor = float(gate_cfg.pred_return_floor)
            week_features = week_features[week_features["pred_return"] >= pred_floor]
            if week_features.empty:
                cash_minus_reserve = cash - reserve
                nav_rows.append(
                    {
                        "week_start": week,
                        "nav": cash,
                        "cash_usd": cash,
                        "positions_value": 0.0,
                        "reserve_usd": reserve,
                        "cash_minus_reserve": cash_minus_reserve,
                        "n_positions": 0,
                    }
                )
                continue

        cash_after_buys = cash
        positions_value = 0.0
        sell_costs_total = 0.0
        positions_count = 0

        for row in week_features.itertuples(index=False):
            if positions_count >= max_positions:
                break
            entry_price = float(row.price)
            realized_return = float(row.label_return)
            buy_cost = entry_price * cost_rate
            required = entry_price + buy_cost
            if cash_after_buys - reserve < required:
                skipped_buys += 1
                if buy_fill_policy == "ranked_strict":
                    break
                continue
            cash_after_buys -= required

            exit_price = entry_price * (1.0 + realized_return)
            sell_cost = exit_price * cost_rate
            proceeds = exit_price - sell_cost
            pnl = proceeds - required
            positions_value += exit_price
            sell_costs_total += sell_cost
            positions_count += 1

            trades.append(
                {
                    "week_start": week,
                    "symbol": row.symbol,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": realized_return,
                    "buy_cost": buy_cost,
                    "sell_cost": sell_cost,
                    "pnl": pnl,
                }
            )

        cash_minus_reserve = cash_after_buys - reserve
        if cash_minus_reserve < 0:
            reserve_violations += 1

        cash = cash_after_buys + positions_value - sell_costs_total
        nav = cash
        nav_row = {
            "week_start": week,
            "nav": nav,
            "cash_usd": cash_after_buys,
            "positions_value": positions_value,
            "reserve_usd": reserve,
            "cash_minus_reserve": cash_minus_reserve,
            "n_positions": positions_count,
        }
        nav_rows.append(nav_row)
        if nav > 0:
            cash_ratios.append(cash_after_buys / nav)

    backtest_dir = artifacts_backtest_dir(cfg)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(trades)
    nav_df = pd.DataFrame(nav_rows)
    if not nav_df.empty:
        nav_df = nav_df.sort_values("week_start").reset_index(drop=True)

    write_parquet_atomic(trades_df, backtest_dir / "trades.parquet")
    write_parquet_atomic(nav_df, backtest_dir / "nav.parquet")

    nav_audit_df = _with_prev_fields(nav_df) if not nav_df.empty else nav_df

    audit_max = _build_max_jump_audit(nav_audit_df, trades_df)
    if audit_max:
        write_state(audit_max, backtest_dir / "audit_max_jump.json")

    audit_min = _build_min_jump_audit(nav_audit_df, trades_df)
    if audit_min:
        write_state(audit_min, backtest_dir / "audit_min_jump.json")

    audit_dd = _build_max_drawdown_audit(nav_audit_df, trades_df)
    if audit_dd:
        write_state(audit_dd, backtest_dir / "audit_max_drawdown.json")

    profile, top_share_by_week = _build_top_share_profile(nav_audit_df, trades_df)
    if profile:
        write_state(profile, backtest_dir / "audit_top_share_profile.json")
    if not top_share_by_week.empty:
        csv_cols = [
            "week_start",
            "weekly_pnl",
            "gross_pnl",
            "top1_share_gross",
            "top5_share_gross",
            "cash_ratio",
            "n_positions",
        ]
        top_share_by_week[csv_cols].to_csv(backtest_dir / "top_share_by_week.csv", index=False)

    nav_series = nav_df["nav"] if not nav_df.empty else pd.Series(dtype=float)
    cash_series = nav_df["cash_usd"] if "cash_usd" in nav_df.columns else pd.Series(dtype=float)
    cash_minus_reserve_series = (
        nav_df["cash_minus_reserve"] if "cash_minus_reserve" in nav_df.columns else pd.Series(dtype=float)
    )

    summary = {
        "start_nav": start_cash,
        "end_nav": nav,
        "return_pct": nav / start_cash - 1.0 if start_cash else 0.0,
        "trades": int(len(trades_df)),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "min_cash_usd": float(cash_series.min()) if not cash_series.empty else None,
        "min_cash_minus_reserve": float(cash_minus_reserve_series.min()) if not cash_minus_reserve_series.empty else None,
        "reserve_violation_count": int(reserve_violations),
        "skipped_buys_insufficient_cash": int(skipped_buys),
        "avg_cash_ratio": float(sum(cash_ratios) / len(cash_ratios)) if cash_ratios else None,
    }
    if gate_cfg.enabled:
        summary.update(
            {
                "regime_gate_enabled": True,
                "regime_gate_action": gate_action,
                "regime_gate_rule": gate_cfg.rule,
                "regime_gate_source": gate_result.source if gate_result else None,
                "regime_gate_ma_days": gate_result.ma_days if gate_result else None,
                "regime_gate_closed_weeks": int(gate_closed_weeks),
            }
        )
    write_state(summary, backtest_dir / "summary.json")

    log_event(logger, "complete", trades=len(trades_df), end_nav=nav)
    return summary
