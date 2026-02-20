from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.jobs import backtest


def _write_config(
    path: Path,
    data_dir: Path,
    artifacts_dir: Path,
    weekly_dir: Path,
    *,
    cash_start: float,
    reserve: float,
    buffer_bps: float,
    guard_enabled: bool,
) -> None:
    def _p(value: Path) -> str:
        return value.as_posix()

    enabled = "true" if guard_enabled else "false"
    content = f"""
project:
  timezone: "America/New_York"
  start_date: "2016-01-01"

alpaca:
  trading_base_url: "https://paper-api.alpaca.markets"
  data_base_url: "https://data.alpaca.markets"

paths:
  data_dir: "{_p(data_dir)}"
  artifacts_dir: "{_p(artifacts_dir)}"

reference:
  assets_path: "{_p(data_dir / "reference" / "assets.parquet")}"
  calendar_path: "{_p(data_dir / "reference" / "calendar.parquet")}"
  seed_symbols_path: "{_p(data_dir / "reference" / "seed_symbols.parquet")}"

seed:
  n_seed: 1

bars:
  timeframe: "1Day"
  feed: "iex"
  adjustment: "raw"
  asof: "-"
  backfill_start: "2016-01-01"
  lookback_days: 20
  mode: "multi_symbol"
  batch_size: 200
  max_workers: 1

corp_actions:
  backfill_start: "2016-01-01"
  lookback_days: 120
  batch_size: 200
  max_workers: 1

snapshots:
  weekly_dir: "{_p(weekly_dir)}"
  min_avg_dollar_vol_20d: 0
  min_trading_days: 1
  feature_lookback_days: 1
  exclude_symbols: []

training:
  train_window_years: 1
  min_train_weeks: 1

selection:
  cash_start_usd: {cash_start}
  cash_reserve_usd: {reserve}
  price_cap: 1000.0
  max_positions: 1
  buy_fill_policy: "ranked_partial"
  estimate_entry_buffer_bps: {buffer_bps}
  min_proba_buy: 0.0
  min_proba_keep: 0.0
  deadband_abs: 0.0
  deadband_rel: 0.0
  min_trade_notional: 0.0

execution:
  deadband_v2:
    enabled: true

cost_model:
  bps_per_side: 0.0

risk:
  regime_gate:
    enabled: false
    rule: "spy_close_above_ma60"
    action: "no_trade"
    spy_symbol: "SPY"
    ma_days: 60
    pred_return_floor: 0.0
  vol_cap:
    enabled: false
    rank_threshold: 0.8
    apply_stage: "selection"
    feature_name: "vol_4w"
    mode: "hard"
    penalty_min: 0.5
  exposure_guard:
    enabled: {enabled}
    trigger: "always"
    mode: "daily"
    base_source: "fixed"
    base_scale: 0.5
    cap_source: "none"
    cap_value: null
    cap_buffer: 0.0
    log_scale: true

backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  initial_cash_usd: {cash_start}

logging:
  level: "INFO"
  jsonl: true
""".strip()
    path.write_text(content, encoding="utf-8")


def _write_snapshots(weekly_dir: Path, *, price: float, label_return: float) -> None:
    weekly_dir.mkdir(parents=True, exist_ok=True)
    weeks = [date(2024, 1, 1), date(2024, 1, 8)]
    features_rows = []
    labels_rows = []
    for week in weeks:
        features_rows.append(
            {
                "week_start": week,
                "symbol": "AAA",
                "price": price,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.1,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        labels_rows.append({"week_start": week, "symbol": "AAA", "label_return": label_return})
    pd.DataFrame(features_rows).to_parquet(weekly_dir / "features.parquet", index=False)
    pd.DataFrame(labels_rows).to_parquet(weekly_dir / "labels.parquet", index=False)


def test_backtest_guard_flags_defined_when_no_positions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"

    _write_config(
        config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        weekly_dir=weekly_dir,
        cash_start=10.0,
        reserve=9.0,
        buffer_bps=0.0,
        guard_enabled=True,
    )
    _write_snapshots(weekly_dir, price=9.0, label_return=0.0)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    backtest.run(cfg, start=date(2024, 1, 1), end=date(2024, 1, 8))

    nav_df = pd.read_parquet(artifacts_dir / "backtest" / "nav.parquet")
    last_row = nav_df.sort_values("week_start").iloc[-1]
    assert bool(last_row["exposure_guard_base_applied"]) is False
    assert bool(last_row["exposure_guard_cap_applied"]) is False


def test_backtest_buffer_is_budget_only_not_realized_pnl(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"

    _write_config(
        config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        weekly_dir=weekly_dir,
        cash_start=100.0,
        reserve=0.0,
        buffer_bps=1000.0,
        guard_enabled=False,
    )
    _write_snapshots(weekly_dir, price=10.0, label_return=0.0)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    backtest.run(cfg, start=date(2024, 1, 1), end=date(2024, 1, 8))

    trades_df = pd.read_parquet(artifacts_dir / "backtest" / "trades.parquet")
    assert len(trades_df) == 1
    assert float(trades_df.iloc[0]["pnl"]) == 0.0

    nav_df = pd.read_parquet(artifacts_dir / "backtest" / "nav.parquet")
    assert float(nav_df.sort_values("week_start").iloc[-1]["nav"]) == 100.0


def test_backtest_uses_pre_start_history_for_training(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    _write_config(
        config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        weekly_dir=weekly_dir,
        cash_start=100.0,
        reserve=0.0,
        buffer_bps=0.0,
        guard_enabled=False,
    )

    features_df = pd.DataFrame(
        [
            {
                "week_start": date(2023, 12, 25),
                "symbol": "AAA",
                "price": 10.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.20,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            },
            {
                "week_start": date(2024, 1, 1),
                "symbol": "AAA",
                "price": 10.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.30,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            },
        ]
    )
    labels_df = pd.DataFrame(
        [
            {"week_start": date(2023, 12, 25), "symbol": "AAA", "label_return": 0.01},
            {"week_start": date(2024, 1, 1), "symbol": "AAA", "label_return": 0.05},
        ]
    )
    features_df.to_parquet(weekly_dir / "features.parquet", index=False)
    labels_df.to_parquet(weekly_dir / "labels.parquet", index=False)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    backtest.run(cfg, start=date(2024, 1, 1), end=date(2024, 1, 1))

    trades_df = pd.read_parquet(artifacts_dir / "backtest" / "trades.parquet")
    assert len(trades_df) == 1
