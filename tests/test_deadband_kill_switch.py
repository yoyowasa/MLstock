from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.state import read_state
from mlstock.jobs import weekly


def _write_config(
    path: Path,
    data_dir: Path,
    artifacts_dir: Path,
    weekly_dir: Path,
    *,
    execution_enabled: bool,
    deadband_abs: float,
    min_trade_notional: float,
) -> None:
    def _p(value: Path) -> str:
        return value.as_posix()

    enabled = "true" if execution_enabled else "false"
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
  n_seed: 2

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
  cash_start_usd: 100.0
  cash_reserve_usd: 0.0
  price_cap: 1000.0
  max_positions: 2
  buy_fill_policy: "ranked_partial"
  estimate_entry_buffer_bps: 0.0
  min_proba_buy: 0.0
  min_proba_keep: 0.0
  deadband_abs: {deadband_abs}
  deadband_rel: 0.0
  min_trade_notional: {min_trade_notional}

execution:
  deadband_v2:
    enabled: {enabled}

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
    enabled: false
    trigger: "vol_cap_enabled"
    mode: "daily"
    base_source: "off_avg_on_avg"
    base_scale: null
    cap_source: "off_p95"
    cap_value: null
    cap_buffer: 0.0
    log_scale: true

backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  initial_cash_usd: 100.0

logging:
  level: "INFO"
  jsonl: true
""".strip()
    path.write_text(content, encoding="utf-8")


def _write_snapshots(weekly_dir: Path) -> None:
    weekly_dir.mkdir(parents=True, exist_ok=True)
    weeks = [date(2024, 1, 1), date(2024, 1, 8)]
    rows = []
    labels = []
    for week in weeks:
        rows.append(
            {
                "week_start": week,
                "symbol": "AAA",
                "price": 10.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.10,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        rows.append(
            {
                "week_start": week,
                "symbol": "BBB",
                "price": 20.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.05,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        labels.append({"week_start": week, "symbol": "AAA", "label_return": 0.05})
        labels.append({"week_start": week, "symbol": "BBB", "label_return": 0.02})

    pd.DataFrame(rows).to_parquet(weekly_dir / "features.parquet", index=False)
    pd.DataFrame(labels).to_parquet(weekly_dir / "labels.parquet", index=False)


def _load_selection(artifacts_dir: Path) -> dict:
    orders_dir = artifacts_dir / "orders"
    selection_files = sorted(orders_dir.glob("selection_*.json"))
    assert selection_files
    return read_state(selection_files[-1])


def _run_weekly(config_path: Path, artifacts_dir: Path) -> dict:
    cfg = load_config(config_path=config_path, local_path=config_path.with_name("config.local.yaml"))
    weekly.run(cfg)
    return _load_selection(artifacts_dir)


def test_deadband_kill_switch_off_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    weekly_dir = data_dir / "snapshots" / "weekly"
    _write_snapshots(weekly_dir)

    artifacts_disabled = tmp_path / "artifacts_disabled"
    config_disabled = tmp_path / "config_disabled.yaml"
    _write_config(
        config_disabled,
        data_dir=data_dir,
        artifacts_dir=artifacts_disabled,
        weekly_dir=weekly_dir,
        execution_enabled=False,
        deadband_abs=0.2,
        min_trade_notional=0.1,
    )
    disabled = _run_weekly(config_disabled, artifacts_disabled)

    artifacts_baseline = tmp_path / "artifacts_baseline"
    config_baseline = tmp_path / "config_baseline.yaml"
    _write_config(
        config_baseline,
        data_dir=data_dir,
        artifacts_dir=artifacts_baseline,
        weekly_dir=weekly_dir,
        execution_enabled=True,
        deadband_abs=0.0,
        min_trade_notional=0.0,
    )
    baseline = _run_weekly(config_baseline, artifacts_baseline)

    assert disabled["deadband_v2_enabled"] is False
    assert math.isclose(
        float(disabled["sum_abs_dw_filtered"]),
        float(disabled["sum_abs_dw_raw"]),
        abs_tol=1e-12,
    )
    assert math.isclose(float(disabled["filtered_trade_fraction"]), 0.0, abs_tol=1e-12)
    assert math.isclose(
        float(disabled["cash_after_exec"]),
        float(baseline["cash_after_exec"]),
        abs_tol=1e-9,
    )
    assert math.isclose(
        float(disabled["sum_abs_dw_raw"]),
        float(baseline["sum_abs_dw_raw"]),
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(disabled["sum_abs_dw_filtered"]),
        float(baseline["sum_abs_dw_filtered"]),
        abs_tol=1e-12,
    )
