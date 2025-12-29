from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.state import read_state
from mlstock.jobs import backtest, weekly


def _write_config(
    path: Path,
    data_dir: Path,
    artifacts_dir: Path,
    weekly_dir: Path,
    cash_start: float,
    reserve: float,
    max_positions: int,
    price_cap: float,
    buffer_bps: float,
) -> None:
    def _p(value: Path) -> str:
        return value.as_posix()

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
  cash_start_usd: {cash_start}
  cash_reserve_usd: {reserve}
  price_cap: {price_cap}
  max_positions: {max_positions}
  buy_fill_policy: "ranked_partial"
  estimate_entry_buffer_bps: {buffer_bps}

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

backtest:
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  initial_cash_usd: {cash_start}

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
                "symbol": "EXP",
                "price": 9.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.10,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        rows.append(
            {
                "week_start": week,
                "symbol": "CHE",
                "price": 5.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.01,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        labels.append({"week_start": week, "symbol": "EXP", "label_return": 0.05})
        labels.append({"week_start": week, "symbol": "CHE", "label_return": 0.02})

    features_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels)
    features_df.to_parquet(weekly_dir / "features.parquet", index=False)
    labels_df.to_parquet(weekly_dir / "labels.parquet", index=False)


def test_backtest_reserve_constraints(tmp_path: Path) -> None:
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
        reserve=2.0,
        max_positions=2,
        price_cap=1000.0,
        buffer_bps=0.0,
    )
    _write_snapshots(weekly_dir)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    backtest.run(cfg, start=date(2024, 1, 1), end=date(2024, 1, 8))

    summary = read_state(artifacts_dir / "backtest" / "summary.json")
    assert summary["reserve_violation_count"] == 0
    assert summary["skipped_buys_insufficient_cash"] >= 1

    nav_df = pd.read_parquet(artifacts_dir / "backtest" / "nav.parquet")
    assert (nav_df["cash_minus_reserve"] >= 0).all()


def test_weekly_orders_budgeted(tmp_path: Path) -> None:
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
        reserve=2.0,
        max_positions=2,
        price_cap=1000.0,
        buffer_bps=0.0,
    )
    _write_snapshots(weekly_dir)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    weekly.run(cfg)

    orders_path = artifacts_dir / "orders"
    orders_files = sorted(
        path for path in orders_path.glob("orders_*.csv") if "orders_candidates" not in path.name
    )
    assert orders_files

    orders_df = pd.read_csv(orders_files[-1])
    buys = orders_df[orders_df["side"] == "buy"].copy()
    assert not buys.empty

    cash = 10.0
    reserve = 2.0
    for _, row in buys.sort_values("priority").iterrows():
        required = float(row["required_est"])
        cash -= required
        assert cash >= reserve
