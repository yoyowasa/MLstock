from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.data.storage.state import write_state
from mlstock.jobs import weekly


def _write_config(path: Path, data_dir: Path, artifacts_dir: Path, weekly_dir: Path) -> None:
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
  n_seed: 4

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
  cash_start_usd: 0.0
  cash_reserve_usd: 0.0
  price_cap: 1000.0
  max_positions: 2
  buy_fill_policy: "ranked_partial"
  estimate_entry_buffer_bps: 0.0
  min_proba_buy: 0.0
  min_proba_keep: 0.0
  deadband_abs: 1.0
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
  initial_cash_usd: 0.0

logging:
  level: "INFO"
  jsonl: true
""".strip()
    path.write_text(content, encoding="utf-8")


def _write_snapshots(weekly_dir: Path) -> None:
    weekly_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = []
    week_train = date(2024, 1, 1)
    week_current = date(2024, 1, 8)
    symbols = [
        ("AAA", 10.0, 0.40),
        ("BBB", 30.0, 0.30),
        ("CCC", 5.0, 0.10),
        ("DDD", 12.0, 0.50),
    ]
    for week in (week_train, week_current):
        for symbol, price, signal in symbols:
            rows.append(
                {
                    "week_start": week,
                    "symbol": symbol,
                    "price": price,
                    "avg_dollar_vol_20d": 1e7,
                    "ret_1w": signal,
                    "ret_4w": 0.0,
                    "vol_4w": 0.0,
                }
            )
            labels.append({"week_start": week, "symbol": symbol, "label_return": signal})
    pd.DataFrame(rows).to_parquet(weekly_dir / "features.parquet", index=False)
    pd.DataFrame(labels).to_parquet(weekly_dir / "labels.parquet", index=False)


def test_weekly_excess_optional_keeps_sell_lower_rank_first(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_dir, artifacts_dir, weekly_dir)
    _write_snapshots(weekly_dir)

    portfolio_path = artifacts_dir / "state" / "portfolio.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(
        {
            "cash_usd": 0.0,
            "positions": {
                "AAA": 1,
                "BBB": 1,
                "CCC": 1,
            },
        },
        portfolio_path,
    )

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    selection = weekly.run(cfg)

    assert selection["sell_symbols"] == ["CCC"]
    assert "BBB" not in selection["sell_symbols"]
