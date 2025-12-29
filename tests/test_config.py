from __future__ import annotations

from pathlib import Path

from mlstock.config.loader import load_config


def test_load_config_merge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    local_path = tmp_path / "config.local.yaml"

    config_path.write_text(
        """
project:
  timezone: \"America/New_York\"
  start_date: \"2016-01-01\"

alpaca:
  trading_base_url: \"https://api.alpaca.markets\"
  data_base_url: \"https://data.alpaca.markets\"

paths:
  data_dir: \"data\"
  artifacts_dir: \"artifacts\"

reference:
  assets_path: \"data/reference/assets.parquet\"
  calendar_path: \"data/reference/calendar.parquet\"
  seed_symbols_path: \"data/reference/seed_symbols.parquet\"

seed:
  n_seed: 2000

bars:
  timeframe: \"1Day\"
  feed: \"iex\"
  adjustment: \"raw\"
  asof: \"-\"
  backfill_start: \"2016-01-01\"
  lookback_days: 20
  mode: \"multi_symbol\"
  batch_size: 200
  max_workers: 2

corp_actions:
  backfill_start: \"2016-01-01\"
  lookback_days: 120
  batch_size: 200
  max_workers: 2

snapshots:
  weekly_dir: \"data/snapshots/weekly\"
  min_avg_dollar_vol_20d: 1000000
  min_trading_days: 20
  feature_lookback_days: 20
  exclude_symbols:
    - \"SPY\"

training:
  train_window_years: 4
  min_train_weeks: 52

selection:
  cash_start_usd: 1000
  cash_reserve_usd: 100
  price_cap: 60
  max_positions: 15
  buy_fill_policy: \"ranked_partial\"
  estimate_entry_buffer_bps: 100

cost_model:
  bps_per_side: 2.0

risk:
  regime_gate:
    enabled: false
    rule: \"spy_close_above_ma60\"
    action: \"no_trade\"
    spy_symbol: \"SPY\"
    ma_days: 60
    pred_return_floor: 0.0

backtest:
  start_date: \"2018-01-01\"
  end_date: \"2024-12-31\"
  initial_cash_usd: 1000

logging:
  level: \"INFO\"
  jsonl: true
""".strip()
    )

    local_path.write_text(
        """
logging:
  level: \"DEBUG\"
""".strip()
    )

    cfg = load_config(config_path=config_path, local_path=local_path)
    assert cfg.logging.level == "DEBUG"
    assert cfg.project.timezone == "America/New_York"
