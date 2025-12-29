from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mlstock.config.loader import load_config
from mlstock.jobs import seed_symbols, weekly


def _write_config(
    path: Path,
    data_dir: Path,
    artifacts_dir: Path,
    weekly_dir: Path,
    exclude_symbols: list[str],
) -> None:
    def _p(value: Path) -> str:
        return value.as_posix()

    if exclude_symbols:
        exclude_block = "\n".join(f'    - "{symbol}"' for symbol in exclude_symbols)
        exclude_yaml = f"  exclude_symbols:\n{exclude_block}"
    else:
        exclude_yaml = "  exclude_symbols: []"

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
{exclude_yaml}

training:
  train_window_years: 1
  min_train_weeks: 1

selection:
  cash_start_usd: 1000
  cash_reserve_usd: 0
  price_cap: 1000
  max_positions: 2
  buy_fill_policy: "ranked_partial"
  estimate_entry_buffer_bps: 0

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
  initial_cash_usd: 1000

logging:
  level: "INFO"
  jsonl: true
""".strip()
    path.write_text(content, encoding="utf-8")


def _write_assets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "Alpha",
                "exchange": "NYSE",
                "tradable": True,
                "status": "active",
                "asset_class": "us_equity",
            },
            {
                "symbol": "BBB",
                "name": "Beta",
                "exchange": "NYSE",
                "tradable": True,
                "status": "active",
                "asset_class": "us_equity",
            },
        ]
    )
    assets_df.to_parquet(path, index=False)


def _write_snapshots(weekly_dir: Path) -> None:
    weekly_dir.mkdir(parents=True, exist_ok=True)
    weeks = [date(2024, 1, 1), date(2024, 1, 8)]
    rows = []
    labels = []
    for week in weeks:
        rows.append(
            {
                "week_start": week,
                "symbol": "SPY",
                "price": 100.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.01,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        rows.append(
            {
                "week_start": week,
                "symbol": "AAA",
                "price": 10.0,
                "avg_dollar_vol_20d": 1e7,
                "ret_1w": 0.02,
                "ret_4w": 0.0,
                "vol_4w": 0.0,
            }
        )
        labels.append({"week_start": week, "symbol": "SPY", "label_return": 0.01})
        labels.append({"week_start": week, "symbol": "AAA", "label_return": 0.02})

    features_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels)
    features_df.to_parquet(weekly_dir / "features.parquet", index=False)
    labels_df.to_parquet(weekly_dir / "labels.parquet", index=False)


def test_seed_includes_spy(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"

    _write_config(
        config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        weekly_dir=weekly_dir,
        exclude_symbols=[],
    )
    _write_assets(data_dir / "reference" / "assets.parquet")

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    seed_df = seed_symbols.run(cfg, n_seed=2)

    assert "SPY" in seed_df["symbol"].astype(str).tolist()
    assert len(seed_df) == 2


def test_weekly_excludes_spy(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weekly_dir = data_dir / "snapshots" / "weekly"
    config_path = tmp_path / "config.yaml"

    _write_config(
        config_path,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        weekly_dir=weekly_dir,
        exclude_symbols=["SPY"],
    )
    _write_snapshots(weekly_dir)

    cfg = load_config(config_path=config_path, local_path=tmp_path / "config.local.yaml")
    weekly.run(cfg)

    orders_dir = artifacts_dir / "orders"
    candidates_files = sorted(orders_dir.glob("orders_candidates_*.csv"))
    assert candidates_files

    candidates_df = pd.read_csv(candidates_files[-1])
    assert "SPY" not in candidates_df["symbol"].astype(str).tolist()
