from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import config_path


@dataclass(frozen=True)
class UniverseConfig:
    min_close: float = 3.0
    max_close: float = 30.0
    min_avg_volume_20: float = 300000.0
    min_avg_dollar_volume_20: float = 2000000.0
    min_market_cap: float = 50000000.0
    max_market_cap: float = 10000000000.0


@dataclass(frozen=True)
class D1Config:
    min_prev_gap_pct: float = 5.0
    min_rel_vol_prev: float = 2.0
    min_close_in_range_prev: float = 0.70
    min_oc_ret_prev: float = 0.0
    lookback_days: int = 20


@dataclass(frozen=True)
class Day0Config:
    min_gap_today_pct: float = 1.0
    min_first5_range_pos: float = 0.60
    min_first5_pace: float = 1.5
    min_first5_oc_ret: float = 0.0
    min_close_vs_vwap_ratio: float = 1.0
    decision_time_et: str = '09:35:05'


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_usd: float = 50.0


@dataclass(frozen=True)
class StrategyConfig:
    universe: UniverseConfig
    d1: D1Config
    day0: Day0Config
    risk: RiskConfig


_DEF = StrategyConfig(
    universe=UniverseConfig(),
    d1=D1Config(),
    day0=Day0Config(),
    risk=RiskConfig(),
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_strategy_config(path: Path | None = None) -> StrategyConfig:
    raw = _read_yaml(path or config_path())
    universe_raw = raw.get('universe', {}) if isinstance(raw.get('universe'), dict) else {}
    d1_raw = raw.get('d1', {}) if isinstance(raw.get('d1'), dict) else {}
    day0_raw = raw.get('day0', {}) if isinstance(raw.get('day0'), dict) else {}
    risk_raw = raw.get('risk', {}) if isinstance(raw.get('risk'), dict) else {}
    return StrategyConfig(
        universe=UniverseConfig(**{**_DEF.universe.__dict__, **universe_raw}),
        d1=D1Config(**{**_DEF.d1.__dict__, **d1_raw}),
        day0=Day0Config(**{**_DEF.day0.__dict__, **day0_raw}),
        risk=RiskConfig(**{**_DEF.risk.__dict__, **risk_raw}),
    )
