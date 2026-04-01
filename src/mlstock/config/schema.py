from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _require(mapping: Dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Missing config key: {full_key}")
    return mapping[key]


def _get_str(mapping: Dict[str, Any], key: str, path: str) -> str:
    value = _require(mapping, key, path)
    if not isinstance(value, str):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be string: {full_key}")
    return value


def _get_bool(mapping: Dict[str, Any], key: str, path: str) -> bool:
    value = _require(mapping, key, path)
    if not isinstance(value, bool):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be bool: {full_key}")
    return value


def _get_int(mapping: Dict[str, Any], key: str, path: str) -> int:
    value = _require(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be int: {full_key}")
    return value


def _get_float(mapping: Dict[str, Any], key: str, path: str) -> float:
    value = _require(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be float: {full_key}")
    return float(value)


def _get_optional_float(mapping: Dict[str, Any], key: str, path: str) -> float | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be float or null: {full_key}")
    return float(value)


def _get_optional_str(mapping: Dict[str, Any], key: str, path: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be string or null: {full_key}")
    return value


def _get_optional_int(mapping: Dict[str, Any], key: str, path: str) -> int | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be int or null: {full_key}")
    return value


def _get_str_list(mapping: Dict[str, Any], key: str, path: str) -> List[str]:
    value = _require(mapping, key, path)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        full_key = f"{path}.{key}" if path else key
        raise ValueError(f"Config key must be list of strings: {full_key}")
    return list(value)


@dataclass(frozen=True)
class ProjectConfig:
    timezone: str
    start_date: str


@dataclass(frozen=True)
class AlpacaConfig:
    trading_base_url: str
    data_base_url: str


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    artifacts_dir: str


@dataclass(frozen=True)
class ReferenceConfig:
    assets_path: str
    calendar_path: str
    seed_symbols_path: str


@dataclass(frozen=True)
class SeedConfig:
    n_seed: int


@dataclass(frozen=True)
class BarsConfig:
    timeframe: str
    feed: str
    adjustment: str
    asof: str
    backfill_start: str
    lookback_days: int
    mode: str
    batch_size: int
    max_workers: int


@dataclass(frozen=True)
class CorpActionsConfig:
    backfill_start: str
    lookback_days: int
    batch_size: int
    max_workers: int


@dataclass(frozen=True)
class SnapshotsConfig:
    weekly_dir: str
    min_avg_dollar_vol_20d: float
    min_price: float
    min_trading_days: int
    feature_lookback_days: int
    exclude_symbols: List[str]


@dataclass(frozen=True)
class TrainingConfig:
    train_window_years: float
    min_train_weeks: int
    model_type: str
    ridge_alpha: float
    ensemble_weight_ridge: float
    lgbm_n_estimators: int | None
    lgbm_max_depth: int | None
    lgbm_learning_rate: float | None


@dataclass(frozen=True)
class WeeklyLabelsConfig:
    benchmark_symbol: str
    use_excess: bool


@dataclass(frozen=True)
class WeeklyConfig:
    labels: WeeklyLabelsConfig


@dataclass(frozen=True)
class SelectionConfig:
    cash_start_usd: float
    cash_reserve_usd: float
    price_cap: float
    max_positions: int
    min_positions: int
    confidence_sizing: bool
    confidence_threshold: float
    buy_fill_policy: str
    estimate_entry_buffer_bps: float
    min_proba_buy: float
    min_proba_keep: float
    deadband_abs: float
    deadband_rel: float
    min_trade_notional: float


@dataclass(frozen=True)
class DeadbandV2Config:
    enabled: bool


@dataclass(frozen=True)
class ExecutionConfig:
    deadband_v2: DeadbandV2Config


@dataclass(frozen=True)
class CostModelConfig:
    bps_per_side: float
    min_cost_usd_per_side: float


@dataclass(frozen=True)
class RegimeGateConfig:
    enabled: bool
    rule: str
    action: str
    spy_symbol: str
    ma_days: int
    pred_return_floor: float


@dataclass(frozen=True)
class VolCapConfig:
    enabled: bool
    rank_threshold: float
    apply_stage: str
    feature_name: str
    mode: str
    penalty_min: float


@dataclass(frozen=True)
class ExposureGuardConfig:
    enabled: bool
    trigger: str
    mode: str
    base_source: str
    base_scale: float | None
    cap_source: str
    cap_value: float | None
    cap_buffer: float
    log_scale: bool


@dataclass(frozen=True)
class RiskConfig:
    regime_gate: RegimeGateConfig
    vol_cap: VolCapConfig
    exposure_guard: ExposureGuardConfig


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    initial_cash_usd: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    jsonl: bool


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    alpaca: AlpacaConfig
    paths: PathsConfig
    reference: ReferenceConfig
    seed: SeedConfig
    bars: BarsConfig
    corp_actions: CorpActionsConfig
    snapshots: SnapshotsConfig
    training: TrainingConfig
    weekly: WeeklyConfig
    selection: SelectionConfig
    execution: ExecutionConfig
    cost_model: CostModelConfig
    risk: RiskConfig
    backtest: BacktestConfig
    logging: LoggingConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        project = _require(data, "project", "")
        alpaca = _require(data, "alpaca", "")
        paths = _require(data, "paths", "")
        reference = _require(data, "reference", "")
        seed = _require(data, "seed", "")
        bars = _require(data, "bars", "")
        corp_actions = _require(data, "corp_actions", "")
        snapshots = _require(data, "snapshots", "")
        training = _require(data, "training", "")
        weekly = data.get("weekly")
        selection = _require(data, "selection", "")
        execution = data.get("execution")
        cost_model = _require(data, "cost_model", "")
        risk = _require(data, "risk", "")
        backtest = _require(data, "backtest", "")
        logging = _require(data, "logging", "")

        project_cfg = ProjectConfig(
            timezone=_get_str(project, "timezone", "project"),
            start_date=_get_str(project, "start_date", "project"),
        )
        alpaca_cfg = AlpacaConfig(
            trading_base_url=_get_str(alpaca, "trading_base_url", "alpaca"),
            data_base_url=_get_str(alpaca, "data_base_url", "alpaca"),
        )
        paths_cfg = PathsConfig(
            data_dir=_get_str(paths, "data_dir", "paths"),
            artifacts_dir=_get_str(paths, "artifacts_dir", "paths"),
        )
        reference_cfg = ReferenceConfig(
            assets_path=_get_str(reference, "assets_path", "reference"),
            calendar_path=_get_str(reference, "calendar_path", "reference"),
            seed_symbols_path=_get_str(reference, "seed_symbols_path", "reference"),
        )
        seed_cfg = SeedConfig(
            n_seed=_get_int(seed, "n_seed", "seed"),
        )
        bars_cfg = BarsConfig(
            timeframe=_get_str(bars, "timeframe", "bars"),
            feed=_get_str(bars, "feed", "bars"),
            adjustment=_get_str(bars, "adjustment", "bars"),
            asof=_get_str(bars, "asof", "bars"),
            backfill_start=_get_str(bars, "backfill_start", "bars"),
            lookback_days=_get_int(bars, "lookback_days", "bars"),
            mode=_get_str(bars, "mode", "bars"),
            batch_size=_get_int(bars, "batch_size", "bars"),
            max_workers=_get_int(bars, "max_workers", "bars"),
        )
        corp_actions_cfg = CorpActionsConfig(
            backfill_start=_get_str(corp_actions, "backfill_start", "corp_actions"),
            lookback_days=_get_int(corp_actions, "lookback_days", "corp_actions"),
            batch_size=_get_int(corp_actions, "batch_size", "corp_actions"),
            max_workers=_get_int(corp_actions, "max_workers", "corp_actions"),
        )
        min_price = _get_optional_float(snapshots, "min_price", "snapshots")
        snapshots_cfg = SnapshotsConfig(
            weekly_dir=_get_str(snapshots, "weekly_dir", "snapshots"),
            min_avg_dollar_vol_20d=_get_float(snapshots, "min_avg_dollar_vol_20d", "snapshots"),
            min_price=1.0 if min_price is None else min_price,
            min_trading_days=_get_int(snapshots, "min_trading_days", "snapshots"),
            feature_lookback_days=_get_int(snapshots, "feature_lookback_days", "snapshots"),
            exclude_symbols=_get_str_list(snapshots, "exclude_symbols", "snapshots"),
        )
        model_type = _get_optional_str(training, "model_type", "training")
        ridge_alpha = _get_optional_float(training, "ridge_alpha", "training")
        ensemble_weight_ridge = _get_optional_float(training, "ensemble_weight_ridge", "training")
        lgbm_n_estimators = _get_optional_int(training, "lgbm_n_estimators", "training")
        lgbm_max_depth = _get_optional_int(training, "lgbm_max_depth", "training")
        lgbm_learning_rate = _get_optional_float(training, "lgbm_learning_rate", "training")
        training_cfg = TrainingConfig(
            train_window_years=_get_float(training, "train_window_years", "training"),
            min_train_weeks=_get_int(training, "min_train_weeks", "training"),
            model_type="ridge" if model_type is None else model_type,
            ridge_alpha=1.0 if ridge_alpha is None else ridge_alpha,
            ensemble_weight_ridge=0.5 if ensemble_weight_ridge is None else ensemble_weight_ridge,
            lgbm_n_estimators=lgbm_n_estimators,
            lgbm_max_depth=lgbm_max_depth,
            lgbm_learning_rate=lgbm_learning_rate,
        )
        if weekly is None:
            weekly_labels_cfg = WeeklyLabelsConfig(
                benchmark_symbol="SPY",
                use_excess=True,
            )
        else:
            if not isinstance(weekly, dict):
                raise ValueError("Config key must be object: weekly")
            weekly_labels = weekly.get("labels")
            if weekly_labels is None:
                weekly_labels_cfg = WeeklyLabelsConfig(
                    benchmark_symbol="SPY",
                    use_excess=True,
                )
            else:
                if not isinstance(weekly_labels, dict):
                    raise ValueError("Config key must be object: weekly.labels")
                benchmark_symbol = _get_optional_str(weekly_labels, "benchmark_symbol", "weekly.labels")
                if "use_excess" in weekly_labels:
                    use_excess = _get_bool(weekly_labels, "use_excess", "weekly.labels")
                else:
                    use_excess = True
                weekly_labels_cfg = WeeklyLabelsConfig(
                    benchmark_symbol="SPY" if benchmark_symbol is None else benchmark_symbol,
                    use_excess=use_excess,
                )
        weekly_cfg = WeeklyConfig(labels=weekly_labels_cfg)
        deadband_abs = _get_optional_float(selection, "deadband_abs", "selection")
        deadband_rel = _get_optional_float(selection, "deadband_rel", "selection")
        min_trade_notional = _get_optional_float(selection, "min_trade_notional", "selection")
        min_positions = _get_optional_int(selection, "min_positions", "selection")
        confidence_sizing_raw = selection.get("confidence_sizing")
        confidence_sizing = bool(confidence_sizing_raw) if confidence_sizing_raw is not None else False
        confidence_threshold = _get_optional_float(selection, "confidence_threshold", "selection")
        selection_cfg = SelectionConfig(
            cash_start_usd=_get_float(selection, "cash_start_usd", "selection"),
            cash_reserve_usd=_get_float(selection, "cash_reserve_usd", "selection"),
            price_cap=_get_float(selection, "price_cap", "selection"),
            max_positions=_get_int(selection, "max_positions", "selection"),
            min_positions=5 if min_positions is None else min_positions,
            confidence_sizing=confidence_sizing,
            confidence_threshold=0.0 if confidence_threshold is None else confidence_threshold,
            buy_fill_policy=_get_str(selection, "buy_fill_policy", "selection"),
            estimate_entry_buffer_bps=_get_float(selection, "estimate_entry_buffer_bps", "selection"),
            min_proba_buy=_get_float(selection, "min_proba_buy", "selection"),
            min_proba_keep=_get_float(selection, "min_proba_keep", "selection"),
            deadband_abs=0.0 if deadband_abs is None else deadband_abs,
            deadband_rel=0.0 if deadband_rel is None else deadband_rel,
            min_trade_notional=0.0 if min_trade_notional is None else min_trade_notional,
        )
        if execution is None:
            deadband_v2_cfg = DeadbandV2Config(enabled=True)
        else:
            deadband_v2 = _require(execution, "deadband_v2", "execution")
            deadband_v2_cfg = DeadbandV2Config(
                enabled=_get_bool(deadband_v2, "enabled", "execution.deadband_v2"),
            )
        execution_cfg = ExecutionConfig(deadband_v2=deadband_v2_cfg)
        min_cost_usd_per_side = _get_optional_float(cost_model, "min_cost_usd_per_side", "cost_model")
        cost_model_cfg = CostModelConfig(
            bps_per_side=_get_float(cost_model, "bps_per_side", "cost_model"),
            min_cost_usd_per_side=0.0 if min_cost_usd_per_side is None else min_cost_usd_per_side,
        )
        regime_gate = _require(risk, "regime_gate", "risk")
        vol_cap = _require(risk, "vol_cap", "risk")
        exposure_guard = _require(risk, "exposure_guard", "risk")
        regime_gate_cfg = RegimeGateConfig(
            enabled=_get_bool(regime_gate, "enabled", "risk.regime_gate"),
            rule=_get_str(regime_gate, "rule", "risk.regime_gate"),
            action=_get_str(regime_gate, "action", "risk.regime_gate"),
            spy_symbol=_get_str(regime_gate, "spy_symbol", "risk.regime_gate"),
            ma_days=_get_int(regime_gate, "ma_days", "risk.regime_gate"),
            pred_return_floor=_get_float(regime_gate, "pred_return_floor", "risk.regime_gate"),
        )
        vol_cap_cfg = VolCapConfig(
            enabled=_get_bool(vol_cap, "enabled", "risk.vol_cap"),
            rank_threshold=_get_float(vol_cap, "rank_threshold", "risk.vol_cap"),
            apply_stage=_get_str(vol_cap, "apply_stage", "risk.vol_cap"),
            feature_name=_get_str(vol_cap, "feature_name", "risk.vol_cap"),
            mode=_get_str(vol_cap, "mode", "risk.vol_cap"),
            penalty_min=_get_float(vol_cap, "penalty_min", "risk.vol_cap"),
        )
        exposure_guard_cfg = ExposureGuardConfig(
            enabled=_get_bool(exposure_guard, "enabled", "risk.exposure_guard"),
            trigger=_get_str(exposure_guard, "trigger", "risk.exposure_guard"),
            mode=_get_str(exposure_guard, "mode", "risk.exposure_guard"),
            base_source=_get_str(exposure_guard, "base_source", "risk.exposure_guard"),
            base_scale=_get_optional_float(exposure_guard, "base_scale", "risk.exposure_guard"),
            cap_source=_get_str(exposure_guard, "cap_source", "risk.exposure_guard"),
            cap_value=_get_optional_float(exposure_guard, "cap_value", "risk.exposure_guard"),
            cap_buffer=_get_float(exposure_guard, "cap_buffer", "risk.exposure_guard"),
            log_scale=_get_bool(exposure_guard, "log_scale", "risk.exposure_guard"),
        )
        risk_cfg = RiskConfig(
            regime_gate=regime_gate_cfg,
            vol_cap=vol_cap_cfg,
            exposure_guard=exposure_guard_cfg,
        )
        backtest_cfg = BacktestConfig(
            start_date=_get_str(backtest, "start_date", "backtest"),
            end_date=_get_str(backtest, "end_date", "backtest"),
            initial_cash_usd=_get_float(backtest, "initial_cash_usd", "backtest"),
        )
        logging_cfg = LoggingConfig(
            level=_get_str(logging, "level", "logging"),
            jsonl=_get_bool(logging, "jsonl", "logging"),
        )

        return AppConfig(
            project=project_cfg,
            alpaca=alpaca_cfg,
            paths=paths_cfg,
            reference=reference_cfg,
            seed=seed_cfg,
            bars=bars_cfg,
            corp_actions=corp_actions_cfg,
            snapshots=snapshots_cfg,
            training=training_cfg,
            weekly=weekly_cfg,
            selection=selection_cfg,
            execution=execution_cfg,
            cost_model=cost_model_cfg,
            risk=risk_cfg,
            backtest=backtest_cfg,
            logging=logging_cfg,
        )
