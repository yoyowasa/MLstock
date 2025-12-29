from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlstock.config.loader import find_project_root
from mlstock.config.schema import AppConfig


def project_root(start_path: Optional[Path] = None) -> Path:
    return find_project_root(start_path)


def resolve_path(path_value: str, root: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = root or project_root()
    return base / path


def data_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.paths.data_dir, root)


def artifacts_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.paths.artifacts_dir, root)


def reference_assets_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.reference.assets_path, root)


def reference_calendar_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.reference.calendar_path, root)


def reference_seed_symbols_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.reference.seed_symbols_path, root)


def raw_bars_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return data_dir(cfg, root) / "raw" / "bars_daily"


def raw_bars_path(cfg: AppConfig, symbol: str, root: Optional[Path] = None) -> Path:
    return raw_bars_dir(cfg, root) / f"{symbol}.parquet"


def raw_corp_actions_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return data_dir(cfg, root) / "raw" / "corp_actions" / "corp_actions.parquet"


def snapshots_weekly_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return resolve_path(cfg.snapshots.weekly_dir, root)


def snapshots_week_map_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return snapshots_weekly_dir(cfg, root) / "week_map.parquet"


def snapshots_universe_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return snapshots_weekly_dir(cfg, root) / "universe.parquet"


def snapshots_features_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return snapshots_weekly_dir(cfg, root) / "features.parquet"


def snapshots_labels_path(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return snapshots_weekly_dir(cfg, root) / "labels.parquet"


def artifacts_orders_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return artifacts_dir(cfg, root) / "orders"


def artifacts_models_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return artifacts_dir(cfg, root) / "models"


def artifacts_backtest_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return artifacts_dir(cfg, root) / "backtest"


def artifacts_state_dir(cfg: AppConfig, root: Optional[Path] = None) -> Path:
    return artifacts_dir(cfg, root) / "state"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
