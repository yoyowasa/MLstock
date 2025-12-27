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


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
