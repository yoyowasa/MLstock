from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from mlstock.config.schema import AppConfig


def find_project_root(start_path: Optional[Path] = None) -> Path:
    start = (start_path or Path.cwd()).resolve()
    for path in [start] + list(start.parents):
        if (path / "config.yaml").exists() or (path / "config" / "config.yaml").exists():
            return path
    raise FileNotFoundError("Could not find config.yaml in parent directories")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_config(
    config_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
) -> AppConfig:
    if config_path is not None:
        config_path = Path(config_path)
        root = config_path.parent
    else:
        root = find_project_root()
        nested_config = root / "config" / "config.yaml"
        config_path = nested_config if nested_config.exists() else root / "config.yaml"

    if local_path is not None:
        local_path = Path(local_path)
    else:
        local_path = config_path.with_name("config.local.yaml")

    load_dotenv(root / ".env", override=False)

    base_cfg = _read_yaml(config_path)
    local_cfg = _read_yaml(local_path)
    merged = _deep_merge(base_cfg, local_cfg)

    return AppConfig.from_dict(merged)
