from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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


@dataclass(frozen=True)
class ProjectConfig:
    timezone: str
    start_date: str


@dataclass(frozen=True)
class AlpacaConfig:
    data_base_url: str


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    artifacts_dir: str


@dataclass(frozen=True)
class ReferenceConfig:
    assets_path: str
    calendar_path: str


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
    logging: LoggingConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        project = _require(data, "project", "")
        alpaca = _require(data, "alpaca", "")
        paths = _require(data, "paths", "")
        reference = _require(data, "reference", "")
        logging = _require(data, "logging", "")

        project_cfg = ProjectConfig(
            timezone=_get_str(project, "timezone", "project"),
            start_date=_get_str(project, "start_date", "project"),
        )
        alpaca_cfg = AlpacaConfig(
            data_base_url=_get_str(alpaca, "data_base_url", "alpaca"),
        )
        paths_cfg = PathsConfig(
            data_dir=_get_str(paths, "data_dir", "paths"),
            artifacts_dir=_get_str(paths, "artifacts_dir", "paths"),
        )
        reference_cfg = ReferenceConfig(
            assets_path=_get_str(reference, "assets_path", "reference"),
            calendar_path=_get_str(reference, "calendar_path", "reference"),
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
            logging=logging_cfg,
        )
