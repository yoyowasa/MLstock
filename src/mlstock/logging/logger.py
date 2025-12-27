from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mlstock.config.schema import AppConfig
from mlstock.data.storage.paths import artifacts_dir


_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {key: value for key, value in record.__dict__.items() if key not in _LOG_RECORD_KEYS}
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=True, default=str)


def build_log_path(
    cfg: AppConfig,
    job_name: str,
    ts: Optional[datetime] = None,
    root: Optional[Path] = None,
) -> Path:
    timestamp = ts or datetime.now(timezone.utc)
    log_dir = artifacts_dir(cfg, root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_name}_{timestamp:%Y%m%d_%H%M%S}.jsonl"
    return log_dir / filename


def setup_logger(name: str, log_path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonlFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"fields": fields} if fields else None)
