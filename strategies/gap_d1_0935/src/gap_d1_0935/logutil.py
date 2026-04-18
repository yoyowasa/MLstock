from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from mlstock.logging.logger import log_event, setup_logger

from .paths import log_path


def build_strategy_logger(name: str, prefix: str, ts: datetime | None = None) -> tuple[logging.Logger, Path]:
    actual_path = log_path(prefix=prefix, ts=ts or datetime.now(timezone.utc))
    logger = setup_logger(name=name, log_path=actual_path, level="INFO")
    return logger, actual_path


__all__ = ["build_strategy_logger", "log_event"]
