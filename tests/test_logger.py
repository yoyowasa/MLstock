from __future__ import annotations

import json
from pathlib import Path

from mlstock.logging.logger import log_event, setup_logger


def test_log_event_writes_flat_extra_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = setup_logger("test_logger_flat_fields", log_path, "INFO")

    log_event(logger, "complete", trades=3, end_nav=123.45)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    assert payload["message"] == "complete"
    assert payload["trades"] == 3
    assert payload["end_nav"] == 123.45
    assert "fields" not in payload
