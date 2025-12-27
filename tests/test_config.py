from __future__ import annotations

from pathlib import Path

from mlstock.config.loader import load_config


def test_load_config_merge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    local_path = tmp_path / "config.local.yaml"

    config_path.write_text(
        """
project:
  timezone: \"America/New_York\"
  start_date: \"2016-01-01\"

alpaca:
  data_base_url: \"https://data.alpaca.markets\"

paths:
  data_dir: \"data\"
  artifacts_dir: \"artifacts\"

reference:
  assets_path: \"data/reference/assets.parquet\"
  calendar_path: \"data/reference/calendar.parquet\"

logging:
  level: \"INFO\"
  jsonl: true
""".strip()
    )

    local_path.write_text(
        """
logging:
  level: \"DEBUG\"
""".strip()
    )

    cfg = load_config(config_path=config_path, local_path=local_path)
    assert cfg.logging.level == "DEBUG"
    assert cfg.project.timezone == "America/New_York"
