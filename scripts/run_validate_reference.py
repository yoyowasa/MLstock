from __future__ import annotations

import json
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_dir, reference_assets_path, reference_calendar_path
from mlstock.validate.reference import (
    build_reference_report,
    validate_assets_file,
    validate_calendar_file,
)


def main() -> None:
    cfg = load_config()
    start_date = date.fromisoformat(cfg.project.start_date)
    tz = ZoneInfo(cfg.project.timezone)
    end_date = datetime.now(tz).date()

    assets_path = reference_assets_path(cfg)
    calendar_path = reference_calendar_path(cfg)

    assets_report = validate_assets_file(assets_path)
    calendar_report = validate_calendar_file(calendar_path, start_date=start_date, end_date=end_date)
    report = build_reference_report(assets_report, calendar_report)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    validate_dir = artifacts_dir(cfg) / "validate"
    validate_dir.mkdir(parents=True, exist_ok=True)
    report_path = validate_dir / f"reference_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)

    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
