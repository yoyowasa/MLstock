from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from mlstock.validate.reference import validate_calendar_df


def test_calendar_schema_pass() -> None:
    df = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 2),
                "open_time_local": "09:30",
                "close_time_local": "16:00",
                "open_ts_utc": datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc),
                "close_ts_utc": datetime(2024, 1, 2, 21, 0, tzinfo=timezone.utc),
                "fetched_at_utc": datetime.now(timezone.utc),
            }
        ]
    )
    report = validate_calendar_df(df, start_date=date(2016, 1, 1), end_date=date(2024, 1, 2))
    assert report["pass"]
