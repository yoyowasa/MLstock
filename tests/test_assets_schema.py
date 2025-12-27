from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mlstock.validate.reference import validate_assets_df


def test_assets_schema_pass() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "name": "Apple",
                "exchange": "NASDAQ",
                "asset_class": "us_equity",
                "status": "active",
                "tradable": True,
                "marginable": True,
                "shortable": True,
                "easy_to_borrow": True,
                "fractionable": True,
                "raw_json": "{}",
                "fetched_at_utc": datetime.now(timezone.utc),
            }
        ]
    )
    report = validate_assets_df(df)
    assert report["pass"]
