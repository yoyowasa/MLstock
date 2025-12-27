from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mlstock.data.storage.parquet import read_parquet, write_parquet_atomic


def test_atomic_write_idempotent(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "fetched_at_utc": datetime.now(timezone.utc),
            }
        ]
    )
    path = tmp_path / "assets.parquet"
    write_parquet_atomic(df, path)
    write_parquet_atomic(df, path)

    loaded = read_parquet(path)
    assert len(loaded) == len(df)
