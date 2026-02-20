from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

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


def test_read_parquet_missing_ok_returns_empty(tmp_path) -> None:
    missing_path = tmp_path / "missing.parquet"
    loaded = read_parquet(missing_path, missing_ok=True)
    assert loaded.empty


def test_read_parquet_missing_default_raises(tmp_path) -> None:
    missing_path = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        read_parquet(missing_path)
