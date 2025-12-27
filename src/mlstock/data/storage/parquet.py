from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
