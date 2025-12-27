from __future__ import annotations

from pathlib import Path

from mlstock.data.storage.paths import resolve_path


def test_resolve_path_relative(tmp_path: Path) -> None:
    result = resolve_path("data/reference/assets.parquet", root=tmp_path)
    assert result == tmp_path / "data" / "reference" / "assets.parquet"


def test_resolve_path_absolute(tmp_path: Path) -> None:
    abs_path = tmp_path / "absolute.parquet"
    result = resolve_path(str(abs_path), root=tmp_path)
    assert result == abs_path
