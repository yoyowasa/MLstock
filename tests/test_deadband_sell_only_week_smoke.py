from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return float(text)


def test_deadband_kpi_sell_only_week_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    selection_src = root / "artifacts" / "orders" / "selection_20260106.json"
    if not selection_src.exists():
        pytest.skip("売りのみ週のfixture (selection_20260106.json) が見つかりません")

    payload = json.loads(selection_src.read_text(encoding="utf-8"))
    expected_total_abs = float(payload["sum_abs_dw_filtered"])
    expected_buy = float(payload.get("turnover_ratio_buy") or payload.get("turnover_ratio_std") or 0.0)
    expected_sell = max(0.0, expected_total_abs - expected_buy)

    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selection_src, orders_dir / selection_src.name)

    output_path = tmp_path / "deadband_weekly_kpi.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_deadband_kpi.py"),
            "--orders-dir",
            str(orders_dir),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=root,
        env=env,
        timeout=60,
    )

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]

    turnover_std = _parse_float(row.get("turnover_ratio_std"))
    turnover_buy = _parse_float(row.get("turnover_ratio_buy"))
    turnover_sell = _parse_float(row.get("turnover_ratio_sell"))
    turnover_total_abs = _parse_float(row.get("turnover_ratio_total_abs"))
    turnover_total_half = _parse_float(row.get("turnover_ratio_total_half"))

    assert turnover_std is not None and math.isclose(turnover_std, expected_buy, abs_tol=1e-12)
    assert turnover_buy is not None and math.isclose(turnover_buy, expected_buy, abs_tol=1e-12)
    assert turnover_total_abs is not None and turnover_total_abs > 0.0
    assert turnover_sell is not None and turnover_sell > 0.0
    assert math.isclose(turnover_total_abs, expected_total_abs, abs_tol=1e-12)
    assert math.isclose(turnover_sell, expected_sell, abs_tol=1e-12)
    assert math.isclose(turnover_total_abs, turnover_buy + turnover_sell, abs_tol=1e-12)
    assert turnover_total_half is not None and math.isclose(turnover_total_half, 0.5 * turnover_total_abs, abs_tol=1e-12)

