from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return float(text)


def test_deadband_kpi_sell_only_week_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = {
        "as_of": "2026-01-06",
        "week_start": "2026-01-05",
        "deadband_v2_enabled": True,
        "deadband_abs": 0.0025,
        "deadband_rel": 0.0,
        "min_trade_notional": 0.0,
        "sum_abs_dw_raw": 0.12,
        "sum_abs_dw_filtered": 0.08,
        "deadband_notional_reduction": 1.0 - (0.08 / 0.12),
        "filtered_trade_fraction_notional": 1.0 - (0.08 / 0.12),
        "filtered_trade_fraction": 1.0 - (0.08 / 0.12),
        "filtered_trade_fraction_count": 0.5,
        "trade_count_raw": 2,
        "trade_count_filtered": 1,
        "turnover_ratio_std": 0.01,
        "turnover_ratio_buy": 0.01,
        "turnover_ratio_sell": 0.07,
        "turnover_ratio_total_abs": 0.08,
        "turnover_ratio_total_half": 0.04,
        "cash_after_exec": 100.0,
        "cash_start_usd": 100.0,
        "cash_est_before_buys": 100.0,
        "cash_est_after_buys": 100.0,
        "n_selected": 0,
        "kept_positions": 1,
        "held_positions": 1,
        "skipped_buys_insufficient_cash": 0,
        "data_max_features_date": "2026-01-05",
        "data_max_labels_date": "2025-12-29",
        "data_max_week_map_date": "2026-01-05",
    }
    expected_total_abs = float(payload["sum_abs_dw_filtered"])
    expected_buy = float(payload.get("turnover_ratio_buy") or payload.get("turnover_ratio_std") or 0.0)
    expected_sell = max(0.0, expected_total_abs - expected_buy)

    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    selection_path = orders_dir / "selection_20260106.json"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

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
    assert turnover_total_half is not None and math.isclose(
        turnover_total_half, 0.5 * turnover_total_abs, abs_tol=1e-12
    )
