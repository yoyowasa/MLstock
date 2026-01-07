from __future__ import annotations

import json
import os
import subprocess
import sys
import math
from pathlib import Path
from typing import Tuple

import pytest

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import (
    snapshots_features_path,
    snapshots_labels_path,
    snapshots_week_map_path,
)


def _has_required_snapshots(cfg) -> Tuple[bool, str]:
    features_path = snapshots_features_path(cfg)
    labels_path = snapshots_labels_path(cfg)
    if not features_path.exists() or not labels_path.exists():
        return False, "snapshots features/labels are missing"
    if cfg.risk.regime_gate.enabled:
        week_map_path = snapshots_week_map_path(cfg)
        if not week_map_path.exists():
            return False, "snapshots week_map is missing"
    return True, ""


def test_deadband_golden_metrics(tmp_path: Path) -> None:
    cfg = load_config()
    ok, reason = _has_required_snapshots(cfg)
    if not ok:
        pytest.skip(reason)

    root = Path(__file__).resolve().parents[1]
    output_prefix = tmp_path / "deadband_golden"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    base_cmd = [
        sys.executable,
        str(root / "scripts" / "run_execution_rolling.py"),
        "--deadband-abs",
        "0.0025",
        "--deadband-rel",
        "0",
        "--min-trade-notional",
        "0",
        "--bps",
        "0,1,5,10,20",
    ]
    cmd = base_cmd + ["--deadband-enabled", "true"]
    subprocess.run(
        cmd + ["--output-prefix", str(output_prefix)],
        check=True,
        cwd=root,
        env=env,
        timeout=60 * 20,
    )

    summary_path = output_prefix.with_suffix(".json")
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    valid = data["rolling"]["valid"]

    diff0 = valid["diff_raw_minus_off"]["return_pct"]
    diff5 = valid["cost_diff_vs_off_bps"]["5.0"]["raw_minus_off"]["return_pct"]
    diff20 = valid["cost_diff_vs_off_bps"]["20.0"]["raw_minus_off"]["return_pct"]
    dd0 = valid["diff_raw_minus_off"]["max_drawdown_pct"]
    dd5 = valid["cost_diff_vs_off_bps"]["5.0"]["raw_minus_off"]["max_drawdown_pct"]
    off_turn = valid["off"]["turnover_ratio_std"]
    raw_turn = valid["on_raw"]["turnover_ratio_std"]

    assert diff0 is not None and diff0 >= 0.001
    assert diff5 is not None and diff5 >= 0.0
    assert diff20 is not None and diff20 >= 0.0
    assert dd0 is not None and dd0 >= 0.0
    assert dd5 is not None and dd5 >= 0.0
    assert (
        off_turn is not None
        and raw_turn is not None
        and raw_turn <= off_turn
    )

    output_prefix_off = tmp_path / "deadband_golden_off"
    subprocess.run(
        base_cmd
        + ["--deadband-enabled", "false", "--output-prefix", str(output_prefix_off)],
        check=True,
        cwd=root,
        env=env,
        timeout=60 * 20,
    )

    summary_off_path = output_prefix_off.with_suffix(".json")
    assert summary_off_path.exists()
    data_off = json.loads(summary_off_path.read_text())
    valid_off = data_off["rolling"]["valid"]

    diff0_off = valid_off["diff_raw_minus_off"]["return_pct"]
    diff5_off = valid_off["cost_diff_vs_off_bps"]["5.0"]["raw_minus_off"]["return_pct"]
    diff20_off = valid_off["cost_diff_vs_off_bps"]["20.0"]["raw_minus_off"]["return_pct"]
    dd0_off = valid_off["diff_raw_minus_off"]["max_drawdown_pct"]
    off_turn_off = valid_off["off"]["turnover_ratio_std"]
    raw_turn_off = valid_off["on_raw"]["turnover_ratio_std"]

    assert diff0_off is not None and math.isclose(diff0_off, 0.0, abs_tol=1e-6)
    assert diff5_off is not None and math.isclose(diff5_off, 0.0, abs_tol=1e-6)
    assert diff20_off is not None and math.isclose(diff20_off, 0.0, abs_tol=1e-6)
    assert dd0_off is not None and math.isclose(dd0_off, 0.0, abs_tol=1e-6)
    assert (
        off_turn_off is not None
        and raw_turn_off is not None
        and math.isclose(raw_turn_off, off_turn_off, abs_tol=1e-9)
    )
