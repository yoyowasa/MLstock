from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check: gate_state=off rows should match OFF returns and turnover."
    )
    parser.add_argument(
        "--updates",
        type=Path,
        default=Path("artifacts/backtest/exposure_rolling_regime_hv13_updates.csv"),
        help="Path to exposure rolling updates CSV.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Allowed absolute difference between guard_return_pct and off_return_pct.",
    )
    parser.add_argument(
        "--turnover-tolerance",
        type=float,
        default=1e-9,
        help="Allowed absolute difference between guard_turnover_ratio and off_turnover_ratio.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        default=False,
        help="Include rows with valid_eval != True.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Maximum mismatch rows to print.",
    )
    args = parser.parse_args()

    if not args.updates.exists():
        print(f"updates_csv_not_found: {args.updates}")
        return 2

    with args.updates.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            print("empty_csv")
            return 2
        required = {
            "vol_cap_gate_state",
            "valid_eval",
            "guard_used_off",
            "guard_return_pct",
            "off_return_pct",
            "guard_turnover_ratio",
            "off_turnover_ratio",
            "update_index",
            "test_start",
            "test_end",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            print(f"missing_columns: {', '.join(missing)}")
            return 2

        gate_off_rows = 0
        checked_rows = 0
        mismatches = []

        for row in reader:
            gate_state = (row.get("vol_cap_gate_state") or "").strip().lower()
            if gate_state != "off":
                continue
            gate_off_rows += 1
            valid_eval = _parse_bool(row.get("valid_eval"))
            if not args.include_invalid and valid_eval is not True:
                continue

            checked_rows += 1
            guard_used_off = _parse_bool(row.get("guard_used_off"))
            guard_return = _parse_float(row.get("guard_return_pct"))
            off_return = _parse_float(row.get("off_return_pct"))
            return_diff = None if guard_return is None or off_return is None else guard_return - off_return
            guard_turnover = _parse_float(row.get("guard_turnover_ratio"))
            off_turnover = _parse_float(row.get("off_turnover_ratio"))
            turnover_diff = None if guard_turnover is None or off_turnover is None else guard_turnover - off_turnover

            reasons = []
            if guard_used_off is not True:
                reasons.append("guard_used_off_not_true")
            if return_diff is None:
                reasons.append("return_missing_or_invalid")
            elif abs(return_diff) > args.tolerance:
                reasons.append("return_diff_exceeds_tolerance")
            if turnover_diff is None:
                reasons.append("turnover_missing_or_invalid")
            elif abs(turnover_diff) > args.turnover_tolerance:
                reasons.append("turnover_diff_exceeds_tolerance")

            if reasons:
                mismatches.append(
                    {
                        "update_index": row.get("update_index"),
                        "test_start": row.get("test_start"),
                        "test_end": row.get("test_end"),
                        "guard_return_pct": guard_return,
                        "off_return_pct": off_return,
                        "return_diff": return_diff,
                        "guard_turnover_ratio": guard_turnover,
                        "off_turnover_ratio": off_turnover,
                        "turnover_diff": turnover_diff,
                        "guard_used_off": guard_used_off,
                        "reason": ",".join(reasons),
                    }
                )

    print(f"updates_csv: {args.updates}")
    print(f"gate_off_rows: {gate_off_rows}")
    print(f"checked_rows: {checked_rows}")
    print(f"mismatches: {len(mismatches)}")

    if mismatches:
        print("mismatch_details:")
        for item in mismatches[: args.max_mismatches]:
            print(
                " - "
                f"idx={item['update_index']} "
                f"{item['test_start']}..{item['test_end']} "
                f"guard={item['guard_return_pct']} "
                f"off={item['off_return_pct']} "
                f"diff={item['return_diff']} "
                f"turnover_guard={item['guard_turnover_ratio']} "
                f"turnover_off={item['off_turnover_ratio']} "
                f"turnover_diff={item['turnover_diff']} "
                f"guard_used_off={item['guard_used_off']} "
                f"reason={item['reason']}"
            )
        if len(mismatches) > args.max_mismatches:
            print(f"... {len(mismatches) - args.max_mismatches} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
