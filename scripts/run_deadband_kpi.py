from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlstock.config.loader import load_config
from mlstock.data.storage.paths import artifacts_dir, artifacts_orders_dir
from mlstock.data.storage.state import read_state


DEFAULT_COLUMNS = [
    "selection_file",
    "as_of",
    "week_start",
    "deadband_v2_enabled",
    "deadband_abs",
    "deadband_rel",
    "min_trade_notional",
    "sum_abs_dw_raw",
    "sum_abs_dw_filtered",
    "deadband_notional_reduction",
    "filtered_trade_fraction_notional",
    "filtered_trade_fraction",
    "filtered_trade_fraction_count",
    "trade_count_raw",
    "trade_count_filtered",
    "turnover_ratio_std",
    "turnover_ratio_buy",
    "turnover_ratio_sell",
    "turnover_ratio_total_abs",
    "turnover_ratio_total_half",
    "cash_after_exec",
    "cash_start_usd",
    "cash_est_before_buys",
    "cash_est_after_buys",
    "n_selected",
    "kept_positions",
    "held_positions",
    "skipped_buys_insufficient_cash",
    "data_max_features_date",
    "data_max_labels_date",
    "data_max_week_map_date",
    "kpi_check_status",
    "kpi_check_notes",
    "kpi_check_detail",
]


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _calc_filtered_fraction(sum_raw: object, sum_filtered: object) -> Optional[float]:
    raw = _as_float(sum_raw)
    filtered = _as_float(sum_filtered)
    if raw is None or filtered is None:
        return None
    if raw == 0:
        return None
    return 1.0 - (filtered / raw)


def _parse_date(value: object) -> Optional[datetime.date]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


REASON_DESCRIPTIONS = {
    "features_date_old": "data_max_features_date < week_start",
    "week_map_date_old": "data_max_week_map_date < week_start",
    "labels_date_old": "data_max_labels_date < week_start-7d",
    "data_max_missing": "data_max_* missing",
    "trade_count_filtered_gt_raw": "trade_count_filtered > trade_count_raw",
    "turnover_mismatch": "turnover_total_abs != buy + sell",
    "turnover_missing": "turnover ratio missing",
    "deadband_reduction_high": "deadband_notional_reduction >= 0.10",
    "deadband_reduction_missing": "deadband_notional_reduction missing",
    "week_start_missing": "week_start missing",
}


def _format_detail(codes: List[str]) -> str:
    parts = []
    for code in codes:
        desc = REASON_DESCRIPTIONS.get(code, "")
        if desc:
            parts.append(f"{code}:{desc}")
        else:
            parts.append(code)
    return ";".join(parts)


def _evaluate_kpi(row: Dict[str, Any]) -> Tuple[str, str, str]:
    warns: List[str] = []
    ngs: List[str] = []

    week_start = _parse_date(row.get("week_start"))
    if week_start is None:
        warns.append("week_start_missing")

    features_date = _parse_date(row.get("data_max_features_date"))
    week_map_date = _parse_date(row.get("data_max_week_map_date"))
    labels_date = _parse_date(row.get("data_max_labels_date"))

    if week_start and features_date and features_date < week_start:
        ngs.append("features_date_old")
    if week_start and week_map_date and week_map_date < week_start:
        ngs.append("week_map_date_old")
    if week_start and labels_date and labels_date < (week_start - timedelta(days=7)):
        warns.append("labels_date_old")
    if week_start and (features_date is None or week_map_date is None):
        warns.append("data_max_missing")

    trade_count_raw = _as_float(row.get("trade_count_raw"))
    trade_count_filtered = _as_float(row.get("trade_count_filtered"))
    if trade_count_raw is not None and trade_count_filtered is not None:
        if trade_count_filtered > trade_count_raw + 1e-9:
            ngs.append("trade_count_filtered_gt_raw")

    turnover_total_abs = _as_float(row.get("turnover_ratio_total_abs"))
    turnover_buy = _as_float(row.get("turnover_ratio_buy"))
    turnover_sell = _as_float(row.get("turnover_ratio_sell"))
    if turnover_total_abs is not None and turnover_buy is not None and turnover_sell is not None:
        if abs(turnover_total_abs - (turnover_buy + turnover_sell)) > 1e-8:
            ngs.append("turnover_mismatch")
    else:
        warns.append("turnover_missing")

    deadband_reduction = _as_float(row.get("deadband_notional_reduction"))
    if deadband_reduction is None:
        warns.append("deadband_reduction_missing")
    elif deadband_reduction >= 0.10:
        warns.append("deadband_reduction_high")

    if ngs:
        status = "NG"
        codes = ngs + warns
    elif warns:
        status = "WARN"
        codes = warns
    else:
        status = "OK"
        codes = []
    detail = _format_detail(codes)
    return status, ";".join(codes), detail


def _build_row(payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
    sum_abs_dw_raw = payload.get("sum_abs_dw_raw")
    sum_abs_dw_filtered = payload.get("sum_abs_dw_filtered")

    deadband_notional_reduction = payload.get("deadband_notional_reduction")
    if deadband_notional_reduction is None:
        deadband_notional_reduction = _calc_filtered_fraction(sum_abs_dw_raw, sum_abs_dw_filtered)

    filtered_trade_fraction_notional = payload.get("filtered_trade_fraction_notional")
    if filtered_trade_fraction_notional is None:
        filtered_trade_fraction_notional = deadband_notional_reduction
    if filtered_trade_fraction_notional is None:
        filtered_trade_fraction_notional = _calc_filtered_fraction(sum_abs_dw_raw, sum_abs_dw_filtered)

    filtered_trade_fraction = payload.get("filtered_trade_fraction")
    if filtered_trade_fraction is None:
        filtered_trade_fraction = filtered_trade_fraction_notional
    if filtered_trade_fraction is None:
        filtered_trade_fraction = _calc_filtered_fraction(sum_abs_dw_raw, sum_abs_dw_filtered)

    filtered_trade_fraction_count = payload.get("filtered_trade_fraction_count")

    turnover_ratio_buy = payload.get("turnover_ratio_buy")
    if turnover_ratio_buy is None:
        turnover_ratio_buy = payload.get("turnover_ratio_std")
    turnover_ratio_total_abs = payload.get("turnover_ratio_total_abs")
    if turnover_ratio_total_abs is None:
        turnover_ratio_total_abs = _as_float(sum_abs_dw_filtered)
    turnover_ratio_total_half = payload.get("turnover_ratio_total_half")
    if turnover_ratio_total_half is None and turnover_ratio_total_abs is not None:
        turnover_ratio_total_half = 0.5 * float(turnover_ratio_total_abs)
    turnover_ratio_sell = payload.get("turnover_ratio_sell")
    if turnover_ratio_sell is None and turnover_ratio_total_abs is not None and turnover_ratio_buy is not None:
        turnover_ratio_sell = max(0.0, float(turnover_ratio_total_abs) - float(turnover_ratio_buy))

    cash_after_exec = payload.get("cash_after_exec")
    if cash_after_exec is None:
        cash_after_exec = payload.get("cash_est_after_buys")

    cash_est_after_buys = payload.get("cash_est_after_buys")
    if cash_est_after_buys is None:
        cash_est_after_buys = cash_after_exec

    row = {
        "selection_file": path.name,
        "as_of": payload.get("as_of"),
        "week_start": payload.get("week_start"),
        "deadband_v2_enabled": payload.get("deadband_v2_enabled"),
        "deadband_abs": payload.get("deadband_abs"),
        "deadband_rel": payload.get("deadband_rel"),
        "min_trade_notional": payload.get("min_trade_notional"),
        "sum_abs_dw_raw": sum_abs_dw_raw,
        "sum_abs_dw_filtered": sum_abs_dw_filtered,
        "deadband_notional_reduction": deadband_notional_reduction,
        "filtered_trade_fraction_notional": filtered_trade_fraction_notional,
        "filtered_trade_fraction": filtered_trade_fraction,
        "filtered_trade_fraction_count": filtered_trade_fraction_count,
        "trade_count_raw": payload.get("trade_count_raw"),
        "trade_count_filtered": payload.get("trade_count_filtered"),
        "turnover_ratio_std": payload.get("turnover_ratio_std"),
        "turnover_ratio_buy": turnover_ratio_buy,
        "turnover_ratio_sell": turnover_ratio_sell,
        "turnover_ratio_total_abs": turnover_ratio_total_abs,
        "turnover_ratio_total_half": turnover_ratio_total_half,
        "cash_after_exec": cash_after_exec,
        "cash_start_usd": payload.get("cash_start_usd"),
        "cash_est_before_buys": payload.get("cash_est_before_buys"),
        "cash_est_after_buys": cash_est_after_buys,
        "n_selected": payload.get("n_selected"),
        "kept_positions": payload.get("kept_positions"),
        "held_positions": payload.get("held_positions"),
        "skipped_buys_insufficient_cash": payload.get("skipped_buys_insufficient_cash"),
        "data_max_features_date": payload.get("data_max_features_date"),
        "data_max_labels_date": payload.get("data_max_labels_date"),
        "data_max_week_map_date": payload.get("data_max_week_map_date"),
    }

    status, notes, detail = _evaluate_kpi(row)
    row["kpi_check_status"] = status
    row["kpi_check_notes"] = notes
    row["kpi_check_detail"] = detail
    return row


def _collect_selection_files(orders_dir: Path) -> List[Path]:
    return sorted(orders_dir.glob("selection_*.json"))


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--latest", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    orders_dir = Path(args.orders_dir) if args.orders_dir else artifacts_orders_dir(cfg)
    selection_files = _collect_selection_files(orders_dir)
    if args.latest:
        selection_files = selection_files[-1:] if selection_files else []
    elif args.limit is not None and args.limit > 0:
        selection_files = selection_files[-args.limit :]

    rows = []
    for selection_path in selection_files:
        payload = read_state(selection_path)
        if not payload:
            continue
        rows.append(_build_row(payload, selection_path))

    output_path = Path(args.output) if args.output else artifacts_dir(cfg) / "monitoring" / "deadband_weekly_kpi.csv"
    _write_csv(rows, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
