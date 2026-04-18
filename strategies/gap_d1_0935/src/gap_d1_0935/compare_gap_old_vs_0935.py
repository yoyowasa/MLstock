from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .paths import compare_report_path, compare_summary_path, scans_dir


@dataclass(frozen=True)
class CompareResult:
    report_path: Path
    summary_path: Path
    rows: int


def _load_old_counts(root_logs_dir: Path) -> Dict[date, int]:
    counts: Dict[date, int] = {}
    for path in sorted(root_logs_dir.glob('gap_trade_*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get('message') != 'scanner_complete':
                continue
            trade_date = payload.get('trade_date')
            count = payload.get('count')
            if trade_date and isinstance(count, int):
                counts[date.fromisoformat(trade_date)] = count
    return counts


def compare_gap_old_vs_0935(root_logs_dir: Path) -> CompareResult:
    old_counts = _load_old_counts(root_logs_dir)
    rows: List[dict[str, object]] = []
    for path in sorted(scans_dir().glob('gap_0935_candidates_*.csv')):
        stem = path.stem.rsplit('_', 1)[-1]
        trade_date = datetime.strptime(stem, '%Y%m%d').date()
        df = pd.read_csv(path)
        new_count = int(df['pass'].fillna(False).sum()) if 'pass' in df.columns else 0
        old_count = old_counts.get(trade_date)
        rows.append(
            {
                'trade_date': trade_date.isoformat(),
                'old_gap_count': old_count,
                'new_0935_count': new_count,
                'delta': None if old_count is None else new_count - old_count,
            }
        )
    report_df = pd.DataFrame(rows).sort_values('trade_date').reset_index(drop=True) if rows else pd.DataFrame(columns=['trade_date','old_gap_count','new_0935_count','delta'])
    summary_path = compare_summary_path()
    report_path = compare_report_path(datetime.now().date())
    report_df.to_csv(summary_path, index=False)
    report_df.to_csv(report_path, index=False)
    return CompareResult(report_path=report_path, summary_path=summary_path, rows=len(report_df))
