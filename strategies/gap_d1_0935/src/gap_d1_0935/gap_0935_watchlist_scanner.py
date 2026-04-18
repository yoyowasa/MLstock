from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import ET, fetch_bars_batch, infer_trade_date, load_alpaca_client, to_local_ts
from .config import StrategyConfig, load_strategy_config
from .logutil import build_strategy_logger, log_event
from .paths import scan_path, watchlist_path


@dataclass(frozen=True)
class Gap0935ScanResult:
    trade_date: date
    csv_path: Path
    log_path: Path
    final_candidate_count: int


def _aggregate_first5(symbol: str, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _inspect_first5_window(symbol, items).get('aggregate')


def _inspect_first5_window(symbol: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    in_window_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for item in items:
        ts = to_local_ts(item.get('t'))
        if ts.hour != 9 or ts.minute < 30 or ts.minute >= 35:
            continue
        raw_open = item.get('o')
        raw_high = item.get('h')
        raw_low = item.get('l')
        raw_close = item.get('c')
        raw_volume = item.get('v')
        row = {
            'ts': ts,
            'raw_open': raw_open,
            'raw_high': raw_high,
            'raw_low': raw_low,
            'raw_close': raw_close,
            'raw_volume': raw_volume,
        }
        in_window_rows.append(row)
        try:
            valid_rows.append(
                {
                    'ts': ts,
                    'open': float(raw_open),
                    'high': float(raw_high),
                    'low': float(raw_low),
                    'close': float(raw_close),
                    'volume': float(raw_volume),
                }
            )
        except (TypeError, ValueError):
            continue
    in_window_rows = sorted(in_window_rows, key=lambda row: row['ts'])
    valid_rows = sorted(valid_rows, key=lambda row: row['ts'])
    open_exists = len(valid_rows) > 0
    minute_bars_in_0930_0935 = len(valid_rows)
    first5_constructible = len(valid_rows) == 5
    remarks: List[str] = []
    if in_window_rows and valid_rows:
        if valid_rows[0]['ts'].minute > 30:
            remarks.append('first_valid_bar_after_0930')
        if len(valid_rows) < len(in_window_rows):
            remarks.append('invalid_rows_present')
    if len(valid_rows) > 1:
        expected = list(range(valid_rows[0]['ts'].minute, valid_rows[0]['ts'].minute + len(valid_rows)))
        actual = [row['ts'].minute for row in valid_rows]
        if actual != expected:
            remarks.append('minute_gap_detected')
    missing_reason = ''
    if len(in_window_rows) == 0:
        missing_reason = 'no_minute_bars'
    elif len(valid_rows) == 0:
        missing_reason = 'unknown'
    elif len(valid_rows) < 5:
        if valid_rows[0]['ts'].minute > 30:
            missing_reason = 'late_open_or_halt'
        else:
            missing_reason = 'partial_minute_bars'
    elif '.' in symbol or '/' in symbol or '-' in symbol:
        missing_reason = 'symbol_issue'
    aggregate = None
    if valid_rows:
        volume = sum(row['volume'] for row in valid_rows)
        pv = sum(row['close'] * row['volume'] for row in valid_rows)
        aggregate = {
            'symbol': symbol,
            'open_D': valid_rows[0]['open'],
            'first5_open': valid_rows[0]['open'],
            'first5_high': max(row['high'] for row in valid_rows),
            'first5_low': min(row['low'] for row in valid_rows),
            'first5_close': valid_rows[-1]['close'],
            'first5_volume': volume,
            'vwap': pv / volume if volume > 0 else valid_rows[-1]['close'],
            'bars_in_first5': len(valid_rows),
        }
    return {
        'symbol': symbol,
        'open_exists': open_exists,
        'minute_bars_in_0930_0935': minute_bars_in_0930_0935,
        'first5_constructible': first5_constructible,
        'missing_reason': missing_reason or ('ok' if first5_constructible else 'unknown'),
        'remarks': '|'.join(remarks),
        'aggregate': aggregate,
    }


def scan_gap_0935_watchlist(
    trade_date: Optional[date] = None,
    config: Optional[StrategyConfig] = None,
) -> Gap0935ScanResult:
    cfg = config or load_strategy_config()
    root_cfg, client = load_alpaca_client()
    actual_trade_date = infer_trade_date(client, trade_date)
    watchlist_csv = watchlist_path(actual_trade_date)
    if not watchlist_csv.exists():
        raise FileNotFoundError(f'Watchlist not found: {watchlist_csv}')
    watchlist_df = pd.read_csv(watchlist_csv)
    logger, log_path = build_strategy_logger('gap_0935_scan', 'gap_0935_scan')
    if watchlist_df.empty:
        csv_path = scan_path(actual_trade_date)
        pd.DataFrame(
            columns=[
                'symbol',
                'trade_date',
                'open_D',
                'close_D-1',
                'gap_today_pct',
                'first5_open',
                'first5_high',
                'first5_low',
                'first5_close',
                'first5_volume',
                'first5_pace',
                'first5_range_pos',
                'first5_oc_ret',
                'vwap',
                'pass',
                'fail_reason',
                'open_exists',
                'minute_bars_in_0930_0935',
                'first5_constructible',
                'missing_reason',
                'remarks',
            ]
        ).to_csv(csv_path, index=False)
        log_event(
            logger,
            'gap_0935_scan_complete',
            trade_date=actual_trade_date.isoformat(),
            watchlist_count=0,
            open_ok_count=0,
            first5_ok_count=0,
            gap_pass_count=0,
            range_pass_count=0,
            pace_pass_count=0,
            vwap_pass_count=0,
            final_candidate_count=0,
            output_path=str(csv_path),
        )
        return Gap0935ScanResult(
            trade_date=actual_trade_date,
            csv_path=csv_path,
            log_path=log_path,
            final_candidate_count=0,
        )

    watchlist_df['symbol'] = watchlist_df['symbol'].astype(str).str.upper()
    symbols = watchlist_df['symbol'].dropna().astype(str).tolist()
    start_local = datetime.combine(actual_trade_date, dtime(9, 30), tzinfo=ET)
    end_local = datetime.combine(actual_trade_date, dtime(9, 35, 5), tzinfo=ET)
    bars_by_symbol = fetch_bars_batch(
        client=client,
        symbols=symbols,
        start=start_local,
        end=end_local,
        timeframe='1Min',
        feed=root_cfg.bars.feed,
        adjustment=root_cfg.bars.adjustment,
        asof=root_cfg.bars.asof,
    )

    rows: List[Dict[str, Any]] = []
    open_ok_count = 0
    first5_ok_count = 0
    gap_pass_count = 0
    range_pass_count = 0
    pace_pass_count = 0
    vwap_pass_count = 0

    for rec in watchlist_df.to_dict(orient='records'):
        symbol = str(rec['symbol']).upper()
        inspection = _inspect_first5_window(symbol, bars_by_symbol.get(symbol, []))
        agg = inspection['aggregate']
        fail_reasons: List[str] = []
        if agg is None:
            fail_reasons.append('missing_first5')
            rows.append(
                {
                    'symbol': symbol,
                    'trade_date': actual_trade_date.isoformat(),
                    'open_D': None,
                    'close_D-1': rec.get('close_D-1'),
                    'gap_today_pct': None,
                    'first5_open': None,
                    'first5_high': None,
                    'first5_low': None,
                    'first5_close': None,
                    'first5_volume': None,
                    'first5_pace': None,
                    'first5_range_pos': None,
                    'first5_oc_ret': None,
                    'vwap': None,
                    'pass': False,
                    'fail_reason': '|'.join(fail_reasons),
                    'open_exists': inspection['open_exists'],
                    'minute_bars_in_0930_0935': inspection['minute_bars_in_0930_0935'],
                    'first5_constructible': inspection['first5_constructible'],
                    'missing_reason': inspection['missing_reason'],
                    'remarks': inspection['remarks'],
                }
            )
            continue

        open_ok_count += 1
        first5_ok_count += 1
        close_d1 = float(rec['close_D-1'])
        avg_volume_20 = float(rec['avg_volume_20'])
        gap_today_pct = (agg['open_D'] - close_d1) / close_d1 * 100.0 if close_d1 > 0 else 0.0
        first5_range = agg['first5_high'] - agg['first5_low']
        first5_range_pos = (agg['first5_close'] - agg['first5_low']) / first5_range if first5_range > 0 else 0.0
        first5_oc_ret = (agg['first5_close'] - agg['first5_open']) / agg['first5_open'] * 100.0 if agg['first5_open'] > 0 else 0.0
        first5_pace = (agg['first5_volume'] * 78.0) / avg_volume_20 if avg_volume_20 > 0 else 0.0
        gap_ok = agg['open_D'] > close_d1 and gap_today_pct >= cfg.day0.min_gap_today_pct
        range_ok = first5_range_pos >= cfg.day0.min_first5_range_pos and first5_oc_ret >= cfg.day0.min_first5_oc_ret
        pace_ok = first5_pace >= cfg.day0.min_first5_pace
        vwap_ok = agg['first5_close'] >= agg['vwap'] * cfg.day0.min_close_vs_vwap_ratio

        if gap_ok:
            gap_pass_count += 1
        else:
            fail_reasons.append('gap_fail')
        if range_ok:
            range_pass_count += 1
        else:
            if first5_range_pos < cfg.day0.min_first5_range_pos:
                fail_reasons.append('range_fail')
            if first5_oc_ret <= cfg.day0.min_first5_oc_ret:
                fail_reasons.append('oc_ret_fail')
        if pace_ok:
            pace_pass_count += 1
        else:
            fail_reasons.append('pace_fail')
        if vwap_ok:
            vwap_pass_count += 1
        else:
            fail_reasons.append('vwap_fail')

        passed = gap_ok and range_ok and pace_ok and vwap_ok
        rows.append(
            {
                'symbol': symbol,
                'trade_date': actual_trade_date.isoformat(),
                'open_D': agg['open_D'],
                'close_D-1': close_d1,
                'gap_today_pct': gap_today_pct,
                'first5_open': agg['first5_open'],
                'first5_high': agg['first5_high'],
                'first5_low': agg['first5_low'],
                'first5_close': agg['first5_close'],
                'first5_volume': agg['first5_volume'],
                'first5_pace': first5_pace,
                'first5_range_pos': first5_range_pos,
                'first5_oc_ret': first5_oc_ret,
                'vwap': agg['vwap'],
                'pass': passed,
                'fail_reason': '' if passed else '|'.join(fail_reasons),
                'open_exists': inspection['open_exists'],
                'minute_bars_in_0930_0935': inspection['minute_bars_in_0930_0935'],
                'first5_constructible': inspection['first5_constructible'],
                'missing_reason': '' if passed else inspection['missing_reason'],
                'remarks': inspection['remarks'],
            }
        )

    result_df = pd.DataFrame(rows).sort_values(['pass', 'symbol'], ascending=[False, True]).reset_index(drop=True)
    csv_path = scan_path(actual_trade_date)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    final_candidate_count = int(result_df['pass'].fillna(False).sum()) if not result_df.empty else 0

    log_event(
        logger,
        'gap_0935_scan_complete',
        trade_date=actual_trade_date.isoformat(),
        watchlist_count=len(watchlist_df),
        open_ok_count=open_ok_count,
        first5_ok_count=first5_ok_count,
        gap_pass_count=gap_pass_count,
        range_pass_count=range_pass_count,
        pace_pass_count=pace_pass_count,
        vwap_pass_count=vwap_pass_count,
        final_candidate_count=final_candidate_count,
        output_path=str(csv_path),
    )

    return Gap0935ScanResult(
        trade_date=actual_trade_date,
        csv_path=csv_path,
        log_path=log_path,
        final_candidate_count=final_candidate_count,
    )
