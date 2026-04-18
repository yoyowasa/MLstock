from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import ET, chunked, fetch_bars_batch, get_previous_trading_day, infer_trade_date, load_alpaca_client, load_seed_symbols, to_local_ts
from .config import StrategyConfig, load_strategy_config
from .logutil import build_strategy_logger, log_event
from .metadata import fetch_symbol_profiles, should_exclude_non_common
from .paths import watchlist_path


@dataclass(frozen=True)
class WatchlistBuildResult:
    trade_date: date
    previous_date: date
    csv_path: Path
    log_path: Path
    selected_count: int


def _build_daily_frame(symbols: List[str], trade_date: date) -> pd.DataFrame:
    cfg, client = load_alpaca_client()
    previous_date = get_previous_trading_day(client, trade_date)
    start_local = datetime.combine(previous_date - timedelta(days=60), dtime(0, 0), tzinfo=ET)
    end_local = datetime.combine(trade_date + timedelta(days=1), dtime(0, 0), tzinfo=ET)
    rows: List[Dict[str, Any]] = []
    for batch in chunked(symbols, min(200, int(cfg.bars.batch_size))):
        response = fetch_bars_batch(
            client=client,
            symbols=batch,
            start=start_local,
            end=end_local,
            timeframe='1Day',
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
        )
        for symbol in batch:
            for item in response.get(symbol, []):
                try:
                    ts = to_local_ts(item.get('t'))
                    rows.append(
                        {
                            'symbol': symbol,
                            'date': ts.date(),
                            'open': float(item.get('o')),
                            'high': float(item.get('h')),
                            'low': float(item.get('l')),
                            'close': float(item.get('c')),
                            'volume': float(item.get('v')),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    return pd.DataFrame(rows)


def build_gap_d1_watchlist(
    trade_date: Optional[date] = None,
    symbols: Optional[List[str]] = None,
    config: Optional[StrategyConfig] = None,
) -> WatchlistBuildResult:
    cfg = config or load_strategy_config()
    _, client = load_alpaca_client()
    actual_trade_date = infer_trade_date(client, trade_date)
    previous_date = get_previous_trading_day(client, actual_trade_date)
    logger, log_path = build_strategy_logger('gap_d1_watchlist', 'gap_d1_watchlist')
    universe = sorted(set(symbols or load_seed_symbols()))
    daily_df = _build_daily_frame(universe, actual_trade_date)
    if daily_df.empty:
        raise ValueError('No daily bars fetched for watchlist build')
    daily_df = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    selected_rows: List[Dict[str, Any]] = []
    excluded_non_common = 0
    excluded_price = 0
    excluded_liquidity = 0
    excluded_market_cap = 0
    excluded_gap = 0
    excluded_rel_vol = 0
    excluded_close_strength = 0
    profiles_cache: Dict[str, Any] = {}

    for symbol, frame in daily_df.groupby('symbol'):
        frame = frame.sort_values('date').reset_index(drop=True)
        frame = frame[frame['date'] <= previous_date]
        if len(frame) < max(cfg.d1.lookback_days, 2):
            continue
        latest = frame.iloc[-1]
        prior = frame.iloc[-2]
        tail = frame.tail(cfg.d1.lookback_days)
        avg_volume_20 = float(tail['volume'].mean())
        avg_dollar_volume_20 = float((tail['close'] * tail['volume']).mean())
        close_d1 = float(latest['close'])

        if not (cfg.universe.min_close <= close_d1 <= cfg.universe.max_close):
            excluded_price += 1
            continue
        if avg_volume_20 < cfg.universe.min_avg_volume_20 or avg_dollar_volume_20 < cfg.universe.min_avg_dollar_volume_20:
            excluded_liquidity += 1
            continue

        if symbol not in profiles_cache:
            profiles_cache.update(fetch_symbol_profiles(client=client, symbols=[symbol], delay_sec=0.0))
        profile = profiles_cache[symbol]
        if should_exclude_non_common(profile.security_type):
            excluded_non_common += 1
            continue
        market_cap = profile.market_cap
        if market_cap is None or market_cap < cfg.universe.min_market_cap or market_cap > cfg.universe.max_market_cap:
            excluded_market_cap += 1
            continue

        prev_gap_pct = (float(latest['open']) - float(prior['close'])) / float(prior['close']) * 100.0
        rel_vol_prev = float(latest['volume']) / avg_volume_20 if avg_volume_20 > 0 else 0.0
        day_range = float(latest['high']) - float(latest['low'])
        close_in_range_prev = (float(latest['close']) - float(latest['low'])) / day_range if day_range > 0 else 0.0
        oc_ret_prev = (float(latest['close']) - float(latest['open'])) / float(latest['open']) * 100.0 if float(latest['open']) > 0 else 0.0

        if prev_gap_pct < cfg.d1.min_prev_gap_pct:
            excluded_gap += 1
            continue
        if rel_vol_prev < cfg.d1.min_rel_vol_prev:
            excluded_rel_vol += 1
            continue
        if close_in_range_prev < cfg.d1.min_close_in_range_prev or oc_ret_prev <= cfg.d1.min_oc_ret_prev:
            excluded_close_strength += 1
            continue

        selected_rows.append(
            {
                'symbol': symbol,
                'trade_date': actual_trade_date.isoformat(),
                'open_D-1': float(latest['open']),
                'high_D-1': float(latest['high']),
                'low_D-1': float(latest['low']),
                'close_D-1': float(latest['close']),
                'close_D-2': float(prior['close']),
                'prev_gap_pct': prev_gap_pct,
                'rel_vol_prev': rel_vol_prev,
                'close_in_range_prev': close_in_range_prev,
                'oc_ret_prev': oc_ret_prev,
                'market_cap': market_cap,
                'avg_volume_20': avg_volume_20,
                'avg_dollar_volume_20': avg_dollar_volume_20,
                'sector': profile.sector,
                'security_type': profile.security_type,
                'index_ret_D-1': None,
                'sector_ret_D-1': None,
                'selected_reason': 'prev_gap+rel_vol+close_strength',
                'quote_type': profile.quote_type,
                'exchange': profile.exchange,
                'suffix_pattern': profile.suffix_pattern,
                'market_cap_bucket': profile.market_cap_bucket,
            }
        )

    columns = [
        'symbol', 'trade_date', 'open_D-1', 'high_D-1', 'low_D-1', 'close_D-1', 'close_D-2', 'prev_gap_pct',
        'rel_vol_prev', 'close_in_range_prev', 'oc_ret_prev', 'market_cap', 'avg_volume_20', 'avg_dollar_volume_20',
        'sector', 'security_type', 'index_ret_D-1', 'sector_ret_D-1', 'selected_reason', 'quote_type', 'exchange',
        'suffix_pattern', 'market_cap_bucket'
    ]
    selected_df = pd.DataFrame(selected_rows, columns=columns).sort_values('symbol').reset_index(drop=True)
    csv_path = watchlist_path(actual_trade_date)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(csv_path, index=False)

    log_event(
        logger,
        'watchlist_complete',
        trade_date=actual_trade_date.isoformat(),
        previous_date=previous_date.isoformat(),
        universe_count=len(universe),
        excluded_non_common_count=excluded_non_common,
        excluded_price_count=excluded_price,
        excluded_liquidity_count=excluded_liquidity,
        excluded_market_cap_count=excluded_market_cap,
        excluded_gap_count=excluded_gap,
        excluded_rel_vol_count=excluded_rel_vol,
        excluded_close_strength_count=excluded_close_strength,
        selected_count=len(selected_df),
        output_path=str(csv_path),
    )

    return WatchlistBuildResult(
        trade_date=actual_trade_date,
        previous_date=previous_date,
        csv_path=csv_path,
        log_path=log_path,
        selected_count=len(selected_df),
    )
