from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from .common import (
    ET,
    chunked,
    fetch_bars_batch,
    get_previous_trading_day,
    get_trading_days,
    load_alpaca_client,
    load_seed_symbols,
)
from .config import StrategyConfig, load_strategy_config
from .gap_0935_watchlist_scanner import _inspect_first5_window
from .logutil import build_strategy_logger, log_event
from .metadata import classify_suffix_pattern, market_cap_bucket
from .paths import reports_dir


@dataclass(frozen=True)
class Phase1AnalysisResult:
    summary_path: Path
    daily_counts_path: Path
    daily_drop_counts_path: Path
    symbol_detail_path: Path
    missing_first5_detail_path: Path
    missing_first5_daily_path: Path
    missing_first5_symbol_path: Path
    coverage_by_type_path: Path
    coverage_by_exchange_path: Path
    coverage_by_market_cap_path: Path
    compare_path: Path
    start_date: date
    end_date: date
    trade_days: int


def _assets_reference() -> pd.DataFrame:
    path = Path(r"C:\BOT\MLStock\data\reference\assets.parquet")
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "exchange"])
    df = pd.read_parquet(path)
    keep = [col for col in ["symbol", "name", "exchange"] if col in df.columns]
    return df[keep].copy()


def _load_old_gap_log_index(log_dir: Path) -> tuple[Dict[date, int], Dict[date, set[str]]]:
    counts: Dict[date, int] = {}
    symbols_by_date: Dict[date, set[str]] = {}
    for path in sorted(log_dir.glob("gap_trade_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("message") != "scanner_complete":
                continue
            trade_date = payload.get("trade_date")
            count = payload.get("count")
            symbols = payload.get("symbols") or []
            if trade_date:
                dt = date.fromisoformat(str(trade_date))
            else:
                ts_utc = payload.get("ts_utc")
                if not ts_utc:
                    continue
                dt = pd.to_datetime(ts_utc, utc=True).tz_convert(ET).date()
            if isinstance(count, int):
                counts[dt] = count
            symbols_by_date[dt] = {str(symbol).upper() for symbol in symbols if str(symbol).strip()}
    return counts, symbols_by_date


def _infer_quote_type(name: str, suffix_pattern: str) -> str:
    upper = str(name or "").upper()
    if "ETF" in upper:
        return "ETF"
    if "ADR" in upper:
        return "ADR"
    if "PREFERRED" in upper:
        return "PREFERRED"
    if suffix_pattern == "unit" or " UNIT" in upper or "UNITS" in upper:
        return "UNIT"
    if suffix_pattern == "rights" or " RIGHTS" in upper or " RIGHTS" in upper:
        return "RIGHTS"
    if suffix_pattern == "warrant" or " WARRANT" in upper or "WARRANTS" in upper:
        return "WARRANT"
    if "COMMON STOCK" in upper or "ORDINARY SHARE" in upper or "CLASS A" in upper:
        return "EQUITY"
    return "UNKNOWN"


def _infer_security_type(name: str, suffix_pattern: str, quote_type: str, exchange: str) -> str:
    upper = str(name or "").upper()
    if suffix_pattern != "common_like":
        return suffix_pattern
    if quote_type == "ETF" or "ETF" in upper:
        return "etf"
    if quote_type == "ADR" or " ADR" in upper:
        return "adr"
    if quote_type == "PREFERRED" or "PREFERRED" in upper:
        return "preferred"
    if "OTC" in str(exchange or "").upper():
        return "otc"
    return "common_stock"


def _build_symbol_profiles(symbols: List[str], trade_candidates: set[str], cache_path: Path) -> pd.DataFrame:
    assets = _assets_reference()
    assets["symbol"] = assets["symbol"].astype(str).str.upper()
    assets_map = assets.set_index("symbol").to_dict(orient="index")
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty and "symbol" in cached.columns:
        cached["symbol"] = cached["symbol"].astype(str).str.upper()
    cached_map = {row["symbol"]: row for row in cached.to_dict(orient="records")} if not cached.empty else {}
    rows: List[Dict[str, Any]] = []
    to_fetch = sorted(symbol for symbol in trade_candidates if symbol not in cached_map)
    for symbol in to_fetch:
        info: Dict[str, Any] = {}
        try:
            info = getattr(yf.Ticker(symbol), "info", {}) or {}
        except Exception:
            info = {}
        name = str(info.get("longName") or info.get("shortName") or assets_map.get(symbol, {}).get("name") or "")
        exchange = str(info.get("exchange") or assets_map.get(symbol, {}).get("exchange") or "UNKNOWN")
        suffix_pattern = classify_suffix_pattern(symbol)
        quote_type = str(info.get("quoteType") or _infer_quote_type(name, suffix_pattern)).upper()
        raw_market_cap = info.get("marketCap")
        try:
            market_cap = float(raw_market_cap) if raw_market_cap is not None else None
        except (TypeError, ValueError):
            market_cap = None
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "quote_type": quote_type,
                "sector": str(info.get("sector") or ""),
                "suffix_pattern": suffix_pattern,
                "security_type": _infer_security_type(name, suffix_pattern, quote_type, exchange),
                "market_cap": market_cap,
                "market_cap_bucket": market_cap_bucket(market_cap),
            }
        )
    merged = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True) if rows else cached.copy()
    if merged.empty:
        merged = pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "exchange",
                "quote_type",
                "sector",
                "suffix_pattern",
                "security_type",
                "market_cap",
                "market_cap_bucket",
            ]
        )
    merged = merged.drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)

    profiles: List[Dict[str, Any]] = []
    merged_map = {row["symbol"]: row for row in merged.to_dict(orient="records")}
    for symbol in symbols:
        asset_row = assets_map.get(symbol, {})
        suffix_pattern = classify_suffix_pattern(symbol)
        cached_row = merged_map.get(symbol, {})
        name = str(cached_row.get("name") or asset_row.get("name") or "")
        exchange = str(cached_row.get("exchange") or asset_row.get("exchange") or "UNKNOWN")
        quote_type = str(cached_row.get("quote_type") or _infer_quote_type(name, suffix_pattern)).upper()
        market_cap = cached_row.get("market_cap")
        if pd.isna(market_cap):
            market_cap = None
        profiles.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "quote_type": quote_type,
                "sector": str(cached_row.get("sector") or ""),
                "suffix_pattern": suffix_pattern,
                "security_type": str(
                    cached_row.get("security_type") or _infer_security_type(name, suffix_pattern, quote_type, exchange)
                ),
                "market_cap": market_cap,
                "market_cap_bucket": str(cached_row.get("market_cap_bucket") or market_cap_bucket(market_cap)),
            }
        )
    return pd.DataFrame(profiles)


def _build_daily_frame(symbols: List[str], start_trade_date: date, end_trade_date: date) -> pd.DataFrame:
    cfg, client = load_alpaca_client()
    start_local = datetime.combine(start_trade_date - timedelta(days=40), dtime(0, 0), tzinfo=ET)
    end_local = datetime.combine(end_trade_date + timedelta(days=1), dtime(0, 0), tzinfo=ET)
    rows: List[Dict[str, Any]] = []
    for batch in chunked(symbols, min(200, int(cfg.bars.batch_size))):
        response = fetch_bars_batch(
            client=client,
            symbols=batch,
            start=start_local,
            end=end_local,
            timeframe="1Day",
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
        )
        for symbol in batch:
            for item in response.get(symbol, []):
                try:
                    ts = pd.to_datetime(item.get("t"), utc=True).tz_convert(ET)
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": ts.date(),
                            "open": float(item.get("o")),
                            "high": float(item.get("h")),
                            "low": float(item.get("l")),
                            "close": float(item.get("c")),
                            "volume": float(item.get("v")),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def _minute_window_map(symbols: List[str], trade_date: date) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}
    cfg, client = load_alpaca_client()
    start_local = datetime.combine(trade_date, dtime(9, 30), tzinfo=ET)
    end_local = datetime.combine(trade_date, dtime(9, 35, 5), tzinfo=ET)
    bars_by_symbol = fetch_bars_batch(
        client=client,
        symbols=symbols,
        start=start_local,
        end=end_local,
        timeframe="1Min",
        feed=cfg.bars.feed,
        adjustment=cfg.bars.adjustment,
        asof=cfg.bars.asof,
    )
    return {symbol: _inspect_first5_window(symbol, items) for symbol, items in bars_by_symbol.items()}


def _analysis_trade_dates(months: int, end_date: date) -> List[date]:
    _, client = load_alpaca_client()
    start_anchor = end_date - timedelta(days=max(90, months * 31))
    return [
        d
        for d in get_trading_days(client, end_date, days_before=(end_date - start_anchor).days + 15, days_after=0)
        if start_anchor <= d <= end_date
    ]


def analyze_phase1_population(
    months: int = 3, end_date: Optional[date] = None, config: Optional[StrategyConfig] = None
) -> Phase1AnalysisResult:
    strategy_cfg = config or load_strategy_config()
    _, client = load_alpaca_client()
    actual_end_date = end_date or get_previous_trading_day(client, datetime.now(ET).date() + timedelta(days=1))
    trade_dates = _analysis_trade_dates(months=months, end_date=actual_end_date)
    start_date = trade_dates[0]
    universe = load_seed_symbols()
    logger, log_path = build_strategy_logger("gap_d1_0935_phase1_analysis", "gap_d1_0935_phase1_analysis")

    daily_df = _build_daily_frame(universe, start_date, actual_end_date)
    asset_profiles_candidates: set[str] = set()
    for symbol, frame in daily_df.groupby("symbol"):
        closes = frame["close"]
        vols = frame["volume"]
        avg_vol = vols.rolling(strategy_cfg.d1.lookback_days).mean()
        avg_dollar = (closes * vols).rolling(strategy_cfg.d1.lookback_days).mean()
        if (
            (closes.between(strategy_cfg.universe.min_close, strategy_cfg.universe.max_close))
            & (avg_vol >= strategy_cfg.universe.min_avg_volume_20)
            & (avg_dollar >= strategy_cfg.universe.min_avg_dollar_volume_20)
        ).any():
            asset_profiles_candidates.add(symbol)
    profiles_df = _build_symbol_profiles(
        universe, asset_profiles_candidates, reports_dir() / "symbol_profiles_cache.csv"
    )
    profiles_map = {row["symbol"]: row for row in profiles_df.to_dict(orient="records")}

    old_counts, old_symbols = _load_old_gap_log_index(Path(r"C:\BOT\MLStock\artifacts\logs"))
    daily_count_rows: List[Dict[str, Any]] = []
    daily_drop_rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    missing_first5_rows: List[Dict[str, Any]] = []

    grouped_daily = {
        symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in daily_df.groupby("symbol")
    }

    for trade_date in trade_dates:
        previous_date = get_previous_trading_day(client, trade_date)
        open_map = _minute_window_map(universe, trade_date)
        watchlist_symbols: List[str] = []
        d1_drop = Counter()
        scan_drop = Counter()
        old_symbol_set = old_symbols.get(trade_date, set())

        for symbol in universe:
            profile = profiles_map.get(symbol, {})
            frame = grouped_daily.get(symbol)
            has_daily_bar = False
            d1_fail_reason = ""
            prev_gap_pct = None
            rel_vol_prev = None
            close_in_range_prev = None
            oc_ret_prev = None
            avg_volume_20 = None
            close_d1 = None
            if frame is None or frame.empty:
                d1_fail_reason = "missing_daily"
            else:
                frame = frame[frame["date"] <= previous_date]
                if len(frame) < max(strategy_cfg.d1.lookback_days, 2):
                    d1_fail_reason = "missing_daily"
                else:
                    has_daily_bar = True
                    latest = frame.iloc[-1]
                    prior = frame.iloc[-2]
                    tail = frame.tail(strategy_cfg.d1.lookback_days)
                    avg_volume_20 = float(tail["volume"].mean())
                    avg_dollar_volume_20 = float((tail["close"] * tail["volume"]).mean())
                    close_d1 = float(latest["close"])
                    prev_gap_pct = (float(latest["open"]) - float(prior["close"])) / float(prior["close"]) * 100.0
                    rel_vol_prev = (
                        float(latest["volume"]) / avg_volume_20 if avg_volume_20 and avg_volume_20 > 0 else 0.0
                    )
                    day_range = float(latest["high"]) - float(latest["low"])
                    close_in_range_prev = (
                        (float(latest["close"]) - float(latest["low"])) / day_range if day_range > 0 else 0.0
                    )
                    oc_ret_prev = (
                        (float(latest["close"]) - float(latest["open"])) / float(latest["open"]) * 100.0
                        if float(latest["open"]) > 0
                        else 0.0
                    )

                    if not (strategy_cfg.universe.min_close <= close_d1 <= strategy_cfg.universe.max_close):
                        d1_fail_reason = "price_fail"
                    elif (
                        avg_volume_20 < strategy_cfg.universe.min_avg_volume_20
                        or avg_dollar_volume_20 < strategy_cfg.universe.min_avg_dollar_volume_20
                    ):
                        d1_fail_reason = "liquidity_fail"
                    elif str(profile.get("security_type", "common_stock")) != "common_stock":
                        d1_fail_reason = "non_common_fail"
                    else:
                        market_cap = profile.get("market_cap")
                        if (
                            market_cap is None
                            or market_cap < strategy_cfg.universe.min_market_cap
                            or market_cap > strategy_cfg.universe.max_market_cap
                        ):
                            d1_fail_reason = "market_cap_fail"
                        elif prev_gap_pct < strategy_cfg.d1.min_prev_gap_pct:
                            d1_fail_reason = "gap_fail"
                        elif rel_vol_prev < strategy_cfg.d1.min_rel_vol_prev:
                            d1_fail_reason = "rel_vol_fail"
                        elif (
                            close_in_range_prev < strategy_cfg.d1.min_close_in_range_prev
                            or oc_ret_prev <= strategy_cfg.d1.min_oc_ret_prev
                        ):
                            d1_fail_reason = "close_strength_fail"
                        else:
                            d1_fail_reason = "selected"
                            watchlist_symbols.append(symbol)
            d1_drop[d1_fail_reason] += 1

            inspection = open_map.get(
                symbol,
                {
                    "symbol": symbol,
                    "open_exists": False,
                    "minute_bars_in_0930_0935": 0,
                    "first5_constructible": False,
                    "missing_reason": "no_minute_bars",
                    "remarks": "",
                    "aggregate": None,
                },
            )
            agg = inspection.get("aggregate")
            has_open_1m_window = agg is not None
            scan_fail_reason = ""
            scan_pass = False
            gap_today_pct = None
            first5_oc_ret = None
            first5_pace = None
            if d1_fail_reason == "selected":
                if agg is None:
                    scan_fail_reason = "missing_first5"
                    missing_first5_rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "symbol": symbol,
                            "open_exists": bool(inspection.get("open_exists")),
                            "minute_bars_in_0930_0935": int(inspection.get("minute_bars_in_0930_0935", 0) or 0),
                            "first5_constructible": bool(inspection.get("first5_constructible")),
                            "missing_reason": str(inspection.get("missing_reason") or "unknown"),
                            "remarks": str(inspection.get("remarks") or ""),
                        }
                    )
                else:
                    close_prev = float(close_d1 or 0.0)
                    gap_today_pct = (agg["open_D"] - close_prev) / close_prev * 100.0 if close_prev > 0 else 0.0
                    first5_range = agg["first5_high"] - agg["first5_low"]
                    first5_range_pos = (
                        (agg["first5_close"] - agg["first5_low"]) / first5_range if first5_range > 0 else 0.0
                    )
                    first5_oc_ret = (
                        (agg["first5_close"] - agg["first5_open"]) / agg["first5_open"] * 100.0
                        if agg["first5_open"] > 0
                        else 0.0
                    )
                    first5_pace = (
                        (agg["first5_volume"] * 78.0) / avg_volume_20 if avg_volume_20 and avg_volume_20 > 0 else 0.0
                    )
                    gap_ok = agg["open_D"] > close_prev and gap_today_pct >= strategy_cfg.day0.min_gap_today_pct
                    range_ok = first5_range_pos >= strategy_cfg.day0.min_first5_range_pos
                    oc_ok = first5_oc_ret >= strategy_cfg.day0.min_first5_oc_ret
                    pace_ok = first5_pace >= strategy_cfg.day0.min_first5_pace
                    vwap_ok = agg["first5_close"] >= agg["vwap"] * strategy_cfg.day0.min_close_vs_vwap_ratio
                    if gap_ok and range_ok and oc_ok and pace_ok and vwap_ok:
                        scan_pass = True
                        scan_fail_reason = "pass"
                    else:
                        reasons: List[str] = []
                        if not gap_ok:
                            reasons.append("gap_fail")
                        if not range_ok:
                            reasons.append("range_fail")
                        if not oc_ok:
                            reasons.append("oc_ret_fail")
                        if not pace_ok:
                            reasons.append("pace_fail")
                        if not vwap_ok:
                            reasons.append("vwap_fail")
                        scan_fail_reason = "|".join(reasons)
                scan_drop[scan_fail_reason or "pass"] += 1

            symbol_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "symbol": symbol,
                    "has_daily_bar": has_daily_bar,
                    "has_open_1m_window": has_open_1m_window,
                    "quote_type": profile.get("quote_type", "UNKNOWN"),
                    "exchange": profile.get("exchange", "UNKNOWN"),
                    "market_cap_bucket": profile.get("market_cap_bucket", "unknown"),
                    "suffix_pattern": profile.get("suffix_pattern", "common_like"),
                    "security_type": profile.get("security_type", "common_stock"),
                    "d1_fail_reason": d1_fail_reason,
                    "scan_fail_reason": scan_fail_reason,
                    "gap_today_pct": gap_today_pct,
                    "first5_oc_ret": first5_oc_ret,
                    "first5_pace": first5_pace,
                    "open_exists": bool(inspection.get("open_exists")),
                    "minute_bars_in_0930_0935": int(inspection.get("minute_bars_in_0930_0935", 0) or 0),
                    "first5_constructible": bool(inspection.get("first5_constructible")),
                    "missing_reason": str(inspection.get("missing_reason") or ""),
                    "remarks": str(inspection.get("remarks") or ""),
                    "old_candidate": symbol in old_symbol_set,
                    "new_candidate": scan_pass,
                }
            )

        new_symbol_set = {
            row["symbol"] for row in symbol_rows if row["trade_date"] == trade_date.isoformat() and row["new_candidate"]
        }
        common_count = len(new_symbol_set & old_symbol_set)
        old_only_count = len(old_symbol_set - new_symbol_set)
        new_only_count = len(new_symbol_set - old_symbol_set)
        daily_count_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "watchlist_count": len(watchlist_symbols),
                "candidate_0935_count": len(new_symbol_set),
                "old_gap_count": old_counts.get(trade_date),
                "common_count": common_count if old_symbol_set else None,
                "old_only_count": old_only_count if old_symbol_set else None,
                "new_only_count": new_only_count if old_symbol_set else None,
            }
        )
        daily_drop_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "universe_count": len(universe),
                "d1_missing_daily_count": d1_drop.get("missing_daily", 0),
                "d1_price_fail_count": d1_drop.get("price_fail", 0),
                "d1_liquidity_fail_count": d1_drop.get("liquidity_fail", 0),
                "d1_non_common_fail_count": d1_drop.get("non_common_fail", 0),
                "d1_market_cap_fail_count": d1_drop.get("market_cap_fail", 0),
                "d1_gap_fail_count": d1_drop.get("gap_fail", 0),
                "d1_rel_vol_fail_count": d1_drop.get("rel_vol_fail", 0),
                "d1_close_strength_fail_count": d1_drop.get("close_strength_fail", 0),
                "watchlist_count": len(watchlist_symbols),
                "scan_missing_first5_count": scan_drop.get("missing_first5", 0),
                "scan_gap_fail_count": sum(
                    1
                    for row in symbol_rows
                    if row["trade_date"] == trade_date.isoformat()
                    and "gap_fail" in str(row["scan_fail_reason"]).split("|")
                ),
                "scan_range_fail_count": sum(
                    1
                    for row in symbol_rows
                    if row["trade_date"] == trade_date.isoformat()
                    and "range_fail" in str(row["scan_fail_reason"]).split("|")
                ),
                "scan_oc_ret_fail_count": sum(
                    1
                    for row in symbol_rows
                    if row["trade_date"] == trade_date.isoformat()
                    and "oc_ret_fail" in str(row["scan_fail_reason"]).split("|")
                ),
                "scan_pace_fail_count": sum(
                    1
                    for row in symbol_rows
                    if row["trade_date"] == trade_date.isoformat()
                    and "pace_fail" in str(row["scan_fail_reason"]).split("|")
                ),
                "scan_vwap_fail_count": sum(
                    1
                    for row in symbol_rows
                    if row["trade_date"] == trade_date.isoformat()
                    and "vwap_fail" in str(row["scan_fail_reason"]).split("|")
                ),
                "candidate_0935_count": len(new_symbol_set),
            }
        )

    symbol_df = pd.DataFrame(symbol_rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    daily_counts_df = pd.DataFrame(daily_count_rows).sort_values("trade_date").reset_index(drop=True)
    drop_df = pd.DataFrame(daily_drop_rows).sort_values("trade_date").reset_index(drop=True)
    missing_first5_df = pd.DataFrame(missing_first5_rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    if missing_first5_df.empty:
        missing_first5_df = pd.DataFrame(
            columns=[
                "trade_date",
                "symbol",
                "open_exists",
                "minute_bars_in_0930_0935",
                "first5_constructible",
                "missing_reason",
                "remarks",
            ]
        )
    missing_first5_daily = (
        missing_first5_df.groupby(["trade_date", "missing_reason"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("trade_date")
    )
    if missing_first5_daily.empty:
        missing_first5_daily = pd.DataFrame(columns=["trade_date"])
    if not daily_counts_df.empty:
        missing_daily_totals = (
            missing_first5_df.groupby("trade_date").size().rename("missing_first5_count").reset_index()
            if not missing_first5_df.empty
            else pd.DataFrame(columns=["trade_date", "missing_first5_count"])
        )
        missing_first5_daily = daily_counts_df[["trade_date", "watchlist_count", "candidate_0935_count"]].merge(
            missing_daily_totals,
            on="trade_date",
            how="left",
        )
        missing_first5_daily["missing_first5_count"] = (
            missing_first5_daily["missing_first5_count"].fillna(0).astype(int)
        )
        if not missing_first5_df.empty:
            reason_pivot = (
                missing_first5_df.groupby(["trade_date", "missing_reason"]).size().unstack(fill_value=0).reset_index()
            )
            missing_first5_daily = missing_first5_daily.merge(reason_pivot, on="trade_date", how="left")
        for col in missing_first5_daily.columns:
            if col != "trade_date":
                missing_first5_daily[col] = missing_first5_daily[col].fillna(0)
    missing_first5_symbol = (
        missing_first5_df.groupby(["symbol", "missing_reason"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "symbol", "missing_reason"], ascending=[False, True, True])
    )
    if missing_first5_symbol.empty:
        missing_first5_symbol = pd.DataFrame(columns=["symbol", "missing_reason", "count"])

    coverage_by_type = (
        symbol_df.groupby("security_type")
        .agg(
            symbol_days=("symbol", "count"),
            open_window_rate=("has_open_1m_window", "mean"),
            selected_rate=("new_candidate", "mean"),
        )
        .reset_index()
        .sort_values(["open_window_rate", "symbol_days"], ascending=[False, False])
    )
    coverage_by_exchange = (
        symbol_df.groupby("exchange")
        .agg(symbol_days=("symbol", "count"), open_window_rate=("has_open_1m_window", "mean"))
        .reset_index()
        .sort_values(["open_window_rate", "symbol_days"], ascending=[False, False])
    )
    coverage_by_market_cap = (
        symbol_df.groupby("market_cap_bucket")
        .agg(symbol_days=("symbol", "count"), open_window_rate=("has_open_1m_window", "mean"))
        .reset_index()
        .sort_values(["open_window_rate", "symbol_days"], ascending=[False, False])
    )

    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{start_date:%Y%m%d}_{actual_end_date:%Y%m%d}"
    summary_path = out_dir / f"phase1_population_summary_{stamp}.csv"
    daily_counts_path = out_dir / f"phase1_daily_counts_{stamp}.csv"
    daily_drop_counts_path = out_dir / f"phase1_daily_drop_counts_{stamp}.csv"
    symbol_detail_path = out_dir / f"phase1_symbol_detail_{stamp}.csv"
    coverage_by_type_path = out_dir / f"phase1_coverage_by_type_{stamp}.csv"
    coverage_by_exchange_path = out_dir / f"phase1_coverage_by_exchange_{stamp}.csv"
    coverage_by_market_cap_path = out_dir / f"phase1_coverage_by_market_cap_{stamp}.csv"
    compare_path = out_dir / f"phase1_old_vs_0935_{stamp}.csv"
    missing_first5_detail_path = out_dir / f"phase1_missing_first5_detail_{stamp}.csv"
    missing_first5_daily_path = out_dir / f"phase1_missing_first5_daily_{stamp}.csv"
    missing_first5_symbol_path = out_dir / f"phase1_missing_first5_symbol_{stamp}.csv"

    summary_df = pd.DataFrame(
        [
            {
                "start_date": start_date.isoformat(),
                "end_date": actual_end_date.isoformat(),
                "trade_days": len(trade_dates),
                "avg_watchlist_count": (
                    float(daily_counts_df["watchlist_count"].mean()) if not daily_counts_df.empty else 0.0
                ),
                "median_watchlist_count": (
                    float(daily_counts_df["watchlist_count"].median()) if not daily_counts_df.empty else 0.0
                ),
                "avg_candidate_0935_count": (
                    float(daily_counts_df["candidate_0935_count"].mean()) if not daily_counts_df.empty else 0.0
                ),
                "median_candidate_0935_count": (
                    float(daily_counts_df["candidate_0935_count"].median()) if not daily_counts_df.empty else 0.0
                ),
                "watchlist_zero_days": (
                    int((daily_counts_df["watchlist_count"] == 0).sum()) if not daily_counts_df.empty else 0
                ),
                "scan_zero_days": (
                    int((daily_counts_df["candidate_0935_count"] == 0).sum()) if not daily_counts_df.empty else 0
                ),
            }
        ]
    )

    summary_df.to_csv(summary_path, index=False)
    daily_counts_df.to_csv(daily_counts_path, index=False)
    drop_df.to_csv(daily_drop_counts_path, index=False)
    symbol_df.to_csv(symbol_detail_path, index=False)
    missing_first5_df.to_csv(missing_first5_detail_path, index=False)
    missing_first5_daily.to_csv(missing_first5_daily_path, index=False)
    missing_first5_symbol.to_csv(missing_first5_symbol_path, index=False)
    coverage_by_type.to_csv(coverage_by_type_path, index=False)
    coverage_by_exchange.to_csv(coverage_by_exchange_path, index=False)
    coverage_by_market_cap.to_csv(coverage_by_market_cap_path, index=False)
    daily_counts_df.to_csv(compare_path, index=False)

    log_event(
        logger,
        "phase1_population_analysis_complete",
        start_date=start_date.isoformat(),
        end_date=actual_end_date.isoformat(),
        trade_days=len(trade_dates),
        summary_path=str(summary_path),
        daily_counts_path=str(daily_counts_path),
        daily_drop_counts_path=str(daily_drop_counts_path),
        symbol_detail_path=str(symbol_detail_path),
        missing_first5_detail_path=str(missing_first5_detail_path),
    )

    return Phase1AnalysisResult(
        summary_path=summary_path,
        daily_counts_path=daily_counts_path,
        daily_drop_counts_path=daily_drop_counts_path,
        symbol_detail_path=symbol_detail_path,
        missing_first5_detail_path=missing_first5_detail_path,
        missing_first5_daily_path=missing_first5_daily_path,
        missing_first5_symbol_path=missing_first5_symbol_path,
        coverage_by_type_path=coverage_by_type_path,
        coverage_by_exchange_path=coverage_by_exchange_path,
        coverage_by_market_cap_path=coverage_by_market_cap_path,
        compare_path=compare_path,
        start_date=start_date,
        end_date=actual_end_date,
        trade_days=len(trade_dates),
    )
