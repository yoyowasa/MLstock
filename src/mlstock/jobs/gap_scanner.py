from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.config.schema import AppConfig
from mlstock.data.alpaca.client import AlpacaClient
from mlstock.data.storage.parquet import read_parquet
from mlstock.data.storage.paths import reference_seed_symbols_path


@dataclass(frozen=True)
class GapScannerSettings:
    min_gap_pct: float
    max_gap_pct: float
    min_volume_pace_ratio: float
    min_avg_volume_30d: float
    max_scan_candidates: int
    lookback_volume_days: int
    market_cap_source: str
    market_cap_delay_sec: float


@dataclass(frozen=True)
class UniverseSettings:
    use_mlstock_seed: bool
    min_price: float
    max_price: float
    min_market_cap_m: float


@dataclass(frozen=True)
class GapCandidate:
    symbol: str
    gap_pct: float
    prev_close: float
    open_price: float
    avg_volume_30d: float
    first_window_volume: float
    daily_volume_pace: float
    volume_pace_ratio: float
    market_cap_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _DailyStats:
    prev_close: float
    avg_volume: float


@dataclass(frozen=True)
class _OpenStats:
    open_price: float
    first_window_volume: float
    bars_in_window: int


@dataclass(frozen=True)
class _MarketCapLookup:
    market_cap_m: float | None
    source: str | None
    reason: str | None


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_settings(raw_cfg: Mapping[str, Any]) -> tuple[GapScannerSettings, UniverseSettings]:
    gap = raw_cfg.get("gap", {}) if isinstance(raw_cfg.get("gap"), dict) else {}
    options = raw_cfg.get("options", {}) if isinstance(raw_cfg.get("options"), dict) else {}
    universe = raw_cfg.get("universe", {}) if isinstance(raw_cfg.get("universe"), dict) else {}

    settings = GapScannerSettings(
        min_gap_pct=_to_float(gap.get("min_gap_pct"), 5.0),
        max_gap_pct=_to_float(gap.get("max_gap_pct"), 20.0),
        min_volume_pace_ratio=_to_float(gap.get("min_volume_pace_ratio"), 1.5),
        min_avg_volume_30d=max(_to_float(gap.get("min_avg_volume_30d"), 100000.0), 0.0),
        max_scan_candidates=max(_to_int(gap.get("max_scan_candidates"), 10), 1),
        lookback_volume_days=max(_to_int(gap.get("lookback_volume_days"), 30), 5),
        market_cap_source=str(gap.get("market_cap_source", "yfinance")).strip().lower(),
        market_cap_delay_sec=max(_to_float(options.get("yfinance_delay_sec"), 2.0), 0.0),
    )
    universe_settings = UniverseSettings(
        use_mlstock_seed=bool(universe.get("use_mlstock_seed", True)),
        min_price=_to_float(universe.get("min_price"), 5.0),
        max_price=_to_float(universe.get("max_price"), 100.0),
        min_market_cap_m=_to_float(universe.get("min_market_cap_m"), 300.0),
    )
    return settings, universe_settings


def _emit(logger: Optional[logging.Logger], message: str, **fields: Any) -> None:
    if logger is None:
        return
    from mlstock.logging.logger import log_event

    log_event(logger, message, **fields)


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()


def _extract_bar_symbol(item: Dict[str, Any]) -> Optional[str]:
    symbol = item.get("S") or item.get("symbol") or item.get("s")
    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    return text or None


def _fetch_bars_batch(
    client: AlpacaClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: str,
    adjustment: str,
    asof: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    collected: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in symbols}
    page_token: Optional[str] = None
    while True:
        response = client.get_bars(
            symbols=symbols,
            start=_iso_utc(start),
            end=_iso_utc(end),
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            asof=asof,
            page_token=page_token,
        )
        bars = response.get("bars") if isinstance(response, dict) else None
        if isinstance(bars, dict):
            for symbol, items in bars.items():
                key = str(symbol).upper()
                if key not in collected:
                    continue
                collected[key].extend([item for item in items if isinstance(item, dict)])
        elif isinstance(bars, list):
            if len(symbols) == 1:
                symbol = symbols[0]
                collected[symbol].extend([item for item in bars if isinstance(item, dict)])
            else:
                for item in bars:
                    if not isinstance(item, dict):
                        continue
                    symbol = _extract_bar_symbol(item)
                    if symbol and symbol in collected:
                        collected[symbol].append(item)
        page_token = response.get("next_page_token") if isinstance(response, dict) else None
        if not page_token:
            break
    return collected


def _load_symbols(cfg: AppConfig, use_mlstock_seed: bool, symbols: Optional[List[str]]) -> List[str]:
    if symbols:
        normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        return sorted(set(normalized))
    if not use_mlstock_seed:
        raise ValueError("symbols が未指定のため、universe.use_mlstock_seed=true で実行してください。")
    seed_path = reference_seed_symbols_path(cfg)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed symbols not found: {seed_path}")
    seed_df = read_parquet(seed_path)
    if seed_df.empty or "symbol" not in seed_df.columns:
        raise ValueError("seed_symbols が空、または symbol 列がありません。")
    raw_symbols = seed_df["symbol"].dropna().astype(str).str.strip().str.upper().tolist()
    return sorted(set(raw_symbols))


def _build_daily_stats(
    client: AlpacaClient,
    cfg: AppConfig,
    symbols: List[str],
    trade_date_local: date,
    lookback_volume_days: int,
) -> Dict[str, _DailyStats]:
    tz = ZoneInfo(cfg.project.timezone)
    start_local = datetime.combine(trade_date_local - timedelta(days=120), dtime(0, 0), tzinfo=tz)
    end_local = datetime.combine(trade_date_local + timedelta(days=1), dtime(0, 0), tzinfo=tz)
    stats: Dict[str, _DailyStats] = {}
    batch_size = max(1, min(200, int(cfg.bars.batch_size)))

    for batch in _chunk(symbols, batch_size):
        response = _fetch_bars_batch(
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
            items = response.get(symbol, [])
            if not items:
                continue
            rows: List[tuple[date, float, float]] = []
            for item in items:
                ts_raw = item.get("t")
                close_raw = item.get("c")
                volume_raw = item.get("v")
                if ts_raw is None or close_raw is None or volume_raw is None:
                    continue
                ts = pd.to_datetime(ts_raw, utc=True)
                local_date = ts.tz_convert(tz).date()
                if local_date >= trade_date_local:
                    continue
                try:
                    close = float(close_raw)
                    volume = float(volume_raw)
                except (TypeError, ValueError):
                    continue
                rows.append((local_date, close, volume))
            if not rows:
                continue
            rows.sort(key=lambda row: row[0])
            prev_close = rows[-1][1]
            volumes = [row[2] for row in rows[-lookback_volume_days:] if row[2] >= 0]
            if not volumes:
                continue
            avg_volume = sum(volumes) / float(len(volumes))
            stats[symbol] = _DailyStats(prev_close=prev_close, avg_volume=avg_volume)
    return stats


def _build_open_stats(
    client: AlpacaClient,
    cfg: AppConfig,
    symbols: List[str],
    trade_date_local: date,
    as_of_local: datetime,
) -> Dict[str, _OpenStats]:
    tz = ZoneInfo(cfg.project.timezone)
    open_local = datetime.combine(trade_date_local, dtime(9, 30), tzinfo=tz)
    close_local = min(as_of_local, datetime.combine(trade_date_local, dtime(9, 35), tzinfo=tz))
    if close_local <= open_local:
        return {}

    stats: Dict[str, _OpenStats] = {}
    batch_size = max(1, min(200, int(cfg.bars.batch_size)))
    for batch in _chunk(symbols, batch_size):
        response = _fetch_bars_batch(
            client=client,
            symbols=batch,
            start=open_local,
            end=close_local,
            timeframe="1Min",
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
        )
        for symbol in batch:
            items = response.get(symbol, [])
            if not items:
                continue
            normalized: List[tuple[datetime, float, float]] = []
            for item in items:
                ts_raw = item.get("t")
                open_raw = item.get("o")
                volume_raw = item.get("v")
                if ts_raw is None or open_raw is None or volume_raw is None:
                    continue
                ts = pd.to_datetime(ts_raw, utc=True).tz_convert(tz).to_pydatetime()
                if ts < open_local or ts >= datetime.combine(trade_date_local, dtime(9, 35), tzinfo=tz):
                    continue
                try:
                    open_price = float(open_raw)
                    volume = float(volume_raw)
                except (TypeError, ValueError):
                    continue
                normalized.append((ts, open_price, volume))
            if not normalized:
                continue
            normalized.sort(key=lambda row: row[0])
            open_price = normalized[0][1]
            first_window_volume = sum(row[2] for row in normalized)
            stats[symbol] = _OpenStats(
                open_price=open_price,
                first_window_volume=first_window_volume,
                bars_in_window=len(normalized),
            )
    return stats


def _safe_mapping_get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        try:
            if isinstance(mapping, dict):
                value = mapping.get(key)
            elif hasattr(mapping, "get"):
                value = mapping.get(key)
            else:
                value = None
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _extract_market_cap_lookup(ticker: Any) -> _MarketCapLookup:
    fast_info = getattr(ticker, "fast_info", None)
    market_cap = _safe_mapping_get(fast_info, "marketCap", "market_cap")
    if market_cap is not None:
        try:
            return _MarketCapLookup(market_cap_m=float(market_cap) / 1_000_000.0, source="fast_info", reason=None)
        except (TypeError, ValueError):
            pass

    info = ticker.info
    if isinstance(info, dict):
        market_cap = info.get("marketCap")
        if market_cap is not None:
            try:
                return _MarketCapLookup(market_cap_m=float(market_cap) / 1_000_000.0, source="info", reason=None)
            except (TypeError, ValueError):
                pass

        shares = info.get("sharesOutstanding") or _safe_mapping_get(fast_info, "shares")
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
            or _safe_mapping_get(fast_info, "lastPrice")
        )
        if shares is not None and price is not None:
            try:
                market_cap_calc = float(shares) * float(price)
                if market_cap_calc > 0:
                    return _MarketCapLookup(
                        market_cap_m=market_cap_calc / 1_000_000.0,
                        source="shares_x_price",
                        reason=None,
                    )
            except (TypeError, ValueError):
                pass

        quote_type = info.get("quoteType")
        if quote_type:
            return _MarketCapLookup(market_cap_m=None, source=None, reason=f"market_cap_missing:{quote_type}")

    return _MarketCapLookup(market_cap_m=None, source=None, reason="market_cap_missing")


def _fetch_market_caps_m(
    symbols: List[str],
    delay_sec: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    if not symbols:
        return {}
    try:
        import yfinance as yf
    except Exception as exc:
        _emit(logger, "market_cap_import_failed", error=str(exc))
        return {}

    results: Dict[str, float] = {}
    failures: List[Dict[str, str]] = []
    for idx, symbol in enumerate(symbols):
        lookup: _MarketCapLookup | None = None
        last_error: str | None = None
        for _attempt in range(2):
            try:
                ticker = yf.Ticker(symbol)
                lookup = _extract_market_cap_lookup(ticker)
                if lookup.market_cap_m is not None:
                    results[symbol] = lookup.market_cap_m
                    break
                last_error = lookup.reason
            except Exception as exc:
                last_error = str(exc)
                lookup = None
                if delay_sec > 0:
                    time.sleep(delay_sec)
        if symbol not in results:
            failures.append({"symbol": symbol, "reason": last_error or "unknown"})
        if delay_sec > 0 and idx < len(symbols) - 1:
            time.sleep(delay_sec)

    _emit(
        logger,
        "market_cap_fetch_summary",
        requested=len(symbols),
        fetched=len(results),
        missing=len(failures),
        failures=failures[:20],
    )
    return results


def scan_gap_candidates(
    cfg: AppConfig,
    gap_cfg: Mapping[str, Any],
    data_client: AlpacaClient,
    logger: Optional[logging.Logger] = None,
    as_of: Optional[datetime] = None,
    symbols: Optional[List[str]] = None,
) -> List[GapCandidate]:
    settings, universe = _load_settings(gap_cfg)
    tz = ZoneInfo(cfg.project.timezone)
    as_of_local = (as_of or datetime.now(tz)).astimezone(tz)
    trade_date = as_of_local.date()

    universe_symbols = _load_symbols(cfg, universe.use_mlstock_seed, symbols)
    if not universe_symbols:
        return []

    daily_stats = _build_daily_stats(
        client=data_client,
        cfg=cfg,
        symbols=universe_symbols,
        trade_date_local=trade_date,
        lookback_volume_days=settings.lookback_volume_days,
    )
    if not daily_stats:
        return []

    open_stats = _build_open_stats(
        client=data_client,
        cfg=cfg,
        symbols=list(daily_stats.keys()),
        trade_date_local=trade_date,
        as_of_local=as_of_local,
    )
    if not open_stats:
        return []

    raw_candidates: List[tuple[str, float, float, float, float, float, float]] = []
    for symbol, dstat in daily_stats.items():
        ostat = open_stats.get(symbol)
        if ostat is None:
            continue
        if dstat.prev_close <= 0 or dstat.avg_volume <= 0:
            continue
        if dstat.avg_volume < settings.min_avg_volume_30d:
            continue
        open_price = float(ostat.open_price)
        if open_price < universe.min_price or open_price > universe.max_price:
            continue
        gap_pct = (open_price - dstat.prev_close) / dstat.prev_close * 100.0
        if gap_pct < settings.min_gap_pct or gap_pct > settings.max_gap_pct:
            continue
        bars_in_window = max(1, min(5, ostat.bars_in_window))
        daily_volume_pace = float(ostat.first_window_volume) * (390.0 / float(bars_in_window))
        volume_pace_ratio = daily_volume_pace / dstat.avg_volume
        if volume_pace_ratio < settings.min_volume_pace_ratio:
            continue
        raw_candidates.append(
            (
                symbol,
                gap_pct,
                dstat.prev_close,
                open_price,
                dstat.avg_volume,
                float(ostat.first_window_volume),
                daily_volume_pace,
                volume_pace_ratio,
            )
        )
    if not raw_candidates:
        return []

    symbols_for_caps = [row[0] for row in raw_candidates]
    market_caps_m = (
        _fetch_market_caps_m(symbols_for_caps, settings.market_cap_delay_sec, logger=logger)
        if settings.market_cap_source == "yfinance"
        else {}
    )

    market_cap_filter_enabled = settings.market_cap_source == "yfinance" and universe.min_market_cap_m > 0
    candidates: List[GapCandidate] = []
    for row in raw_candidates:
        symbol, gap_pct, prev_close, open_price, avg_volume, first_window_volume, daily_volume_pace, volume_pace_ratio = (
            row
        )
        market_cap_m = market_caps_m.get(symbol, -1.0)
        if market_cap_filter_enabled:
            if market_cap_m < 0:
                _emit(logger, "market_cap_filter_drop", symbol=symbol, reason="market_cap_unavailable")
                continue
            if market_cap_m < universe.min_market_cap_m:
                _emit(
                    logger,
                    "market_cap_filter_drop",
                    symbol=symbol,
                    reason="market_cap_below_min",
                    market_cap_m=market_cap_m,
                    min_market_cap_m=universe.min_market_cap_m,
                )
                continue
        candidates.append(
            GapCandidate(
                symbol=symbol,
                gap_pct=gap_pct,
                prev_close=prev_close,
                open_price=open_price,
                avg_volume_30d=avg_volume,
                first_window_volume=first_window_volume,
                daily_volume_pace=daily_volume_pace,
                volume_pace_ratio=volume_pace_ratio,
                market_cap_m=market_cap_m,
            )
        )

    candidates.sort(key=lambda item: (item.volume_pace_ratio, item.gap_pct), reverse=True)
    return candidates[: settings.max_scan_candidates]
