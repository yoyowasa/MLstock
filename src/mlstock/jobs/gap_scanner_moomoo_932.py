"""
gap_scanner_moomoo_932.py  —  実験版 gap scanner (moomoo / 9:32 ET 判定)

【現行 scanner との違い】
  - データ源: moomoo OpenD (snapshot + K_1M)
  - open_price: snapshot.open_price  (9:30 bar 代替)
  - volume 指標: 9:31 single-bar volume / avg_volume_30d  (現行の 9:30-9:35 pace と非互換)
  - 判定時刻: 9:32:05 ET を想定 (9:31 bar が確定済み前提)
  - avg_volume_30d: Alpaca daily bars から取得 (moomoo にはなし)
  - market_cap: yfinance (現行と同じ)

【共通】
  - gap_pct 閾値: gap_config.yaml の min_gap_pct / max_gap_pct
  - 価格帯フィルタ: gap_config.yaml の min_price / max_price
  - market_cap フィルタ: gap_config.yaml の min_market_cap_m
  - universe: seed_symbols.parquet (現行と同じ)

【live/order/unlock_trade】
  - 一切触らない。scan-only 出力のみ。
"""

from __future__ import annotations

import logging
import socket
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

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoomooGapCandidate:
    symbol: str
    open_price: float
    prev_close_price: float
    gap_pct: float
    bar_0931_volume: float  # 9:31 bar の volume (単一 bar)
    avg_volume_30d: float
    volume_ratio: float  # bar_0931_volume / avg_volume_30d (※日次換算なし)
    volume_pace_annualized: float  # bar_0931_volume * 390  / avg_volume_30d (簡易換算)
    market_cap_m: float
    snapshot_update_time: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MoomooScanDiagnostics:
    trade_date: str
    universe_count: int
    snapshot_ok_count: int
    kline_0931_ok_count: int
    missing_snapshot_count: int
    missing_kline_count: int
    daily_stats_count: int
    price_filter_drop_count: int
    gap_filter_drop_count: int
    volume_filter_drop_count: int
    market_cap_drop_count: int
    final_candidate_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# OpenD preflight
# ---------------------------------------------------------------------------


def check_opend(host: str, port: int, timeout: float = 3.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


# ---------------------------------------------------------------------------
# Alpaca daily stats (avg_volume / prev_close)
# ---------------------------------------------------------------------------


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()


def _fetch_daily_stats(
    alpaca_client: AlpacaClient,
    cfg: AppConfig,
    symbols: List[str],
    trade_date: date,
    lookback_days: int = 30,
) -> Dict[str, tuple[float, float]]:
    """symbol -> (prev_close, avg_volume_30d) using Alpaca daily bars."""
    tz = ET
    start = datetime.combine(trade_date - timedelta(days=120), dtime(0, 0), tzinfo=tz)
    end = datetime.combine(trade_date, dtime(0, 0), tzinfo=tz)
    batch_size = max(1, min(200, int(cfg.bars.batch_size)))
    result: Dict[str, tuple[float, float]] = {}

    for batch in _chunk(symbols, batch_size):
        resp = alpaca_client.get_bars(
            symbols=batch,
            start=_iso_utc(start),
            end=_iso_utc(end),
            timeframe="1Day",
            feed=cfg.bars.feed,
            adjustment=cfg.bars.adjustment,
            asof=cfg.bars.asof,
        )
        bars_map = resp.get("bars", {}) if isinstance(resp, dict) else {}
        if not isinstance(bars_map, dict):
            continue
        for sym in batch:
            items = bars_map.get(sym, [])
            if not items:
                continue
            rows: list[tuple[date, float, float]] = []
            for item in items:
                ts_raw = item.get("t")
                close_raw = item.get("c")
                vol_raw = item.get("v")
                if None in (ts_raw, close_raw, vol_raw):
                    continue
                try:
                    local_date = pd.to_datetime(ts_raw, utc=True).tz_convert(tz).date()
                    close = float(close_raw)
                    vol = float(vol_raw)
                except (TypeError, ValueError):
                    continue
                if local_date >= trade_date:
                    continue
                rows.append((local_date, close, vol))
            if not rows:
                continue
            rows.sort(key=lambda r: r[0])
            prev_close = rows[-1][1]
            volumes = [r[2] for r in rows[-lookback_days:] if r[2] >= 0]
            if not volumes:
                continue
            result[sym] = (prev_close, sum(volumes) / len(volumes))
    return result


# ---------------------------------------------------------------------------
# moomoo snapshot batch
# ---------------------------------------------------------------------------


def _fetch_moomoo_snapshots(
    quote_ctx: Any,
    codes_mm: List[str],
    batch_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """symbol (no prefix) -> snapshot dict.

    moomoo は 1 銘柄でも Unknown だとバッチ全体が error を返す。
    バッチ失敗時は 1 銘柄ずつ fallback して未知銘柄をスキップする。
    """
    import moomoo as ft

    result: Dict[str, Dict[str, Any]] = {}

    def _try_batch(codes: List[str]) -> bool:
        ret, snap = quote_ctx.get_market_snapshot(codes)
        if ret != ft.RET_OK:
            return False
        for row in snap.to_dict(orient="records"):
            code = str(row.get("code", "")).replace("US.", "")
            result[code] = row
        return True

    for batch in _chunk(codes_mm, batch_size):
        if _try_batch(batch):
            continue
        # バッチ失敗 → 1 銘柄ずつ fallback
        for code in batch:
            _try_batch([code])

    return result


# ---------------------------------------------------------------------------
# moomoo K_1M 9:31 bar
# ---------------------------------------------------------------------------


def _fetch_moomoo_0931_bars(
    quote_ctx: Any,
    codes_mm: List[str],
    trade_date: date,
) -> Dict[str, Dict[str, Any]]:
    """symbol -> 9:31 bar dict {open, close, volume, ...}."""
    import moomoo as ft

    start_str = f"{trade_date.isoformat()} 09:31:00"
    end_str = f"{trade_date.isoformat()} 09:32:00"
    result: Dict[str, Dict[str, Any]] = {}

    # subscribe batch (必要)
    for batch in _chunk(codes_mm, 100):
        quote_ctx.subscribe(batch, [ft.SubType.K_1M], subscribe_push=False)

    for code in codes_mm:
        sym = code.replace("US.", "")
        ret, kdata, _ = quote_ctx.request_history_kline(
            code,
            start=start_str,
            end=end_str,
            ktype=ft.KLType.K_1M,
            autype=ft.AuType.NONE,
            max_count=5,
        )
        if ret != 0:
            continue
        if kdata is None or kdata.empty:
            continue
        # 9:31 bar のみ抽出
        kdata["time_key"] = pd.to_datetime(kdata["time_key"])
        target = pd.Timestamp(f"{trade_date.isoformat()} 09:31:00")
        bar = kdata[kdata["time_key"] == target]
        if bar.empty:
            # 9:31 がなければ最初の bar を使う (replay 時の揺れ許容)
            bar = kdata.head(1)
        row = bar.iloc[0].to_dict()
        result[sym] = row
    return result


# ---------------------------------------------------------------------------
# market cap (yfinance, same as gap_scanner.py)
# ---------------------------------------------------------------------------


def _fetch_market_caps_m(
    symbols: List[str],
    delay_sec: float,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    if not symbols:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        return {}

    results: Dict[str, float] = {}
    for idx, sym in enumerate(symbols):
        for _attempt in range(2):
            try:
                t = yf.Ticker(sym)
                fi = getattr(t, "fast_info", None)
                mc = None
                if fi is not None:
                    mc = getattr(fi, "marketCap", None) or (fi.get("marketCap") if hasattr(fi, "get") else None)
                if mc is None:
                    info = t.info
                    if isinstance(info, dict):
                        mc = info.get("marketCap")
                if mc is not None:
                    results[sym] = float(mc) / 1_000_000.0
                    break
            except Exception:
                if delay_sec > 0:
                    time.sleep(delay_sec)
        if delay_sec > 0 and idx < len(symbols) - 1:
            time.sleep(delay_sec)
    return results


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------


def scan_gap_candidates_moomoo_932(
    cfg: AppConfig,
    gap_cfg: Mapping[str, Any],
    alpaca_client: AlpacaClient,
    trade_date: date,
    opend_host: str = "127.0.0.1",
    opend_port: int = 11111,
    symbols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[List[MoomooGapCandidate], MoomooScanDiagnostics]:
    """
    9:32:05 ET 判定版 gap scanner (moomoo データ源)。

    Returns:
        (candidates, diagnostics)
    """
    import moomoo as ft

    # --- config ---
    gap = gap_cfg.get("gap", {}) if isinstance(gap_cfg.get("gap"), dict) else {}
    uni = gap_cfg.get("universe", {}) if isinstance(gap_cfg.get("universe"), dict) else {}
    opts = gap_cfg.get("options", {}) if isinstance(gap_cfg.get("options"), dict) else {}

    min_gap_pct = float(gap.get("min_gap_pct", 5.0))
    max_gap_pct = float(gap.get("max_gap_pct", 20.0))
    min_volume_pace_ratio = float(gap.get("min_volume_pace_ratio", 1.5))
    min_avg_volume_30d = float(gap.get("min_avg_volume_30d", 100_000.0))
    max_candidates = max(int(gap.get("max_scan_candidates", 10)), 1)
    lookback_days = max(int(gap.get("lookback_volume_days", 30)), 5)
    market_cap_source = str(gap.get("market_cap_source", "yfinance")).lower()
    yfinance_delay = max(float(opts.get("yfinance_delay_sec", 2.0)), 0.0)
    min_price = float(uni.get("min_price", 5.0))
    max_price = float(uni.get("max_price", 100.0))
    min_market_cap_m = float(uni.get("min_market_cap_m", 300.0))
    market_cap_filter_enabled = market_cap_source == "yfinance" and min_market_cap_m > 0

    # --- universe ---
    if symbols:
        universe_syms = sorted(set(s.strip().upper() for s in symbols if s.strip()))
    else:
        seed_path = reference_seed_symbols_path(cfg)
        seed_df = read_parquet(seed_path)
        universe_syms = sorted(set(seed_df["symbol"].dropna().astype(str).str.strip().str.upper().tolist()))

    universe_count = len(universe_syms)
    # --- OpenD check ---
    if not check_opend(opend_host, opend_port):
        raise ConnectionError(f"OpenD not reachable at {opend_host}:{opend_port}")

    # --- Alpaca daily stats ---
    _log(logger, "moomoo_932_scan_step", step="fetch_daily_stats", count=universe_count)
    daily_stats = _fetch_daily_stats(alpaca_client, cfg, universe_syms, trade_date, lookback_days)
    daily_stats_count = len(daily_stats)

    # daily stats がある銘柄だけ moomoo 問い合わせ
    syms_with_daily = [s for s in universe_syms if s in daily_stats]
    codes_mm_filtered = [f"US.{s}" for s in syms_with_daily]

    quote_ctx = ft.OpenQuoteContext(host=opend_host, port=opend_port)
    try:
        # --- moomoo snapshot ---
        _log(logger, "moomoo_932_scan_step", step="fetch_snapshot", count=len(syms_with_daily))
        snapshots = _fetch_moomoo_snapshots(quote_ctx, codes_mm_filtered)
        snapshot_ok_count = len(snapshots)
        missing_snapshot_count = len(syms_with_daily) - snapshot_ok_count

        # --- moomoo 9:31 K_1M ---
        _log(logger, "moomoo_932_scan_step", step="fetch_kline_0931", count=len(syms_with_daily))
        kline_0931 = _fetch_moomoo_0931_bars(quote_ctx, codes_mm_filtered, trade_date)
        kline_0931_ok_count = len(kline_0931)
        missing_kline_count = len(syms_with_daily) - kline_0931_ok_count
    finally:
        quote_ctx.close()

    # --- filter pipeline ---
    price_filter_drop = 0
    gap_filter_drop = 0
    volume_filter_drop = 0

    raw_candidates: list[dict] = []

    for sym in syms_with_daily:
        dstat = daily_stats.get(sym)
        snap = snapshots.get(sym)
        bar = kline_0931.get(sym)

        if dstat is None or snap is None:
            continue

        prev_close, avg_vol = dstat
        if prev_close <= 0 or avg_vol <= 0 or avg_vol < min_avg_volume_30d:
            continue

        open_price_raw = snap.get("open_price")
        if open_price_raw is None:
            continue
        try:
            open_price = float(open_price_raw)
        except (TypeError, ValueError):
            continue

        if open_price < min_price or open_price > max_price:
            price_filter_drop += 1
            continue

        gap_pct = (open_price - prev_close) / prev_close * 100.0
        if gap_pct < min_gap_pct or gap_pct > max_gap_pct:
            gap_filter_drop += 1
            continue

        # 9:31 bar volume
        bar_vol = 0.0
        if bar is not None:
            try:
                bar_vol = float(bar.get("volume", 0) or 0)
            except (TypeError, ValueError):
                bar_vol = 0.0

        # volume_ratio: raw bar / avg (現行 pace_ratio と非互換、参考値)
        volume_ratio = bar_vol / avg_vol if avg_vol > 0 else 0.0
        # 簡易換算: 1分足 → 日次換算 (390分/日)
        volume_pace_annualized = bar_vol * 390.0 / avg_vol if avg_vol > 0 else 0.0

        # volume フィルタ: 換算 pace で現行閾値と比較
        if volume_pace_annualized < min_volume_pace_ratio:
            volume_filter_drop += 1
            continue

        update_time = str(snap.get("update_time", ""))
        raw_candidates.append(
            {
                "symbol": sym,
                "open_price": open_price,
                "prev_close_price": prev_close,
                "gap_pct": gap_pct,
                "bar_0931_volume": bar_vol,
                "avg_volume_30d": avg_vol,
                "volume_ratio": volume_ratio,
                "volume_pace_annualized": volume_pace_annualized,
                "snapshot_update_time": update_time,
            }
        )

    # --- market cap ---
    market_cap_drop = 0
    candidates: List[MoomooGapCandidate] = []

    if raw_candidates:
        syms_for_cap = [r["symbol"] for r in raw_candidates]
        mcaps = _fetch_market_caps_m(syms_for_cap, yfinance_delay, logger) if market_cap_filter_enabled else {}

        for r in raw_candidates:
            sym = r["symbol"]
            mc_m = mcaps.get(sym, -1.0)
            if market_cap_filter_enabled:
                if mc_m < 0 or mc_m < min_market_cap_m:
                    market_cap_drop += 1
                    continue
            candidates.append(
                MoomooGapCandidate(
                    symbol=sym,
                    open_price=r["open_price"],
                    prev_close_price=r["prev_close_price"],
                    gap_pct=r["gap_pct"],
                    bar_0931_volume=r["bar_0931_volume"],
                    avg_volume_30d=r["avg_volume_30d"],
                    volume_ratio=r["volume_ratio"],
                    volume_pace_annualized=r["volume_pace_annualized"],
                    market_cap_m=mc_m,
                    snapshot_update_time=r["snapshot_update_time"],
                )
            )

    candidates.sort(key=lambda c: (c.volume_pace_annualized, c.gap_pct), reverse=True)
    limited = candidates[:max_candidates]

    diag = MoomooScanDiagnostics(
        trade_date=trade_date.isoformat(),
        universe_count=universe_count,
        snapshot_ok_count=snapshot_ok_count,
        kline_0931_ok_count=kline_0931_ok_count,
        missing_snapshot_count=missing_snapshot_count,
        missing_kline_count=missing_kline_count,
        daily_stats_count=daily_stats_count,
        price_filter_drop_count=price_filter_drop,
        gap_filter_drop_count=gap_filter_drop,
        volume_filter_drop_count=volume_filter_drop,
        market_cap_drop_count=market_cap_drop,
        final_candidate_count=len(limited),
    )

    _log(logger, "moomoo_932_scanner_diagnostics", **diag.to_dict())
    return limited, diag


# ---------------------------------------------------------------------------
# internal logger helper
# ---------------------------------------------------------------------------


def _log(logger: Optional[logging.Logger], event: str, **fields: Any) -> None:
    if logger is None:
        return
    try:
        from mlstock.logging.logger import log_event

        log_event(logger, event, **fields)
    except Exception:
        pass
