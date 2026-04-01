from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from mlstock.jobs.gap_scanner import GapCandidate


@dataclass(frozen=True)
class OptionsFilterSettings:
    min_call_volume: float
    min_call_oi_ratio: float
    min_call_put_ratio: float
    yfinance_delay_sec: float
    min_expiry_days: int
    allow_error_fallback: bool


@dataclass(frozen=True)
class OptionsQualifiedCandidate:
    candidate: GapCandidate
    call_volume: float
    call_open_interest: float
    put_volume: float
    call_oi_ratio: float
    call_put_ratio: float
    chosen_expiry: str | None
    expiry_dte: int | None
    used_fallback: bool
    fallback_reason: str | None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload


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


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _load_settings(raw_cfg: Mapping[str, Any]) -> OptionsFilterSettings:
    options = raw_cfg.get("options", {}) if isinstance(raw_cfg.get("options"), dict) else {}
    return OptionsFilterSettings(
        min_call_volume=max(_to_float(options.get("min_call_volume"), 100.0), 0.0),
        min_call_oi_ratio=max(_to_float(options.get("min_call_oi_ratio"), 0.25), 0.0),
        min_call_put_ratio=max(_to_float(options.get("min_call_put_ratio"), 1.8), 0.0),
        yfinance_delay_sec=max(_to_float(options.get("yfinance_delay_sec"), 2.0), 0.0),
        min_expiry_days=max(_to_int(options.get("min_expiry_days"), 1), 0),
        allow_error_fallback=_to_bool(options.get("allow_error_fallback"), False),
    )


def _safe_sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return float(series.sum())


def _select_expiry(expiries: Sequence[str], min_expiry_days: int) -> tuple[str, int]:
    if not expiries:
        raise RuntimeError("expiry_not_found")
    today_ny = datetime.now(ZoneInfo("America/New_York")).date()
    parsed: List[tuple[str, int]] = []
    for item in expiries:
        try:
            expiry_date = datetime.strptime(str(item), "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (expiry_date - today_ny).days
        if dte >= 0:
            parsed.append((str(item), dte))
    if not parsed:
        raise RuntimeError("expiry_parse_failed")
    parsed.sort(key=lambda row: row[1])
    for expiry, dte in parsed:
        if dte >= min_expiry_days:
            return expiry, dte
    raise RuntimeError(f"expiry_under_min_dte:{min_expiry_days}")


def _fetch_option_metrics(symbol: str, min_expiry_days: int) -> tuple[float, float, float, str, int]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = list(getattr(ticker, "options", []) or [])
    chosen_expiry, dte = _select_expiry(expiries, min_expiry_days=min_expiry_days)
    chain = ticker.option_chain(chosen_expiry)
    calls = chain.calls if isinstance(chain.calls, pd.DataFrame) else pd.DataFrame()
    puts = chain.puts if isinstance(chain.puts, pd.DataFrame) else pd.DataFrame()
    call_volume = _safe_sum(calls, "volume")
    call_open_interest = _safe_sum(calls, "openInterest")
    put_volume = _safe_sum(puts, "volume")
    return call_volume, call_open_interest, put_volume, chosen_expiry, dte


def filter_unusual_options_activity(
    candidates: Sequence[GapCandidate],
    gap_cfg: Mapping[str, Any],
    max_candidates: int,
) -> List[OptionsQualifiedCandidate]:
    settings = _load_settings(gap_cfg)
    limit = max(int(max_candidates), 1)
    if not candidates:
        return []

    try:
        import yfinance  # noqa: F401
    except Exception:
        if not settings.allow_error_fallback:
            return []
        return [
            OptionsQualifiedCandidate(
                candidate=item,
                call_volume=0.0,
                call_open_interest=0.0,
                put_volume=0.0,
                call_oi_ratio=0.0,
                call_put_ratio=0.0,
                chosen_expiry=None,
                expiry_dte=None,
                used_fallback=True,
                fallback_reason="yfinance_import_error",
            )
            for item in list(candidates)[:limit]
        ]

    passed: List[OptionsQualifiedCandidate] = []
    fallback: List[OptionsQualifiedCandidate] = []

    for idx, item in enumerate(candidates):
        try:
            call_volume, call_oi, put_volume, chosen_expiry, expiry_dte = _fetch_option_metrics(
                item.symbol,
                min_expiry_days=settings.min_expiry_days,
            )
            call_oi_ratio = (call_volume / call_oi) if call_oi > 0 else 0.0
            call_put_ratio = (call_volume / put_volume) if put_volume > 0 else float("inf")
            qualifies = (
                call_volume >= settings.min_call_volume
                and call_oi_ratio >= settings.min_call_oi_ratio
                and call_put_ratio >= settings.min_call_put_ratio
            )
            if qualifies:
                passed.append(
                    OptionsQualifiedCandidate(
                        candidate=item,
                        call_volume=call_volume,
                        call_open_interest=call_oi,
                        put_volume=put_volume,
                        call_oi_ratio=call_oi_ratio,
                        call_put_ratio=call_put_ratio,
                        chosen_expiry=chosen_expiry,
                        expiry_dte=expiry_dte,
                        used_fallback=False,
                        fallback_reason=None,
                    )
                )
        except Exception as exc:
            if settings.allow_error_fallback:
                fallback.append(
                    OptionsQualifiedCandidate(
                        candidate=item,
                        call_volume=0.0,
                        call_open_interest=0.0,
                        put_volume=0.0,
                        call_oi_ratio=0.0,
                        call_put_ratio=0.0,
                        chosen_expiry=None,
                        expiry_dte=None,
                        used_fallback=True,
                        fallback_reason=str(exc),
                    )
                )
        if settings.yfinance_delay_sec > 0 and idx < len(candidates) - 1:
            time.sleep(settings.yfinance_delay_sec)

    passed.sort(key=lambda row: (row.call_put_ratio, row.call_oi_ratio, row.candidate.gap_pct), reverse=True)
    selected = passed[:limit]
    if settings.allow_error_fallback and len(selected) < limit and fallback:
        selected.extend(fallback[: max(0, limit - len(selected))])
    return selected
