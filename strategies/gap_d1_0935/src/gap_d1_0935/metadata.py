from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable

import yfinance as yf

from mlstock.data.alpaca.client import AlpacaClient


@dataclass(frozen=True)
class SymbolProfile:
    symbol: str
    quote_type: str
    exchange: str
    market_cap: float | None
    market_cap_bucket: str
    sector: str
    security_type: str
    suffix_pattern: str


_SUFFIX_PATTERNS = [
    ("unit", re.compile(r"(?:[./-](?:U|UN))$")),
    ("unit", re.compile(r"^[A-Z]{4,5}U$")),
    ("rights", re.compile(r"(?:[./-](?:R|RT))$")),
    ("rights", re.compile(r"^[A-Z]{4,5}R$")),
    ("warrant", re.compile(r"(?:[./-](?:W|WS|WT))$")),
    ("warrant", re.compile(r"^[A-Z]{4,5}W$")),
    ("preferred", re.compile(r"(?:[./-]PR?[A-Z])$")),
    ("preferred", re.compile(r"(?:[./-]P[A-Z])$")),
]


def classify_suffix_pattern(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    for label, pattern in _SUFFIX_PATTERNS:
        if pattern.search(text):
            return label
    return "common_like"


def normalize_quote_type(raw: object) -> str:
    text = str(raw or "").strip().upper()
    return text or "UNKNOWN"


def market_cap_bucket(market_cap: float | None) -> str:
    if market_cap is None:
        return "unknown"
    if market_cap < 50_000_000:
        return "lt_50m"
    if market_cap < 300_000_000:
        return "50m_to_300m"
    if market_cap < 2_000_000_000:
        return "300m_to_2b"
    if market_cap <= 10_000_000_000:
        return "2b_to_10b"
    return "gt_10b"


def determine_security_type(symbol: str, quote_type: str, exchange: str) -> str:
    suffix_type = classify_suffix_pattern(symbol)
    if suffix_type != "common_like":
        return suffix_type
    qt = normalize_quote_type(quote_type)
    exch = str(exchange or "").upper()
    if qt == "ETF":
        return "etf"
    if qt == "MUTUALFUND":
        return "fund"
    if qt == "ADR":
        return "adr"
    if "OTC" in exch or qt == "OTC EQUITY":
        return "otc"
    if qt in {"EQUITY", "COMMON STOCK", "UNKNOWN"}:
        return "common_stock"
    return qt.lower().replace(" ", "_")


def should_exclude_non_common(security_type: str) -> bool:
    return security_type not in {"common_stock"}


def fetch_symbol_profiles(
    client: AlpacaClient,
    symbols: Iterable[str],
    delay_sec: float = 0.0,
) -> Dict[str, SymbolProfile]:
    profiles: Dict[str, SymbolProfile] = {}
    for symbol in symbols:
        market_cap = None
        sector = ""
        quote_type = "UNKNOWN"
        exchange = ""
        try:
            ticker = yf.Ticker(symbol)
            info = getattr(ticker, "info", {}) or {}
        except Exception:
            info = {}
        quote_type = normalize_quote_type(info.get("quoteType"))
        sector = str(info.get("sector") or "")
        exchange = str(info.get("exchange") or "")
        raw_market_cap = info.get("marketCap")
        if raw_market_cap is not None:
            try:
                market_cap = float(raw_market_cap)
            except (TypeError, ValueError):
                market_cap = None
        if not exchange:
            try:
                asset = client.get_asset(symbol)
                exchange = str((asset or {}).get("exchange") or "")
            except Exception:
                exchange = ""
        security_type = determine_security_type(symbol, quote_type, exchange)
        profiles[symbol] = SymbolProfile(
            symbol=symbol,
            quote_type=quote_type,
            exchange=exchange or "UNKNOWN",
            market_cap=market_cap,
            market_cap_bucket=market_cap_bucket(market_cap),
            sector=sector,
            security_type=security_type,
            suffix_pattern=classify_suffix_pattern(symbol),
        )
        if delay_sec > 0:
            time.sleep(delay_sec)
    return profiles


def summarize_by_key(profiles: Dict[str, SymbolProfile], key: str) -> Dict[str, int]:
    counter = Counter(getattr(profile, key) for profile in profiles.values())
    return dict(counter)
