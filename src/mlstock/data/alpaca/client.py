from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from mlstock.data.alpaca import endpoints


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    api_secret: str


def load_alpaca_credentials() -> AlpacaCredentials:
    load_dotenv(override=False)
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise EnvironmentError("Alpaca API keys not found in environment")
    return AlpacaCredentials(api_key=api_key, api_secret=api_secret)


class AlpacaClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            }
        )

    @classmethod
    def from_env(cls, base_url: str) -> "AlpacaClient":
        creds = load_alpaca_credentials()
        return cls(base_url=base_url, api_key=creds.api_key, api_secret=creds.api_secret)

    def _retry_after_seconds(self, header_value: Optional[str]) -> Optional[float]:
        if not header_value:
            return None
        text = str(header_value).strip()
        if not text:
            return None
        try:
            seconds = float(text)
            return max(0.0, seconds)
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        retry_after_seconds: Optional[float] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                retry_after_seconds = None
            else:
                if response.status_code in (429,) or 500 <= response.status_code < 600:
                    last_error = RuntimeError(f"Transient Alpaca error: {response.status_code}")
                    if response.status_code == 429:
                        retry_after_seconds = self._retry_after_seconds(response.headers.get("Retry-After"))
                    else:
                        retry_after_seconds = None
                elif not response.ok:
                    raise RuntimeError(f"Alpaca error {response.status_code}: {response.text}")
                else:
                    if response.status_code == 204:
                        return None
                    try:
                        return response.json()
                    except ValueError:
                        if not response.text:
                            return None
                        return response.text

            if attempt < self.max_retries:
                sleep_for = self.backoff_seconds * (2**attempt)
                if retry_after_seconds is not None:
                    sleep_for = max(sleep_for, retry_after_seconds)
                time.sleep(sleep_for)

        if last_error:
            raise last_error
        raise RuntimeError("Alpaca request failed without error details")

    def get_assets(
        self,
        status: str,
        asset_class: str,
        exchange: str,
    ) -> Any:
        params = {
            "status": status,
            "asset_class": asset_class,
            "exchange": exchange,
        }
        return self._request("GET", endpoints.ASSETS, params=params)

    def get_asset(self, symbol: str) -> Any:
        return self._request("GET", f"{endpoints.ASSETS}/{symbol}")

    def get_calendar(self, start: str, end: str) -> Any:
        params = {
            "start": start,
            "end": end,
        }
        return self._request("GET", endpoints.CALENDAR, params=params)

    def get_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str,
        feed: str,
        adjustment: str,
        asof: Optional[str] = None,
        limit: int = 10000,
        page_token: Optional[str] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "limit": limit,
        }
        if asof is not None:
            params["asof"] = asof
        if page_token:
            params["page_token"] = page_token
        return self._request("GET", endpoints.BARS, params=params)

    def get_corporate_actions(
        self,
        symbols: list[str],
        types: str,
        start: str,
        end: str,
        limit: int = 1000,
        page_token: Optional[str] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "symbols": ",".join(symbols),
            "types": types,
            "start": start,
            "end": end,
            "limit": limit,
        }
        if page_token:
            params["page_token"] = page_token
        return self._request("GET", endpoints.CORPORATE_ACTIONS, params=params)

    def get_clock(self) -> Any:
        return self._request("GET", endpoints.CLOCK)

    def get_account(self) -> Any:
        return self._request("GET", endpoints.ACCOUNT)

    def list_positions(self) -> Any:
        return self._request("GET", endpoints.POSITIONS)

    def close_position(self, symbol: str) -> Any:
        return self._request("DELETE", f"{endpoints.POSITIONS}/{symbol}")

    def close_all_positions(self) -> Any:
        return self._request("DELETE", endpoints.POSITIONS)

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Any:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return self._request("POST", endpoints.ORDERS, json_body=payload)
