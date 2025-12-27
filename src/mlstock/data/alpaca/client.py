from __future__ import annotations

import os
import time
from dataclasses import dataclass
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

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code in (429,) or 500 <= response.status_code < 600:
                    last_error = RuntimeError(
                        f"Transient Alpaca error: {response.status_code}"
                    )
                elif not response.ok:
                    raise RuntimeError(
                        f"Alpaca error {response.status_code}: {response.text}"
                    )
                else:
                    return response.json()

            if attempt < self.max_retries:
                sleep_for = self.backoff_seconds * (2 ** attempt)
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

    def get_calendar(self, start: str, end: str) -> Any:
        params = {
            "start": start,
            "end": end,
        }
        return self._request("GET", endpoints.CALENDAR, params=params)
