from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class TwelveDataCredentials:
    api_key: str


def load_twelvedata_api_key() -> TwelveDataCredentials:
    load_dotenv(override=False)
    api_key = os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise EnvironmentError("Twelve Data API key not found in environment")
    return TwelveDataCredentials(api_key=api_key)


class TwelveDataClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.twelvedata.com",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()

    @classmethod
    def from_env(cls, base_url: Optional[str] = None) -> "TwelveDataClient":
        creds = load_twelvedata_api_key()
        return cls(api_key=creds.api_key, base_url=base_url or os.getenv("TWELVEDATA_BASE_URL", "https://api.twelvedata.com"))

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = dict(params or {})
        payload["apikey"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = RuntimeError(f"Transient Twelve Data error: {response.status_code}")
                elif not response.ok:
                    raise RuntimeError(f"Twelve Data error {response.status_code}: {response.text}")
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise RuntimeError("Twelve Data returned non-JSON response") from exc
                    if isinstance(data, dict) and str(data.get("status", "")).lower() == "error":
                        raise RuntimeError(f"Twelve Data API error: {data.get('code')} {data.get('message')}")
                    return data if isinstance(data, dict) else {"data": data}

            if attempt < self.max_retries:
                time.sleep(self.backoff_seconds * (2**attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Twelve Data request failed without error details")

    def get_time_series(
        self,
        symbol: str,
        interval: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outputsize: Optional[int] = None,
        timezone: Optional[str] = None,
        prepost: bool = False,
        previous_close: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "format": "JSON",
        }
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if outputsize is not None:
            params["outputsize"] = outputsize
        if timezone is not None:
            params["timezone"] = timezone
        if prepost:
            params["prepost"] = "true"
        if previous_close:
            params["previous_close"] = "true"
        return self._request("/time_series", params=params)
