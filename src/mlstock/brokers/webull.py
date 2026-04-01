from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mlstock.brokers.base import BrokerOrderResult, BrokerPosition, OrderBroker


@dataclass(frozen=True)
class WebullCredentials:
    app_key: str
    app_secret: str
    region: str
    base_url: str
    account_tax_type: str
    access_token: Optional[str]
    token_dir: Optional[str]
    paper_trading: bool


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_webull_credentials() -> WebullCredentials:
    load_dotenv(override=False)
    app_key = os.getenv("WEBULL_APP_KEY", "").strip()
    app_secret = os.getenv("WEBULL_APP_SECRET", "").strip()
    region = os.getenv("WEBULL_REGION", "jp").strip().lower() or "jp"
    base_url = os.getenv("WEBULL_BASE_URL", "https://api.webull.co.jp").strip()
    access_token = os.getenv("WEBULL_ACCESS_TOKEN")
    token_dir = os.getenv("WEBULL_OPENAPI_TOKEN_DIR")
    account_tax_type = os.getenv("WEBULL_ACCOUNT_TAX_TYPE", "SPECIFIC").strip().upper() or "SPECIFIC"
    paper_raw = os.getenv("WEBULL_PAPER_TRADING", "0").strip().lower()
    if not app_key or not app_secret:
        raise EnvironmentError("Webull API credentials not found in environment")
    return WebullCredentials(
        app_key=app_key,
        app_secret=app_secret,
        region=region,
        base_url=base_url,
        account_tax_type=account_tax_type,
        access_token=access_token.strip() if isinstance(access_token, str) and access_token.strip() else None,
        token_dir=token_dir.strip() if isinstance(token_dir, str) and token_dir.strip() else None,
        paper_trading=paper_raw in {"1", "true", "yes", "on"},
    )


class WebullOrderBroker(OrderBroker):
    def __init__(
        self,
        base_url: str,
        app_key: str,
        app_secret: str,
        region: str = "jp",
        account_tax_type: str = "SPECIFIC",
        access_token: Optional[str] = None,
        token_dir: Optional[str] = None,
        paper_trading: bool = False,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.app_key = app_key
        self.app_secret = app_secret
        self.region = region.strip().lower() or "jp"
        self.account_tax_type = account_tax_type.strip().upper() or "SPECIFIC"
        self.access_token = access_token
        self.token_dir = token_dir
        self.paper_trading = paper_trading
        self.timeout = timeout
        self._api_client = None
        self._api = None
        self._account_id: Optional[str] = None
        self._instrument_cache: Dict[str, Dict[str, str]] = {}

    @classmethod
    def from_env(cls, base_url: Optional[str] = None) -> "WebullOrderBroker":
        creds = load_webull_credentials()
        return cls(
            base_url=base_url or creds.base_url,
            app_key=creds.app_key,
            app_secret=creds.app_secret,
            region=creds.region,
            account_tax_type=creds.account_tax_type,
            access_token=creds.access_token,
            token_dir=creds.token_dir,
            paper_trading=creds.paper_trading,
        )

    def get_buying_power(self) -> float:
        account_id = self._get_account_id()
        payload = self._response_json(self._api.account.get_account_balance(account_id, "USD"))
        for row in self._as_list(payload.get("account_currency_assets")):
            currency = str(row.get("currency", "")).upper()
            if currency != "USD":
                continue
            for key in ("stock_power", "available_to_withdraw", "settled_cash", "total_cash"):
                value = _to_float(row.get(key), 0.0)
                if value > 0:
                    return value
        return 0.0

    def list_positions(self) -> List[BrokerPosition]:
        account_id = self._get_account_id()
        response = self._api.account.get_account_position(account_id, page_size=100)
        payload = self._response_json(response)
        items: List[BrokerPosition] = []
        for row in self._as_list(payload.get("holdings")):
            symbol = str(row.get("symbol", "")).strip().upper()
            qty = _to_int(row.get("qty"), 0)
            avg_entry_price = _to_float(row.get("unit_cost"), 0.0)
            if symbol and qty > 0:
                items.append(BrokerPosition(symbol=symbol, qty=qty, avg_entry_price=avg_entry_price))
        return items

    def submit_market_buy(self, symbol: str, qty: int) -> BrokerOrderResult:
        return self._place_market_order(symbol=symbol, qty=qty, side="BUY")

    def submit_market_sell(self, symbol: str, qty: int) -> BrokerOrderResult:
        return self._place_market_order(symbol=symbol, qty=qty, side="SELL")

    def close_position(self, symbol: str) -> BrokerOrderResult:
        normalized = str(symbol).strip().upper()
        if not normalized:
            raise ValueError("symbol is required")
        for position in self.list_positions():
            if position.symbol == normalized and position.qty > 0:
                return self.submit_market_sell(normalized, position.qty)
        raise RuntimeError(f"Webull position not found: {normalized}")

    def _ensure_api(self) -> None:
        if self._api is not None:
            return

        from webullsdkcore.client import ApiClient
        from webullsdktrade.api import API

        endpoint = self.base_url.replace("https://", "").replace("http://", "")
        api_client = ApiClient(
            self.app_key,
            self.app_secret,
            self.region,
            verify=True,
            timeout=self.timeout,
            connect_timeout=min(self.timeout, 10),
        )
        api_client.add_endpoint(self.region, endpoint)
        self._api_client = api_client
        self._api = API(api_client)

    def _get_account_id(self) -> str:
        self._ensure_api()
        if self._account_id:
            return self._account_id
        payload = self._response_json(self._api.account.get_app_subscriptions())
        account_id = self._extract_account_id(payload)
        if not account_id:
            raise RuntimeError("Webull account_id not found in get_app_subscriptions response")
        self._account_id = account_id
        return account_id

    def _extract_account_id(self, payload: Any) -> Optional[str]:
        for row in self._as_list(payload):
            if not isinstance(row, dict):
                continue
            account_id = str(row.get("account_id", "")).strip()
            if account_id:
                return account_id
        if isinstance(payload, dict):
            for key in ("data", "items", "subscriptions", "results"):
                nested = payload.get(key)
                account_id = self._extract_account_id(nested)
                if account_id:
                    return account_id
        return None

    def _lookup_instrument(self, symbol: str) -> Dict[str, str]:
        normalized = str(symbol).strip().upper()
        if not normalized:
            raise ValueError("symbol is required")
        cached = self._instrument_cache.get(normalized)
        if cached is not None:
            return cached

        self._ensure_api()
        response = self._api.instrument.get_instrument(normalized, "US_STOCK")
        payload = self._response_json(response)
        info = self._extract_instrument_info(payload, normalized)
        if info is None:
            raise RuntimeError(f"Webull instrument info not found for symbol: {normalized}")
        self._instrument_cache[normalized] = info
        return info

    def _extract_instrument_info(self, payload: Any, symbol: str) -> Optional[Dict[str, str]]:
        for row in self._as_list(payload):
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol", "")).strip().upper()
            instrument_id = str(row.get("instrument_id", "")).strip()
            if row_symbol == symbol and instrument_id:
                return {
                    "symbol": row_symbol,
                    "instrument_id": instrument_id,
                    "instrument_type": "EQUITY",
                    "market": "US",
                }
        if isinstance(payload, dict):
            for key in ("data", "items", "results", "list"):
                nested = payload.get(key)
                info = self._extract_instrument_info(nested, symbol)
                if info is not None:
                    return info
        return None

    def _place_market_order(self, symbol: str, qty: int, side: str) -> BrokerOrderResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        account_id = self._get_account_id()
        instrument = self._lookup_instrument(symbol)
        client_order_id = self._client_order_id(symbol, side)
        new_orders = {
            "client_order_id": client_order_id,
            "symbol": instrument["symbol"],
            "instrument_id": instrument["instrument_id"],
            "instrument_type": instrument["instrument_type"],
            "market": instrument["market"],
            "order_type": "MARKET",
            "quantity": str(int(qty)),
            "support_trading_session": "N",
            "side": side,
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "account_tax_type": self.account_tax_type,
        }
        response = self._api.order_v2.place_order(account_id=account_id, new_orders=new_orders)
        payload = self._response_json(response)
        order_payload = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        return BrokerOrderResult(
            order_id=str(order_payload.get("client_order_id") or client_order_id),
            symbol=instrument["symbol"],
            qty=int(qty),
            side=side.lower(),
            filled_avg_price=_to_float(order_payload.get("filled_avg_price"), 0.0),
            raw=payload if isinstance(payload, dict) else {"payload": payload},
        )

    def _client_order_id(self, symbol: str, side: str) -> str:
        return f"gap-{side.lower()}-{symbol.lower()}-{uuid.uuid4().hex[:20]}"

    def _response_json(self, response: Any) -> Any:
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            body_text = ""
            try:
                body_text = response.text
            except Exception:
                body_text = str(response)
            raise RuntimeError(f"Webull API error status={status_code} body={body_text}")
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError("Webull response json parse failed") from exc

    def _as_list(self, payload: Any) -> List[Any]:
        if isinstance(payload, list):
            return payload
        return []
