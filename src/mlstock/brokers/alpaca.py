from __future__ import annotations

from typing import List

from mlstock.brokers.base import BrokerOrderResult, BrokerPosition, OrderBroker
from mlstock.data.alpaca.client import AlpacaClient


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AlpacaOrderBroker(OrderBroker):
    def __init__(self, trading_client: AlpacaClient) -> None:
        self.trading_client = trading_client

    def get_buying_power(self) -> float:
        account = self.trading_client.get_account()
        if isinstance(account, dict):
            for key in ("daytrading_buying_power", "buying_power", "cash"):
                value = _to_float(account.get(key), 0.0)
                if value > 0:
                    return value
        return 0.0

    def list_positions(self) -> List[BrokerPosition]:
        payload = self.trading_client.list_positions()
        if not isinstance(payload, list):
            return []
        items: List[BrokerPosition] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            qty = int(_to_float(row.get("qty"), 0.0))
            avg_entry_price = _to_float(row.get("avg_entry_price"), 0.0)
            items.append(BrokerPosition(symbol=symbol, qty=qty, avg_entry_price=avg_entry_price))
        return items

    def submit_market_buy(self, symbol: str, qty: int) -> BrokerOrderResult:
        payload = self.trading_client.submit_order(symbol=symbol, qty=qty, side="buy", order_type="market", time_in_force="day")
        return self._normalize_order(payload, symbol=symbol, qty=qty, side="buy")

    def submit_market_sell(self, symbol: str, qty: int) -> BrokerOrderResult:
        payload = self.trading_client.submit_order(symbol=symbol, qty=qty, side="sell", order_type="market", time_in_force="day")
        return self._normalize_order(payload, symbol=symbol, qty=qty, side="sell")

    def close_position(self, symbol: str) -> BrokerOrderResult:
        payload = self.trading_client.close_position(symbol)
        qty = int(_to_float(payload.get("qty") if isinstance(payload, dict) else None, 0.0))
        return self._normalize_order(payload, symbol=symbol, qty=qty, side="sell")

    def _normalize_order(self, payload: object, symbol: str, qty: int, side: str) -> BrokerOrderResult:
        raw = payload if isinstance(payload, dict) else {"raw": payload}
        filled_avg_price = _to_float(raw.get("filled_avg_price"), 0.0)
        order_id = str(raw.get("id", f"alpaca-{side}-{symbol}"))
        normalized_qty = int(_to_float(raw.get("qty"), float(qty)))
        return BrokerOrderResult(
            order_id=order_id,
            symbol=symbol,
            qty=normalized_qty,
            side=side,
            filled_avg_price=filled_avg_price,
            raw=raw,
        )
