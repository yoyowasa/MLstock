from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: int
    avg_entry_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerOrderResult:
    order_id: str
    symbol: str
    qty: int
    side: str
    filled_avg_price: float
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderBroker(ABC):
    @abstractmethod
    def get_buying_power(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def list_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError

    @abstractmethod
    def submit_market_buy(self, symbol: str, qty: int) -> BrokerOrderResult:
        raise NotImplementedError

    @abstractmethod
    def submit_market_sell(self, symbol: str, qty: int) -> BrokerOrderResult:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str) -> BrokerOrderResult:
        raise NotImplementedError
