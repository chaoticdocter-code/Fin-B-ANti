"""Broker-neutral order types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from finb.sim.constraints import AssetClass, Side


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: float | None = None
    notional: float | None = None
    """Dollar-denominated order. The only way a $500 account can hold more than
    a couple of positions — most equities cost more per share than the account
    can spare."""

    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce | None = None
    """Left unset by default and resolved from `asset_class`.

    Crypto and equities do not accept the same values: Alpaca rejects `day` on
    crypto with `invalid crypto time_in_force`, and crypto markets have no
    trading day for a day-order to expire at. Defaulting to a single value meant
    every crypto order was rejected at the venue — so the default is now derived
    rather than assumed.
    """

    asset_class: AssetClass = AssetClass.CRYPTO
    client_id: str | None = None

    def __post_init__(self) -> None:
        if (self.qty is None) == (self.notional is None):
            raise ValueError("specify exactly one of qty or notional")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require a limit_price")

        if self.time_in_force is None:
            resolved = (
                TimeInForce.GTC if self.asset_class is AssetClass.CRYPTO else TimeInForce.DAY
            )
            object.__setattr__(self, "time_in_force", resolved)
        elif (
            self.asset_class is AssetClass.CRYPTO
            and self.time_in_force is TimeInForce.DAY
        ):
            # Fail here rather than at the venue, where the message is a numeric
            # error code and the order has already made a round trip.
            raise ValueError(
                "crypto orders cannot use time_in_force=day — use gtc or ioc"
            )


@dataclass(frozen=True, slots=True)
class OrderResult:
    accepted: bool
    order_id: str | None = None
    symbol: str = ""
    submitted_qty: float = 0.0
    submitted_notional: float = 0.0
    """Dollar value actually committed, after risk shrank the order.

    Reported separately from the requested size because the two routinely differ
    and only this one is true. A caller that logs the request instead of the
    result will show a $250 order that was in fact $0.50."""

    reason: str = ""
    raw: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    positions: list[BrokerPosition]
    is_paper: bool
    taken_at: datetime

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions)


@runtime_checkable
class Broker(Protocol):
    def account(self) -> AccountSnapshot: ...
    def submit(self, order: OrderRequest) -> OrderResult: ...
    def cancel_all(self) -> int: ...
