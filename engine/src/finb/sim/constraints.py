"""US account mechanics: settlement, and broker-imposed trading limits.

> [!important] The Pattern Day Trader rule no longer exists.
> FINRA's amendments to Rule 4210 took effect **4 June 2026** (SEC approved
> 14 April 2026, Regulatory Notice 26-10). They eliminate the "pattern day
> trader" definition, the day-trade count, the $25,000 minimum equity
> requirement, and "day trading buying power" — "replaced in their entirety."
> A margin account is now judged on whether it carries an **intraday margin
> deficit** at any point in the day, regardless of whether it day trades.
> Firms may phase this in until **20 October 2027**.

What that changes, and what it does not:

- **Equities, margin account.** No day-trade cap. The old "3 day trades per 5
  business days under $25k" ceiling is gone. Separately and unchanged: a margin
  account still needs $2,000 minimum equity to use margin at all, so a $500
  account is effectively a cash account whatever it is labelled.
- **Equities, cash account.** Unaffected by the amendment. Proceeds still settle
  T+1, and reusing unsettled proceeds then selling before they settle is still a
  good-faith violation — three in twelve months restricts the account to settled
  cash for 90 days. This remains the real constraint at $500.
- **Crypto.** Never had either. Instant settlement, no limits.

**Brokers may still be stricter than FINRA**, especially during the phase-in, so
day trades are still counted here and limits are expressed as a per-broker
`BrokerPolicy` rather than as a regulation. Counting them also remains useful
for cost analysis — see `finb.sim.costs`, where turnover, not regulation, is
what actually constrains a $500 account.

Sources: FINRA Regulatory Notice 26-10 (Rule 4210 amendments, eff. 2026-06-04);
SEC Rule 15c6-1 as amended May 2024 for T+1 settlement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from finb.clock import ET, previous_trading_day, settlement_date

DAY_TRADE_WINDOW_DAYS = 5        # rolling business days, for broker-imposed caps
MARGIN_MINIMUM_EQUITY = 2_000.0  # FINRA 4210(b); separate from the repealed PDT rule
GFV_LIMIT = 3                    # good-faith violations before a 90-day restriction


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class AccountType(StrEnum):
    CASH = "cash"
    MARGIN = "margin"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: Side
    qty: float
    price: float
    ts: datetime
    asset_class: AssetClass = AssetClass.EQUITY

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def trade_date(self) -> date:
        """Calendar date in Eastern time — the date the rules are keyed on."""
        return self.ts.astimezone(ET).date()


@dataclass(frozen=True, slots=True)
class BrokerPolicy:
    """Limits a specific broker imposes, which may exceed what regulation requires.

    FINRA no longer caps day trades, but a broker is free to, and several are
    expected to keep their own limits through the phase-in. Model the broker,
    not the regulator.
    """

    name: str
    max_day_trades_per_5d: int | None = None
    """None means unlimited — the post-2026-06-04 default."""

    min_margin_equity: float = MARGIN_MINIMUM_EQUITY
    min_notional_usd: float = 1.0
    supports_fractional: bool = True


ALPACA = BrokerPolicy(name="alpaca", max_day_trades_per_5d=None, min_notional_usd=1.0)

LEGACY_PDT = BrokerPolicy(name="legacy-pdt", max_day_trades_per_5d=3)
"""The pre-June-2026 regime, kept only so historical backtests can reproduce
the constraints that actually applied at the time. Do not use for live sizing."""


@dataclass(frozen=True, slots=True)
class TradingStatus:
    day_trades_in_window: int
    total_trades_in_window: int
    restricted: bool
    """True only if a broker-imposed cap has been hit."""

    remaining_day_trades: int | None
    """None when the broker imposes no cap."""

    reason: str = ""


class DayTradeMonitor:
    """Counts day trades on a rolling 5-business-day window.

    Since June 2026 this is no longer a regulatory matter, but it stays because
    (a) brokers may impose their own caps and (b) turnover is the single most
    important cost driver at $500. Crypto fills are recorded but never counted.
    """

    def __init__(self) -> None:
        self._fills: list[Fill] = []

    def record(self, fill: Fill) -> None:
        self._fills.append(fill)

    # ------------------------------------------------------------------ #

    def _day_trades_on(self, d: date) -> int:
        """Round trips per symbol on one trading day.

        Counted as min(buys, sells) per symbol: three buys and one sell is one
        day trade, not three.
        """
        by_symbol: dict[str, dict[Side, float]] = defaultdict(
            lambda: {Side.BUY: 0.0, Side.SELL: 0.0}
        )
        for f in self._fills:
            if f.asset_class is not AssetClass.EQUITY or f.trade_date != d:
                continue
            by_symbol[f.symbol][f.side] += f.qty

        total = 0
        for sides in by_symbol.values():
            if sides[Side.BUY] > 0 and sides[Side.SELL] > 0:
                total += 1
        return total

    def _window(self, asof: date) -> list[date]:
        """The 5 business days ending on `asof`, inclusive."""
        days, d = [asof], asof
        for _ in range(DAY_TRADE_WINDOW_DAYS - 1):
            d = previous_trading_day(d)
            days.append(d)
        return days

    def status(
        self,
        asof: date,
        equity: float,
        account: AccountType,
        broker: BrokerPolicy = ALPACA,
    ) -> TradingStatus:
        """Where the account stands against its broker's limits."""
        window = set(self._window(asof))

        day_trades = sum(self._day_trades_on(d) for d in window)
        total = sum(
            1 for f in self._fills
            if f.asset_class is AssetClass.EQUITY and f.trade_date in window
        )

        cap = broker.max_day_trades_per_5d
        if cap is None:
            reason = (
                "no day-trade cap — FINRA's pattern day trader rule was repealed "
                "effective 2026-06-04"
            )
            if account is AccountType.CASH:
                reason += "; T+1 settlement still binds"
            elif equity < broker.min_margin_equity:
                reason += (
                    f"; equity ${equity:,.0f} is below the ${broker.min_margin_equity:,.0f} "
                    "margin minimum, so this trades as a cash account"
                )
            return TradingStatus(day_trades, total, False, None, reason)

        restricted = day_trades >= cap
        return TradingStatus(
            day_trades_in_window=day_trades,
            total_trades_in_window=total,
            restricted=restricted,
            remaining_day_trades=max(0, cap - day_trades),
            reason=(
                f"{broker.name} caps day trades at {cap} per 5 business days"
                if restricted
                else ""
            ),
        )

    def would_be_day_trade(self, symbol: str, side: Side, asof: date) -> bool:
        """Whether closing `symbol` now would complete a same-day round trip."""
        opposite = Side.SELL if side is Side.BUY else Side.BUY
        return any(
            f.symbol == symbol
            and f.side is opposite
            and f.trade_date == asof
            and f.asset_class is AssetClass.EQUITY
            for f in self._fills
        )


@dataclass
class _Lot:
    symbol: str
    qty: float
    bought_on: date
    unsettled_until: date | None
    """If funded with unsettled proceeds, selling before this date is a
    good-faith violation."""


@dataclass
class CashSettlement:
    """T+1 settlement for a cash account, and good-faith violation detection.

    You *may* buy with unsettled proceeds. The violation is selling what you
    bought before the money that paid for it has settled.
    """

    settled_cash: float
    pending: list[tuple[date, float]] = field(default_factory=list)
    lots: dict[str, _Lot] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def advance_to(self, d: date) -> None:
        """Settle everything due on or before `d`."""
        self.settled_cash += sum(amt for sd, amt in self.pending if sd <= d)
        self.pending = [(sd, amt) for sd, amt in self.pending if sd > d]

    @property
    def unsettled_cash(self) -> float:
        return sum(amt for _, amt in self.pending)

    @property
    def buying_power(self) -> float:
        """A cash account can spend unsettled proceeds — with strings attached."""
        return self.settled_cash + self.unsettled_cash

    def buy(self, symbol: str, qty: float, price: float, on: date) -> None:
        cost = qty * price
        if cost > self.buying_power + 1e-9:
            raise ValueError(
                f"insufficient buying power: need ${cost:,.2f}, have ${self.buying_power:,.2f}"
            )

        # Settled cash first; anything beyond it draws on unsettled proceeds and
        # inherits the latest settlement date it touched.
        unsettled_used = max(0.0, cost - self.settled_cash)
        self.settled_cash = max(0.0, self.settled_cash - cost)

        unsettled_until = None
        if unsettled_used > 0:
            remaining = unsettled_used
            for sd, amt in sorted(self.pending):
                unsettled_until = sd
                remaining -= amt
                if remaining <= 0:
                    break

        self.lots[symbol] = _Lot(symbol, qty, on, unsettled_until)

    def sell(self, symbol: str, qty: float, price: float, on: date) -> bool:
        """Sell. Returns True if this created a good-faith violation."""
        proceeds = qty * price
        self.pending.append((settlement_date(on), proceeds))

        lot = self.lots.get(symbol)
        gfv = False
        if lot and lot.unsettled_until is not None and on < lot.unsettled_until:
            gfv = True
            self.violations.append(
                f"good-faith violation: sold {symbol} on {on} but the funds that "
                f"bought it do not settle until {lot.unsettled_until}"
            )
        if lot:
            lot.qty -= qty
            if lot.qty <= 1e-9:
                del self.lots[symbol]
        return gfv

    @property
    def restricted(self) -> bool:
        """Three good-faith violations in 12 months means settled-cash only."""
        return len(self.violations) >= GFV_LIMIT


def round_trip_capacity(
    account: AccountType, asset: AssetClass, broker: BrokerPolicy = ALPACA
) -> str:
    """Plain-language answer to 'how often can $500 actually trade?'

    Note the answer is now almost always "as often as you can afford to", not
    "as often as you are allowed to". Since the PDT repeal the binding
    constraint is cost, not regulation — see `finb.sim.costs`.
    """
    if asset is AssetClass.CRYPTO:
        return "unlimited — instant settlement, 24/7. Cost is the only limit."

    if broker.max_day_trades_per_5d is not None:
        return (
            f"{broker.max_day_trades_per_5d} day trades per rolling 5 business days "
            f"({broker.name} policy; FINRA no longer imposes one)"
        )

    if account is AccountType.CASH:
        return (
            "no day-trade limit, but roughly one full-balance round trip per day: "
            "proceeds settle T+1, and reusing them before settlement risks a "
            f"good-faith violation ({GFV_LIMIT} in 12 months means a 90-day restriction)"
        )

    return (
        "no day-trade limit since the 2026-06-04 PDT repeal; a margin account "
        f"under ${MARGIN_MINIMUM_EQUITY:,.0f} equity trades as cash, so T+1 "
        "settlement is the practical constraint"
    )
