"""Is this symbol actually tradeable right now?

Daily bars exist for every listed pair. Flow does not. Measured on Alpaca's
crypto venue on a Sunday morning:

| Symbol | Last trade | Spread |
|---|---|---|
| BTC/USD | 190s | 4.6 bps |
| ETH/USD | 519s | 11.6 bps |
| AAVE/USD | 91 min | 16.1 bps |
| SHIB/USD | 4 hours | 40.1 bps |
| CRV/USD | **40 hours** | 23.0 bps |

A backtest on daily closes will happily trade CRV, because a daily bar exists
for every day. The venue had no trade in it for nearly two days. That gap
between "a price is published" and "you could have transacted" is where small
accounts lose money they never see coming.

Two screens, both cheap:

- **Staleness** — how long since the last print. A symbol nobody has traded in
  hours cannot be entered or exited on demand at anything like the shown price.
- **Spread** — measured live, not assumed. The cost model's default half-spread
  for crypto was 2.5 bps; SHIB quotes at 40 bps wide. That single difference is
  larger than the edge most strategies are chasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from finb.log import get_logger

log = get_logger("liquidity")

MAX_TRADE_AGE_S = 900.0
"""15 minutes. A symbol with no print in that long is not a market you can
rebalance into at the quoted price."""

MAX_SPREAD_BPS = 30.0
"""Full quoted spread. Crossing costs half of this per side, so 30 bps means a
15 bps entry and a 15 bps exit before any fee — already half a day's expected
edge at a realistic information coefficient."""


@dataclass(frozen=True, slots=True)
class Liquidity:
    symbol: str
    last_trade_age_s: float | None
    spread_bps: float | None
    bid: float
    ask: float
    price: float

    @property
    def stale(self) -> bool:
        return self.last_trade_age_s is None or self.last_trade_age_s > MAX_TRADE_AGE_S

    @property
    def wide(self) -> bool:
        return self.spread_bps is None or self.spread_bps > MAX_SPREAD_BPS

    @property
    def tradeable(self) -> bool:
        return not (self.stale or self.wide)

    @property
    def reason(self) -> str:
        if self.last_trade_age_s is None:
            return "no recent trade"
        bits = []
        if self.stale:
            mins = self.last_trade_age_s / 60
            bits.append(f"last trade {mins:.0f} min ago" if mins < 120
                        else f"last trade {mins / 60:.1f} hours ago")
        if self.wide:
            bits.append(f"spread {self.spread_bps:.0f} bps")
        return "; ".join(bits)


def measure(symbols: list[str]) -> dict[str, Liquidity]:
    """Live trade age and quoted spread for each symbol. Read-only."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestQuoteRequest, CryptoLatestTradeRequest

    from finb.watch import to_data_symbol

    wanted = sorted({to_data_symbol(s) for s in symbols})
    if not wanted:
        return {}

    client = CryptoHistoricalDataClient()
    now = datetime.now(UTC)
    trades = client.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=wanted))
    quotes = client.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=wanted))

    out: dict[str, Liquidity] = {}
    for sym in wanted:
        t, q = trades.get(sym), quotes.get(sym)
        age = (now - t.timestamp).total_seconds() if t else None
        bid = float(q.bid_price) if q else 0.0
        ask = float(q.ask_price) if q else 0.0
        mid = (bid + ask) / 2 if bid and ask else 0.0
        spread = 1e4 * (ask - bid) / mid if mid > 0 else None
        out[sym] = Liquidity(
            symbol=sym,
            last_trade_age_s=age,
            spread_bps=spread,
            bid=bid,
            ask=ask,
            price=float(t.price) if t else mid,
        )
    return out


def screen(symbols: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split a universe into (tradeable, {rejected: why}).

    Use this before ranking, not after. Ranking an untradeable symbol and then
    discovering you cannot fill it wastes the slot — and worse, it silently
    changes the strategy from the one that was backtested.
    """
    liq = measure(symbols)
    keep, rejected = [], {}
    for sym, x in liq.items():
        if x.tradeable:
            keep.append(sym)
        else:
            rejected[sym] = x.reason
    if rejected:
        log.info(f"{len(keep)} tradeable, {len(rejected)} screened out")
    return sorted(keep), rejected
