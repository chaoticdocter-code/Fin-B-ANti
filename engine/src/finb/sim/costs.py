"""Trading costs, and what they do to a $500 account.

At institutional size the dominant cost is market impact. At $500 it is
effectively zero — a $50 order is a rounding error in any liquid book, so the
square-root impact law contributes fractions of a basis point and is ignored
here.

What actually costs money at this size:

1. **The spread.** Crossing it costs half the quoted spread per side. On liquid
   crypto majors that is a few bps; on a thin altcoin or a small-cap equity it
   can be 50bps or more, which no strategy overcomes.
2. **Commission.** Zero on US equities at Alpaca. *Not* zero on crypto anywhere.
3. **Regulatory fees.** Small, sell-side only, and easy to forget. Included
   because "commission-free" is not the same as "free".

The number that matters is `round_trip_bps`: how far price must move in your
favour before you have broken even. Compare it against the volatility of your
horizon. If a round trip costs 50bps and your edge is a 20bps move, no amount of
model quality saves you — and that comparison should be made *before* building
the model, not after.

> The venue fee figures below are defaults pending verification against current
> published schedules. Treat them as conservative placeholders, and see
> [[Open Questions]].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from finb.sim.constraints import AssetClass, Side

# US regulatory fees on equity SALES only. Both are periodically revised by the
# SEC and FINRA; these are order-of-magnitude correct and immaterial at $500,
# but they are modelled so nothing is silently free.
SEC_FEE_PER_DOLLAR = 27.80 / 1_000_000  # SEC Section 31, per dollar of proceeds
FINRA_TAF_PER_SHARE = 0.000166
FINRA_TAF_CAP = 8.30


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    commission: float
    spread: float
    slippage: float
    regulatory: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage + self.regulatory

    def bps_of(self, notional: float) -> float:
        return 1e4 * self.total / notional if notional else 0.0

    def __str__(self) -> str:
        return (
            f"${self.total:.4f} "
            f"(commission ${self.commission:.4f}, spread ${self.spread:.4f}, "
            f"slippage ${self.slippage:.4f}, regulatory ${self.regulatory:.4f})"
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-venue cost assumptions, in basis points unless stated."""

    name: str
    asset: AssetClass

    commission_bps: float = 0.0
    half_spread_bps: float = 1.0
    """Half the quoted spread. A marketable order pays this; a passive one may
    earn it, which is why `passive=True` flips the sign."""

    slippage_bps: float = 0.5
    """Adverse move between decision and fill. Latency, queue position, and the
    fact that the book moves away from you more often than toward you."""

    min_commission: float = 0.0
    us_regulatory_fees: bool = False
    penny_rounded_fees: bool = True
    """Regulatory fees round up to the cent, which is a real cost floor on
    small clips. Only meaningful alongside `us_regulatory_fees`."""

    adverse_selection_bps: float | None = None
    """Cost charged to *filled* passive orders. Defaults to `half_spread_bps`.

    Without this the model says a resting limit order earns the spread for
    nothing, and an evolving strategy will immediately discover that infinite
    free money and optimise straight into it. In reality your passive order
    fills preferentially when the market is moving against you — that is what
    adverse selection means, and in a competitive book it roughly cancels the
    spread you captured. Setting this equal to the half-spread makes passive
    trading approximately free rather than profitable, which is the honest
    baseline. Alpaca's paper engine models none of this."""

    passive_fill_probability: float = 0.5
    """Fraction of resting orders that fill at all. The other half either miss
    the trade entirely or chase — neither is free."""

    def fill_cost(
        self,
        notional: float,
        *,
        side: Side = Side.BUY,
        qty: float = 0.0,
        passive: bool = False,
    ) -> CostBreakdown:
        """Cost of one fill, in dollars."""
        notional = abs(notional)

        commission = max(notional * self.commission_bps / 1e4, self.min_commission)

        # A resting limit order that gets hit earns the spread instead of paying
        # it — but only when it fills, and it fills when it least wants to.
        spread = notional * self.half_spread_bps / 1e4
        if passive:
            adverse = (
                self.half_spread_bps
                if self.adverse_selection_bps is None
                else self.adverse_selection_bps
            )
            spread = -spread + notional * adverse / 1e4

        slippage = 0.0 if passive else notional * self.slippage_bps / 1e4

        regulatory = 0.0
        if self.us_regulatory_fees and side is Side.SELL:
            regulatory = notional * SEC_FEE_PER_DOLLAR
            regulatory += min(qty * FINRA_TAF_PER_SHARE, FINRA_TAF_CAP)
            if self.penny_rounded_fees and regulatory > 0:
                # Regulatory fees are assessed rounded up to the cent. On a
                # $100 clip that turns a $0.0028 fee into $0.01 — 1bp instead of
                # 0.28bp. Small clips pay several times the headline rate, which
                # is precisely the regime a $500 account lives in.
                regulatory = math.ceil(regulatory * 100.0) / 100.0

        return CostBreakdown(commission, spread, slippage, regulatory)

    def round_trip_bps(self, notional: float = 500.0, *, passive: bool = False) -> float:
        """Total cost of buying and selling, in basis points of notional.

        This is the breakeven hurdle: price must move at least this far in your
        favour before the trade makes a cent.
        """
        buy = self.fill_cost(notional, side=Side.BUY, passive=passive)
        sell = self.fill_cost(notional, side=Side.SELL, qty=notional / 100.0, passive=passive)
        return 1e4 * (buy.total + sell.total) / notional

    def with_spread(self, half_spread_bps: float) -> CostModel:
        """A copy at a different spread — for stress-testing a thin book."""
        return replace(self, half_spread_bps=half_spread_bps)


# --------------------------------------------------------------------------- #
#  Venue presets
# --------------------------------------------------------------------------- #

ALPACA_EQUITY = CostModel(
    name="alpaca-equity",
    asset=AssetClass.EQUITY,
    commission_bps=0.0,          # commission-free
    half_spread_bps=1.0,         # ~1bp on liquid large caps; far worse on small caps
    slippage_bps=0.5,
    us_regulatory_fees=True,
)

ALPACA_EQUITY_BROAD = CostModel(
    name="alpaca-equity-broad",
    asset=AssetClass.EQUITY,
    commission_bps=0.0,
    half_spread_bps=5.0,   # realistic once the universe leaves the megacaps
    slippage_bps=1.5,
    us_regulatory_fees=True,
)
"""Use this, not `ALPACA_EQUITY`, for any universe wider than the top ~100 names.
The 1bp spread on megacaps is not representative and quietly flatters every
backtest run on a broad cross-section."""

ALPACA_CRYPTO = CostModel(
    name="alpaca-crypto",
    asset=AssetClass.CRYPTO,
    commission_bps=25.0,         # ~0.25% taker at the entry volume tier
    half_spread_bps=6.0,         # measured: BTC 11.9bps / ETH 11.6bps full spread
    slippage_bps=1.0,
)
"""Majors only — BTC and ETH. The 2.5bps half-spread this used to carry was an
assumption; measuring the live book gave 11.9bps full on BTC, and 39.9bps median
across the wider universe. See `ALPACA_CRYPTO_ALTS`."""

ALPACA_CRYPTO_ALTS = CostModel(
    name="alpaca-crypto-alts",
    asset=AssetClass.CRYPTO,
    commission_bps=25.0,
    half_spread_bps=20.0,        # measured median 39.9bps full across 22 pairs
    slippage_bps=2.0,
)
"""Anything outside the top handful. Measured spreads on this venue: DOGE 28bps,
SOL 40bps, LTC 54bps, AVAX 58bps, PAXG 122bps. Most of these also fail the
liquidity screen on trade staleness, so this model describes what it would cost
*if* you could trade them."""

BINANCE_SPOT = CostModel(
    name="binance-spot",
    asset=AssetClass.CRYPTO,
    commission_bps=10.0,         # 0.10% spot taker, standard tier
    half_spread_bps=1.0,
    slippage_bps=1.0,
)

COINBASE_ADVANCED = CostModel(
    name="coinbase-advanced",
    asset=AssetClass.CRYPTO,
    commission_bps=60.0,         # entry-tier taker; drops steeply with volume
    half_spread_bps=1.5,
    slippage_bps=1.0,
)

KRAKEN_SPOT = CostModel(
    name="kraken-spot",
    asset=AssetClass.CRYPTO,
    commission_bps=26.0,         # 0.26% entry-tier taker
    half_spread_bps=1.5,
    slippage_bps=1.0,
)

VENUES: dict[str, CostModel] = {
    m.name: m
    for m in (
        ALPACA_EQUITY,
        ALPACA_EQUITY_BROAD,
        ALPACA_CRYPTO,
        ALPACA_CRYPTO_ALTS,
        BINANCE_SPOT,
        COINBASE_ADVANCED,
        KRAKEN_SPOT,
    )
}


def breakeven_table(notional: float = 500.0) -> list[tuple[str, float, float]]:
    """(venue, taker round-trip bps, maker round-trip bps) for every preset.

    Print this before designing a strategy. It is the cheapest reality check
    available, and it rules out whole horizons in one glance.
    """
    return [
        (m.name, m.round_trip_bps(notional), m.round_trip_bps(notional, passive=True))
        for m in VENUES.values()
    ]


def min_hold_days(
    cost_bps: float,
    daily_vol: float,
    *,
    ic: float = 0.03,
    selection_z: float = 1.75,
    coverage: float = 2.0,
) -> float:
    """Shortest holding period at which gross edge covers cost `coverage` times.

    The model, which is deliberately crude but has the right shape:

        edge_bps(h) = IC x sigma_h x E[z | selected] where sigma_h = sigma_daily x sqrt(h)

    Set that equal to ``coverage x cost_bps`` and solve for h. Because edge grows
    with √h while cost is fixed per round trip, **holding period is the only free
    variable that improves the cost ratio.** Not model quality, not features.

    Defaults: `ic=0.03` is the realistic cross-sectional information coefficient
    ceiling for a retail system on liquid markets; `selection_z=1.75` is roughly
    E[z] for a top-decile pick. Both are optimistic. Treat the output as a floor.
    """
    if ic <= 0 or daily_vol <= 0:
        return float("inf")
    edge_per_unit_sqrt_h = ic * daily_vol * 1e4 * selection_z
    if edge_per_unit_sqrt_h <= 0:
        return float("inf")
    return (coverage * cost_bps / edge_per_unit_sqrt_h) ** 2


def feasibility(
    venue: CostModel,
    daily_vol: float,
    *,
    notional: float = 500.0,
    coverage: float = 2.0,
    ic: float = 0.03,
) -> dict:
    """Per-venue answer to: how long must we hold, and how many trades is that?

    This is the feasibility map. It rules strategy families in or out before a
    single feature is computed, which is the cheapest possible way to find out
    that something cannot work.
    """
    cost = venue.round_trip_bps(notional)
    breakeven = min_hold_days(cost, daily_vol, ic=ic, coverage=1.0)
    target = min_hold_days(cost, daily_vol, ic=ic, coverage=coverage)
    days_per_year = 365.0 if venue.asset is AssetClass.CRYPTO else 252.0
    per_year = days_per_year / target if target > 0 else float("inf")

    # One position held for `target` days. Breadth multiplies this — 20 symbols
    # traded in parallel gives 20x the observations for the same holding period,
    # which is the entire argument for cross-sectional breadth over frequency.
    years_to_100 = 100.0 / per_year if per_year > 0 else float("inf")

    return {
        "venue": venue.name,
        "round_trip_bps": cost,
        "breakeven_hold_days": breakeven,
        "min_hold_days": target,
        "trades_per_year": per_year,
        "years_to_100_trades": years_to_100,
        # Viability is not just "can it cover costs" — it is "can it ever be
        # validated". A strategy needing 11 years to reach 100 observations
        # cannot clear the luck hurdle within a human timeframe.
        "viable": target < 60 and per_year >= 25,
    }


def edge_required(cost_bps: float, trades_per_day: int, target_annual_return: float = 0.20) -> float:
    """Average per-trade edge (bps) needed to net `target_annual_return`.

    Assumes ~252 trading days for equities. For crypto, pass a larger
    `trades_per_day` — the market runs continuously, and so does the fee drag.
    """
    trades_per_year = trades_per_day * 252
    if trades_per_year <= 0:
        return float("inf")
    gross_needed_bps = 1e4 * target_annual_return / trades_per_year
    return gross_needed_bps + cost_bps
