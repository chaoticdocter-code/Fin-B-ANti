"""Cost arithmetic. The assertions encode the conclusions the strategy design
rests on, so if a fee assumption is ever revised these fail loudly."""

from __future__ import annotations

import pytest

from finb.sim.constraints import AssetClass, Side
from finb.sim.costs import (
    ALPACA_CRYPTO,
    ALPACA_EQUITY,
    BINANCE_SPOT,
    COINBASE_ADVANCED,
    VENUES,
    CostModel,
    breakeven_table,
    edge_required,
)


def test_equity_commission_is_zero_but_trading_is_not_free():
    c = ALPACA_EQUITY.fill_cost(500.0, side=Side.BUY)
    assert c.commission == 0.0
    assert c.total > 0          # spread and slippage remain
    assert c.regulatory == 0.0  # buys carry no regulatory fee


def test_regulatory_fees_apply_only_to_equity_sales():
    buy = ALPACA_EQUITY.fill_cost(500.0, side=Side.BUY, qty=10)
    sell = ALPACA_EQUITY.fill_cost(500.0, side=Side.SELL, qty=10)
    assert buy.regulatory == 0.0
    assert sell.regulatory > 0.0
    # $0.0156 on a $500 sale — 0.31bps. Immaterial next to the spread, but not
    # zero: "commission-free" is not "free".
    assert 1e4 * sell.regulatory / 500.0 < 1.0


def test_finra_taf_is_capped():
    huge = ALPACA_EQUITY.fill_cost(1e9, side=Side.SELL, qty=10_000_000)
    assert huge.regulatory < 1e9 * 3e-5 + 8.31


def test_passive_orders_are_cheaper_than_taking_but_are_not_free_money():
    """A resting order captures the spread and then gives it back to adverse
    selection — it fills preferentially when the market is moving against you.

    This matters because Alpaca's paper engine models no queue and no adverse
    selection, so a passive strategy backtests as a money printer. If our own
    model let `spread` go negative, the search would find that infinite edge
    and optimise straight into it.
    """
    taker = ALPACA_CRYPTO.fill_cost(500.0)
    maker = ALPACA_CRYPTO.fill_cost(500.0, passive=True)

    assert maker.spread == pytest.approx(0.0)   # captured, then surrendered
    assert maker.slippage == 0.0
    assert maker.total < taker.total            # still saves the crossing cost
    assert maker.total > 0                      # but never pays you to trade


def test_no_venue_offers_a_negative_round_trip():
    for m in VENUES.values():
        assert m.round_trip_bps(500.0, passive=True) >= 0, f"{m.name} pays you to trade"


def test_crypto_costs_an_order_of_magnitude_more_than_equities():
    """The finding that drives strategy design: the venue with no PDT limit is
    the venue that charges most for using it."""
    eq = ALPACA_EQUITY.round_trip_bps(500.0)
    cr = ALPACA_CRYPTO.round_trip_bps(500.0)

    assert eq < 5
    assert cr > 50
    assert cr / eq > 15


def test_breakeven_hurdles_are_where_we_think_they_are():
    """Crypto was 57.0 bps until the live book was measured.

    The half-spread had been assumed at 2.5 bps; BTC quotes 11.9 bps full and
    the 22-pair median is 39.9. Raising it to a measured 6.0 moved the round
    trip to 64 bps for majors and 94 for everything else. See `0013`.
    """
    table = {name: taker for name, taker, _ in breakeven_table(500.0)}
    assert table["alpaca-equity"] == pytest.approx(3.3, abs=0.2)
    assert table["alpaca-crypto"] == pytest.approx(64.0, abs=0.5)
    assert table["alpaca-crypto-alts"] == pytest.approx(94.0, abs=0.5)
    assert table["binance-spot"] == pytest.approx(24.0, abs=0.5)
    assert table["coinbase-advanced"] == pytest.approx(125.0, abs=0.5)


def test_alts_cost_meaningfully_more_than_majors():
    """Measured: BTC 11.9 bps spread, SOL 39.9, PAXG 122.3."""
    from finb.sim.costs import ALPACA_CRYPTO_ALTS

    assert ALPACA_CRYPTO_ALTS.round_trip_bps(500.0) > ALPACA_CRYPTO.round_trip_bps(500.0) * 1.4


def test_frequent_crypto_trading_demands_an_implausible_edge():
    """20 round trips a day on retail crypto fees needs a ~57bps edge on every
    one of them. Nothing about model quality changes that."""
    hurdle = edge_required(ALPACA_CRYPTO.round_trip_bps(500.0), trades_per_day=20)
    assert hurdle > 55

    # A typical crypto daily move is ~2%. The round trip alone eats a quarter of it.
    daily_move_bps = 200.0
    assert ALPACA_CRYPTO.round_trip_bps(500.0) / daily_move_bps > 0.25


def test_holding_longer_shrinks_the_cost_as_a_share_of_the_move():
    """Same cost, bigger move. This is the whole argument for multi-day holds."""
    cost = ALPACA_CRYPTO.round_trip_bps(500.0)
    daily_vol_bps = 200.0

    one_day = cost / daily_vol_bps
    ten_day = cost / (daily_vol_bps * 10**0.5)   # vol scales with sqrt(time)
    forty_day = cost / (daily_vol_bps * 40**0.5)

    assert one_day > 0.30          # a third of a day's move, at measured spreads
    assert ten_day < one_day
    assert forty_day < 0.06        # which is why the minimum hold is ~48 days


def test_cost_scales_with_notional_not_with_account_size():
    small = ALPACA_CRYPTO.fill_cost(50.0)
    big = ALPACA_CRYPTO.fill_cost(500.0)
    assert big.total == pytest.approx(small.total * 10)
    # In bps terms they are identical — fees do not punish small size directly,
    # they punish frequency.
    assert big.bps_of(500.0) == pytest.approx(small.bps_of(50.0))


def test_minimum_commission_bites_at_small_notional():
    m = CostModel("toy", AssetClass.EQUITY, commission_bps=1.0, min_commission=1.0)
    assert m.fill_cost(50.0).commission == 1.0    # floor dominates
    assert m.fill_cost(50_000.0).commission == 5.0


def test_stress_testing_a_thin_book():
    thin = BINANCE_SPOT.with_spread(50.0)
    assert thin.round_trip_bps(500.0) > BINANCE_SPOT.round_trip_bps(500.0) + 90
    assert BINANCE_SPOT.commission_bps == 10.0  # original untouched


def test_edge_required_falls_with_frequency_toward_the_cost_floor():
    c = COINBASE_ADVANCED.round_trip_bps(500.0)
    assert edge_required(c, 1) > edge_required(c, 20) > c
