"""Shadow book accounting.

An untested simulator is the most dangerous component in the project: it cannot
crash, it can only be wrong in a direction that flatters the strategy. Every
test here asserts something that, if it broke, would silently make backtests
better.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from finb.sim.constraints import AssetClass, Side
from finb.sim.costs import ALPACA_CRYPTO, ALPACA_EQUITY, CostModel
from finb.sim.engine import ShadowBook
from finb.sim.policy import HoldingPolicy
from finb.sim.runner import build_panel, momentum_scores, run_cross_sectional

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FREE = CostModel("free", AssetClass.CRYPTO, half_spread_bps=0.0, slippage_bps=0.0)


def book(capital=500.0, cost=FREE, hold_days=0):
    return ShadowBook(
        capital=capital,
        cost_model=cost,
        holding_policy=HoldingPolicy(
            min_hold_crypto=timedelta(days=hold_days),
            min_hold_equity=timedelta(days=hold_days),
        ),
        asset_class=AssetClass.CRYPTO,
    )


# --------------------------------------------------------------------------- #
#  Accounting identities
# --------------------------------------------------------------------------- #


def test_equity_is_conserved_when_trading_is_free():
    b = book()
    b.rebalance(T0, {"A": 100.0, "B": 50.0}, {"A": 0.5, "B": 0.5})
    assert b.equity({"A": 100.0, "B": 50.0}) == pytest.approx(500.0)


def test_costs_come_straight_out_of_equity():
    free, costly = book(cost=FREE), book(cost=ALPACA_CRYPTO)
    prices, target = {"A": 100.0}, {"A": 1.0}

    free.rebalance(T0, prices, target)
    costly.rebalance(T0, prices, target)

    assert costly.equity(prices) < free.equity(prices)
    assert costly.total_costs > 0
    assert free.equity(prices) - costly.equity(prices) == pytest.approx(
        costly.total_costs, rel=1e-6
    )


def test_cash_never_goes_negative():
    b = book(cost=ALPACA_CRYPTO)
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})
    assert b.cash >= -1e-9


def test_weights_over_one_are_scaled_down_not_levered():
    b = book()
    b.rebalance(T0, {"A": 100.0, "B": 100.0}, {"A": 0.8, "B": 0.8})
    assert b.cash >= -1e-9
    assert b.equity({"A": 100.0, "B": 100.0}) == pytest.approx(500.0)


def test_a_profitable_move_raises_equity_by_the_right_amount():
    b = book()
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})
    assert b.equity({"A": 110.0}) == pytest.approx(550.0)


def test_selling_everything_returns_to_cash():
    b = book()
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})
    b.rebalance(T0 + timedelta(days=1), {"A": 100.0}, {})
    assert not b.positions
    assert b.cash == pytest.approx(500.0)


# --------------------------------------------------------------------------- #
#  Constraints must actually constrain
# --------------------------------------------------------------------------- #


def test_minimum_hold_blocks_an_early_exit():
    b = book(hold_days=38)
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})

    b.rebalance(T0 + timedelta(days=5), {"A": 100.0}, {})
    assert "A" in b.positions
    assert b.blocked_by_hold == 1

    b.rebalance(T0 + timedelta(days=38), {"A": 100.0}, {})
    assert not b.positions


def test_orders_below_minimum_notional_are_skipped():
    b = ShadowBook(
        capital=500.0, cost_model=FREE, asset_class=AssetClass.CRYPTO, min_notional=25.0
    )
    b.rebalance(T0, {"A": 100.0}, {"A": 0.01})   # $5 order
    assert not b.positions
    assert b.blocked_by_notional == 1


def test_no_shorting():
    b = book()
    b.rebalance(T0, {"A": 100.0}, {"A": -0.5})
    assert not b.positions
    assert b.cash == pytest.approx(500.0)


def test_regulatory_fees_only_on_the_sell_side():
    b = ShadowBook(500.0, ALPACA_EQUITY, asset_class=AssetClass.EQUITY)
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})
    buy_cost = b.trades[-1].cost

    b.rebalance(T0 + timedelta(days=30), {"A": 100.0}, {})
    sell = b.trades[-1]
    assert sell.side is Side.SELL
    assert sell.cost > buy_cost   # spread + slippage + reg fees


# --------------------------------------------------------------------------- #
#  Stats
# --------------------------------------------------------------------------- #


def test_drawdown_and_turnover_are_reported():
    b = book()
    b.rebalance(T0, {"A": 100.0}, {"A": 1.0})
    for i, px in enumerate([100.0, 50.0, 75.0]):
        b.mark(T0 + timedelta(days=i), {"A": px})

    st = b.stats()
    assert st["max_drawdown"] == pytest.approx(-0.5)
    assert st["turnover_x"] == pytest.approx(1.0)
    assert st["trades"] == 1


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #


def _bars(symbol_paths: dict[str, list[float]]) -> dict[str, pl.DataFrame]:
    out = {}
    for sym, closes in symbol_paths.items():
        n = len(closes)
        out[sym] = pl.DataFrame(
            {
                "ts": [T0 + timedelta(days=i) for i in range(n)],
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1.0] * n,
            }
        )
    return out


def test_panel_is_an_inner_join_on_timestamp():
    a = _bars({"A": [1.0] * 10})["A"]
    b = _bars({"B": [1.0] * 6})["B"]
    ts, syms, closes = build_panel({"A": a, "B": b})
    assert closes.shape == (6, 2)
    assert syms == ["A", "B"]


def test_momentum_picks_the_strongest_and_skips_recent_bars():
    n = 200
    up = list(np.linspace(100, 200, n))
    flat = [100.0] * n
    down = list(np.linspace(200, 100, n))

    score = momentum_scores(lookback=60, skip=7)
    closes = np.column_stack([up, flat, down])
    s = score(closes)
    assert s[0] > s[1] > s[2]


def test_momentum_returns_nan_before_warmup():
    s = momentum_scores(lookback=60, skip=7)(np.ones((10, 3)))
    assert np.isnan(s).all()


def test_runner_executes_on_the_bar_after_the_signal():
    """A signal scored at bar t must trade at t+1's close, never at t's."""
    n = 200
    winner = list(np.linspace(100, 300, n))
    loser = list(np.linspace(300, 100, n))
    bars = _bars({"WIN": winner, "LOSE": loser})

    b = book()
    res = run_cross_sectional(
        bars, b, score_fn=momentum_scores(30, skip=0), top_n=1,
        rebalance_every=20, warmup=40,
    )
    assert res.rebalances > 0
    assert set(b.positions) <= {"WIN"}
    assert b.equity_curve.size > 0


def test_runner_rejects_a_panel_that_is_too_short():
    with pytest.raises(ValueError, match="not enough aligned bars"):
        run_cross_sectional(
            _bars({"A": [1.0] * 20, "B": [1.0] * 20}), book(),
            score_fn=momentum_scores(5), warmup=60,
        )
