"""Risk limits.

These tests describe what happens when something is already going wrong, which
is the only time this layer matters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import Side

T0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def engine(**kw):
    return RiskEngine(RiskLimits(**kw))


def buy(e, *, qty=1.0, price=100.0, equity=500.0, held=0.0, gross=0.0):
    return e.check(
        side=Side.BUY, symbol="X", qty=qty, price=price,
        equity=equity, current_position_value=held, gross_exposure=gross,
    )


# --------------------------------------------------------------------------- #
#  Kill switch
# --------------------------------------------------------------------------- #


def test_drawdown_trips_the_kill_switch():
    # Daily limit disabled so this isolates the drawdown rule — otherwise a
    # 500 -> 450 move in one session trips the daily loss limit first.
    e = engine(max_drawdown_pct=0.20, max_daily_loss_pct=1.0)
    e.update(T0, 500.0)
    e.update(T0, 450.0)
    assert not e.state.halted

    e.update(T0, 399.0)
    assert e.state.halted
    assert "drawdown" in e.state.halt_reason
    assert not buy(e, equity=399.0)


def test_daily_loss_trips_independently_of_drawdown():
    """A sharp one-day loss halts even if the account is still near its peak."""
    e = engine(max_daily_loss_pct=0.05, max_drawdown_pct=0.50)
    e.update(T0, 500.0)
    e.update(T0 + timedelta(hours=1), 470.0)

    assert e.state.halted
    assert "daily loss" in e.state.halt_reason
    assert e.state.drawdown > -0.50   # drawdown limit was nowhere near


def test_closing_a_long_is_always_permitted_even_when_halted():
    """A kill switch that trapped positions would be worse than none.

    Note `current_qty` — it is what marks this as *closing*. The same order with
    no position behind it opens a short, which is risk-increasing and refused.
    """
    e = engine()
    e.update(T0, 500.0)
    e.update(T0, 300.0)
    assert e.state.halted

    assert e.check(
        side=Side.SELL, symbol="X", qty=1.0, price=100.0, equity=300.0, current_qty=1.0
    ).allowed

    # Same order, nothing held: this would open a short.
    assert not e.check(
        side=Side.SELL, symbol="X", qty=1.0, price=100.0, equity=300.0, current_qty=0.0
    )


def test_recovery_is_manual_and_requires_a_written_reason():
    e = engine()
    e.update(T0, 500.0)
    e.update(T0, 300.0)

    with pytest.raises(ValueError, match="written acknowledgement"):
        e.reset_halt("   ")

    e.reset_halt("data gap caused a false mark; verified against the exchange")
    assert not e.state.halted
    assert buy(e, equity=300.0)
    assert any("resumed" in h for h in e.state.history)


def test_a_new_day_resets_the_daily_budget_but_not_the_drawdown():
    e = engine(max_daily_loss_pct=0.05, max_drawdown_pct=0.20)
    e.update(T0, 500.0)
    e.update(T0 + timedelta(days=1), 460.0)   # new day, -8% from peak

    assert not e.state.halted
    assert e.state.trades_today == 0
    assert e.state.drawdown == pytest.approx(-0.08)


# --------------------------------------------------------------------------- #
#  Exposure
# --------------------------------------------------------------------------- #


def test_position_concentration_is_capped_by_shrinking_not_rejecting():
    e = engine(max_position_pct=0.25)
    e.update(T0, 500.0)

    d = buy(e, qty=2.0, price=100.0, held=0.0)   # $200 vs $125 cap
    assert d.allowed
    assert d.qty * 100.0 == pytest.approx(125.0)


def test_a_full_position_is_rejected_outright():
    e = engine(max_position_pct=0.25)
    e.update(T0, 500.0)
    assert not buy(e, qty=1.0, held=125.0)


def test_leverage_is_impossible():
    e = engine(max_gross_exposure=1.0, max_position_pct=1.0)
    e.update(T0, 500.0)

    d = buy(e, qty=5.0, price=100.0, gross=400.0)   # wants $500 on top of $400
    assert d.qty * 100.0 == pytest.approx(100.0)

    assert not buy(e, qty=1.0, gross=500.0)


def test_dust_orders_are_rejected():
    e = engine(min_notional=1.0)
    e.update(T0, 500.0)

    d = buy(e, qty=0.001, price=100.0)   # a $0.10 order
    assert not d
    assert "minimum notional" in d.reason


def test_daily_trade_cap_is_a_runaway_backstop():
    e = engine(max_trades_per_day=3)
    e.update(T0, 500.0)
    for _ in range(3):
        assert buy(e)
        e.record_fill()

    d = buy(e)
    assert not d
    assert "runaway-loop backstop" in d.reason


# --------------------------------------------------------------------------- #
#  Sizing
# --------------------------------------------------------------------------- #


def test_volatile_assets_get_smaller_positions():
    e = engine(vol_target_annual=0.20, max_position_pct=0.25)
    calm = e.position_size(500.0, 100.0, asset_volatility=0.005)
    wild = e.position_size(500.0, 100.0, asset_volatility=0.05)
    assert wild < calm


def test_sizing_never_exceeds_the_concentration_cap():
    e = engine(max_position_pct=0.25)
    # Volatility far below target — scaling would otherwise want a huge position.
    qty = e.position_size(500.0, 100.0, asset_volatility=1e-5)
    assert qty * 100.0 <= 500.0 * 0.25 + 1e-9


def test_conviction_scales_the_position():
    e = engine()
    full = e.position_size(500.0, 100.0, asset_volatility=0.02, conviction=1.0)
    half = e.position_size(500.0, 100.0, asset_volatility=0.02, conviction=0.5)
    assert half == pytest.approx(full / 2)


def test_sizing_returns_zero_rather_than_a_dust_position():
    e = engine(min_notional=50.0)
    assert e.position_size(500.0, 100.0, asset_volatility=0.02, conviction=0.01) == 0.0


def test_crypto_volatility_forces_a_much_smaller_book():
    """At 70% annualised vol against a 20% target, sizing should cut hard."""
    e = engine(vol_target_annual=0.20, max_position_pct=0.25)
    crypto_daily_vol = 0.70 / (365**0.5)
    qty = e.position_size(500.0, 100.0, asset_volatility=crypto_daily_vol)
    assert qty * 100.0 < 0.25 * 500.0 * 0.35


def test_status_string_is_readable():
    e = engine()
    e.update(T0, 500.0)
    assert "ok" in e.status()
    e.update(T0, 300.0)
    assert "HALTED" in e.status()
