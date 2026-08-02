"""Short-side risk.

The bug these exist to prevent: the risk engine used to return
`RiskDecision(True, qty)` for *every* sell, on the reasoning that selling reduces
risk. That is true of closing a long and false of opening a short — two orders
that are identical on the wire and opposite in risk. Enabling shorting against
that code would have permitted unlimited short exposure.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import Side

T0 = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)


def engine(**kw):
    kw.setdefault("max_position_pct", 0.25)
    e = RiskEngine(RiskLimits(capital=500.0, **kw))
    e.update(T0, 500.0)
    return e


def sell(e, *, qty, price=50.0, current_qty=0.0, gross_short=0.0, allow_short=True):
    return e.check(
        side=Side.SELL, symbol="X", qty=qty, price=price, equity=500.0,
        current_qty=current_qty, gross_short=gross_short, allow_short=allow_short,
    )


# --------------------------------------------------------------------------- #
#  Closing versus opening
# --------------------------------------------------------------------------- #


def test_closing_a_long_is_always_allowed():
    d = sell(engine(), qty=2.0, current_qty=2.0)
    assert d.allowed and d.qty == 2.0


def test_selling_more_than_held_only_closes_when_shorts_are_disabled():
    """1 share held, 3 sold: 1 closes, 2 would open a short."""
    d = sell(engine(), qty=3.0, current_qty=1.0, allow_short=False)
    assert d.allowed
    assert d.qty == 1.0
    assert "shorts disabled" in d.reason


def test_opening_a_short_with_no_position_is_refused_when_disabled():
    d = sell(engine(), qty=2.0, current_qty=0.0, allow_short=False)
    assert not d
    assert "not enabled" in d.reason


def test_a_short_inside_the_cap_is_permitted_at_full_size():
    # Default short cap is 10% of $500 = $50. A $25 short fits.
    d = sell(engine(), qty=0.5, price=50.0, current_qty=0.0)
    assert d.allowed and d.qty == pytest.approx(0.5)


def test_a_short_above_the_cap_is_shrunk_rather_than_refused():
    d = sell(engine(), qty=2.0, price=50.0, current_qty=0.0)   # wants $100
    assert d.allowed
    assert d.qty * 50.0 == pytest.approx(50.0)                 # 10% of $500


# --------------------------------------------------------------------------- #
#  Short limits are tighter than long limits
# --------------------------------------------------------------------------- #


def test_short_position_cap_is_tighter_than_the_long_cap():
    e = engine(max_short_position_pct=0.10)
    # $125 long cap vs $50 short cap on a $500 book.
    d = sell(e, qty=10.0, price=50.0)       # wants $500 short
    assert d.qty * 50.0 == pytest.approx(50.0)


def test_the_short_cap_can_never_exceed_the_long_cap():
    """Misconfiguration must not silently loosen the tighter limit."""
    e = engine(max_position_pct=0.10, max_short_position_pct=0.90)
    assert e.max_short_position_pct_effective == 0.10


def test_gross_short_exposure_is_capped_across_all_names():
    e = engine(max_short_position_pct=0.25, max_gross_short_pct=0.30)
    # $150 gross short ceiling; $140 already short elsewhere.
    d = sell(e, qty=10.0, price=50.0, gross_short=140.0)
    assert d.qty * 50.0 == pytest.approx(10.0)


def test_an_existing_short_counts_toward_its_own_cap():
    e = engine(max_short_position_pct=0.10)
    # $50 cap, already short $40 (current_qty is negative).
    d = sell(e, qty=5.0, price=50.0, current_qty=-0.8)
    assert d.qty * 50.0 == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
#  Halted
# --------------------------------------------------------------------------- #


def test_a_halted_account_may_close_but_not_open_a_short():
    e = engine()
    e.update(T0, 300.0)          # trips the drawdown limit
    assert e.state.halted

    assert not sell(e, qty=2.0, current_qty=0.0)

    d = sell(e, qty=3.0, current_qty=2.0)
    assert d.allowed
    assert d.qty == 2.0          # closes the long, opens nothing
    assert "closing only" in d.reason


def test_buying_is_still_blocked_while_halted():
    e = engine()
    e.update(T0, 300.0)
    d = e.check(side=Side.BUY, symbol="X", qty=1.0, price=50.0, equity=300.0)
    assert not d
