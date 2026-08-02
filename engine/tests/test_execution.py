"""Execution guards.

No live endpoint is ever contacted here. These tests assert that constructing a
live broker is impossible unless three independent locks are all open, and that
risk sizing happens before submission rather than after.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from finb.config import LIVE_MAGIC, Settings
from finb.execution.alpaca_paper import AlpacaBroker
from finb.execution.base import (
    AccountSnapshot,
    BrokerPosition,
    OrderRequest,
    OrderType,
    TimeInForce,
)
from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import AssetClass, Side


def settings(**kw) -> Settings:
    base = {
        "alpaca_api_key_id": "PK_TEST",
        "alpaca_api_secret_key": "SECRET_TEST",
        "alpaca_paper": True,
        "finb_allow_live": "no",
    }
    return Settings(**{**base, **kw})


# --------------------------------------------------------------------------- #
#  The three locks
# --------------------------------------------------------------------------- #


def test_paper_is_the_default_and_needs_nothing_special():
    b = AlpacaBroker(settings(), dry_run=True)
    assert b.is_paper


def test_live_endpoint_is_unreachable_without_the_magic_string():
    with pytest.raises(RuntimeError, match="FINB_ALLOW_LIVE is not set"):
        AlpacaBroker(settings(alpaca_paper=False), allow_live=True)


def test_live_endpoint_is_unreachable_without_the_explicit_flag():
    with pytest.raises(RuntimeError, match="did not pass allow_live=True"):
        AlpacaBroker(settings(alpaca_paper=False, finb_allow_live=LIVE_MAGIC))


def test_a_truthy_looking_value_does_not_open_the_lock():
    """'yes', 'true', '1' must all fail — only the exact string works."""
    for value in ("yes", "true", "1", "I_UNDERSTAND", LIVE_MAGIC.lower()):
        with pytest.raises(RuntimeError, match="FINB_ALLOW_LIVE is not set"):
            AlpacaBroker(
                settings(alpaca_paper=False, finb_allow_live=value), allow_live=True
            )


def test_all_three_locks_open_constructs_a_live_broker():
    """Documents the only path that reaches live. No network call is made."""
    b = AlpacaBroker(
        settings(alpaca_paper=False, finb_allow_live=LIVE_MAGIC),
        allow_live=True,
        dry_run=True,
    )
    assert not b.is_paper


def test_missing_credentials_fail_loudly():
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        AlpacaBroker(settings(alpaca_api_key_id=None))


# --------------------------------------------------------------------------- #
#  Risk runs before submission
# --------------------------------------------------------------------------- #


def snapshot(equity=500.0, positions=None) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        cash=equity,
        buying_power=equity,
        positions=positions or [],
        is_paper=True,
        taken_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def broker(monkeypatch):
    b = AlpacaBroker(settings(), RiskEngine(RiskLimits(max_position_pct=0.25)), dry_run=True)
    monkeypatch.setattr(b, "account", lambda: snapshot())
    monkeypatch.setattr(b, "_latest_price", lambda symbol: 100.0)
    return b


def test_an_oversized_order_is_shrunk_not_rejected(broker):
    r = broker.submit(OrderRequest("BTC/USD", Side.BUY, notional=400.0))
    assert r.accepted
    assert r.submitted_qty * 100.0 == pytest.approx(125.0)   # the 25% cap


def test_a_halted_account_blocks_buys_but_lets_you_close(broker, monkeypatch):
    held = BrokerPosition("BTCUSD", 0.5, 50.0, 100.0, 0.0)
    monkeypatch.setattr(broker, "account", lambda: snapshot(positions=[held]))

    broker.risk.update(datetime(2026, 8, 1, 12, 0, tzinfo=UTC), 500.0)
    broker.risk.update(datetime(2026, 8, 1, 13, 0, tzinfo=UTC), 300.0)
    assert broker.risk.state.halted

    buy = broker.submit(OrderRequest("BTC/USD", Side.BUY, notional=50.0))
    assert not buy
    assert "halted" in buy.reason

    # Closing the held position is permitted even while halted — a kill switch
    # that trapped positions would be worse than none.
    close = broker.submit(OrderRequest("BTC/USD", Side.SELL, qty=0.5))
    assert close.accepted


def test_shorting_is_off_unless_explicitly_enabled(broker):
    """A sell with nothing held opens a short. Crypto cannot be shorted on this
    venue at all, so the default must refuse rather than pass it to Alpaca."""
    assert not broker.allow_short
    r = broker.submit(OrderRequest("BTC/USD", Side.SELL, qty=0.1))
    assert not r
    assert "not enabled" in r.reason


def test_dry_run_never_sends(broker, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("dry run must not reach the venue")

    monkeypatch.setattr(broker, "_send", explode)
    r = broker.submit(OrderRequest("BTC/USD", Side.BUY, notional=50.0))
    assert r.accepted
    assert "dry run" in r.reason


def test_an_unpriceable_symbol_is_refused(broker, monkeypatch):
    monkeypatch.setattr(broker, "_latest_price", lambda symbol: None)
    r = broker.submit(OrderRequest("NOPE/USD", Side.BUY, notional=50.0))
    assert not r
    assert "no reference price" in r.reason


def test_a_crypto_position_is_matched_despite_alpacas_two_spellings(broker, monkeypatch):
    """Orders go out as BTC/USD; positions come back as BTCUSD.

    Observed live: without normalisation the concentration check never matched,
    so two orders against a $125 cap built a $164.67 position.
    """
    held = BrokerPosition("BTCUSD", 1.2, 120.0, 100.0, 0.0)   # note: no slash
    monkeypatch.setattr(broker, "account", lambda: snapshot(positions=[held]))

    r = broker.submit(OrderRequest("BTC/USD", Side.BUY, notional=100.0))
    assert r.submitted_qty * 100.0 == pytest.approx(5.0), "cap must see the held position"


def test_canonical_symbol_folds_both_spellings():
    assert AlpacaBroker.canonical("BTC/USD") == AlpacaBroker.canonical("BTCUSD") == "BTCUSD"
    assert AlpacaBroker.canonical("aapl") == "AAPL"


def test_existing_position_counts_toward_the_concentration_cap(broker, monkeypatch):
    # qty 1.2 at market value $120 implies a price of $100, which is also the
    # price the broker derives from the position rather than from the tape.
    held = BrokerPosition("BTC/USD", 1.2, 120.0, 100.0, 0.0)
    monkeypatch.setattr(broker, "account", lambda: snapshot(positions=[held]))

    # Cap is 25% of $500 = $125, and $120 is already held.
    r = broker.submit(OrderRequest("BTC/USD", Side.BUY, notional=100.0))
    assert r.submitted_qty * 100.0 == pytest.approx(5.0)   # only $5 of room left


def test_price_is_taken_from_the_held_position_when_there_is_one(monkeypatch):
    """Avoids a needless market-data call, and keeps sizing consistent with the
    same mark the equity figure was computed from.

    Given its own $2,000 allocation so the sizing limits stay out of the way —
    this test is about where the *price* comes from.
    """
    b = AlpacaBroker(
        settings(), RiskEngine(RiskLimits(capital=2000.0, max_position_pct=0.25)),
        dry_run=True, allocation=2000.0,
    )
    held = BrokerPosition("BTC/USD", 2.0, 300.0, 150.0, 0.0)
    monkeypatch.setattr(b, "account", lambda: snapshot(equity=2000.0, positions=[held]))
    monkeypatch.setattr(
        b, "_latest_price", lambda s: pytest.fail("should not hit market data")
    )

    r = b.submit(OrderRequest("BTC/USD", Side.BUY, notional=150.0))
    assert r.accepted
    assert r.submitted_qty == pytest.approx(1.0)   # $150 at the $150 position mark


# --------------------------------------------------------------------------- #
#  Order validation
# --------------------------------------------------------------------------- #


def test_an_order_must_specify_exactly_one_of_qty_or_notional():
    with pytest.raises(ValueError, match="exactly one"):
        OrderRequest("BTC/USD", Side.BUY)
    with pytest.raises(ValueError, match="exactly one"):
        OrderRequest("BTC/USD", Side.BUY, qty=1.0, notional=100.0)


def test_limit_orders_require_a_price():
    with pytest.raises(ValueError, match="require a limit_price"):
        OrderRequest("BTC/USD", Side.BUY, qty=1.0, order_type=OrderType.LIMIT)


def test_notional_orders_are_supported():
    """The only way a $500 account holds more than a couple of positions."""
    o = OrderRequest("AAPL", Side.BUY, notional=25.0)
    assert o.notional == 25.0 and o.qty is None


# --------------------------------------------------------------------------- #
#  Time in force is venue-specific
# --------------------------------------------------------------------------- #


def test_crypto_defaults_to_gtc_not_day():
    """Alpaca rejects day-orders on crypto with `invalid crypto time_in_force`,
    and this default previously caused every crypto order to fail at the venue."""
    o = OrderRequest("BTC/USD", Side.BUY, notional=40.0, asset_class=AssetClass.CRYPTO)
    assert o.time_in_force is TimeInForce.GTC


def test_equities_default_to_day():
    o = OrderRequest("AAPL", Side.BUY, notional=40.0, asset_class=AssetClass.EQUITY)
    assert o.time_in_force is TimeInForce.DAY


def test_an_explicit_day_order_on_crypto_is_rejected_locally():
    """Caught here rather than after a round trip to a numeric error code."""
    with pytest.raises(ValueError, match="cannot use time_in_force=day"):
        OrderRequest(
            "BTC/USD", Side.BUY, notional=40.0,
            asset_class=AssetClass.CRYPTO, time_in_force=TimeInForce.DAY,
        )


def test_explicit_ioc_on_crypto_is_allowed():
    o = OrderRequest(
        "BTC/USD", Side.BUY, notional=40.0,
        asset_class=AssetClass.CRYPTO, time_in_force=TimeInForce.IOC,
    )
    assert o.time_in_force is TimeInForce.IOC


# --------------------------------------------------------------------------- #
#  The allocation cap
# --------------------------------------------------------------------------- #


def test_the_brokers_generous_balance_does_not_become_our_budget(monkeypatch):
    """Alpaca funds paper accounts with $100k and 4x margin. Sizing against that
    would discard the $500 constraint entirely — a 25% cap on $95k is $23,750."""
    b = AlpacaBroker(
        settings(), RiskEngine(RiskLimits(capital=500.0, max_position_pct=0.25)),
        dry_run=True, allocation=500.0,
    )
    monkeypatch.setattr(b, "account", lambda: snapshot(equity=95_468.36))
    monkeypatch.setattr(b, "_latest_price", lambda s: 100.0)

    assert b.budget(95_468.36) == 500.0

    r = b.submit(OrderRequest("BTC/USD", Side.BUY, notional=50_000.0))
    assert r.accepted
    assert r.submitted_qty * 100.0 == pytest.approx(125.0)   # 25% of $500, not of $95k


def test_budget_is_capped_by_the_account_when_the_account_is_smaller(monkeypatch):
    b = AlpacaBroker(settings(), dry_run=True, allocation=500.0)
    assert b.budget(120.0) == 120.0


def test_allocation_defaults_to_the_configured_capital():
    b = AlpacaBroker(settings(finb_capital_usd=500.0), dry_run=True)
    assert b.allocation == 500.0
