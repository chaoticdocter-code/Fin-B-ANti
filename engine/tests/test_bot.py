"""Bot decision logic.

Two behaviours matter most and both were found by running it, not by writing it:
it must converge rather than churn, and when it cannot fund a rebalance it must
say so instead of firing dust orders at the minimum notional.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from finb.bot import PositionState
from finb.execution.alpaca_paper import AlpacaBroker
from finb.execution.base import AccountSnapshot, BrokerPosition, OrderRequest
from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import Side

T0 = datetime(2026, 8, 2, 8, 21, tzinfo=UTC)


# --------------------------------------------------------------------------- #
#  Position state
# --------------------------------------------------------------------------- #


def test_entry_times_survive_a_restart(tmp_path):
    """Alpaca does not report when a position was opened, and the holding policy
    needs it. A bot that forgets churns straight through its minimum hold."""
    p = tmp_path / "positions.json"
    s = PositionState(p)
    s.opened("BTC/USD", T0)

    assert PositionState(p).entry_time("BTC/USD") == T0


def test_entry_time_matches_across_both_symbol_spellings(tmp_path):
    s = PositionState(tmp_path / "p.json")
    s.opened("BTC/USD", T0)
    assert s.entry_time("BTCUSD") == T0
    assert s.entry_time("btc/usd") == T0


def test_reopening_does_not_reset_the_clock(tmp_path):
    """Topping up a position must not restart its minimum hold."""
    s = PositionState(tmp_path / "p.json")
    s.opened("BTC/USD", T0)
    s.opened("BTC/USD", T0 + timedelta(days=10))
    assert s.entry_time("BTC/USD") == T0


def test_closing_forgets_the_position(tmp_path):
    s = PositionState(tmp_path / "p.json")
    s.opened("BTC/USD", T0)
    s.closed("BTC/USD")
    assert s.entry_time("BTC/USD") is None


def test_reconcile_drops_positions_that_no_longer_exist(tmp_path):
    s = PositionState(tmp_path / "p.json")
    s.opened("BTC/USD", T0)
    s.opened("ETH/USD", T0)
    s.reconcile({"BTCUSD"})
    assert s.entry_time("BTC/USD") == T0
    assert s.entry_time("ETH/USD") is None


def test_corrupt_state_file_does_not_crash_the_bot(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("{ this is not json", encoding="utf-8")
    s = PositionState(p)
    assert s.entry_time("BTC/USD") is None
    s.opened("BTC/USD", T0)
    assert json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  Dust orders
# --------------------------------------------------------------------------- #


def settings(**kw):
    from finb.config import Settings

    return Settings(
        alpaca_api_key_id="PK", alpaca_api_secret_key="SEC",
        alpaca_paper=True, finb_allow_live="no", **kw
    )


def snapshot(equity=500.0, positions=None):
    return AccountSnapshot(
        equity=equity, cash=equity, buying_power=equity,
        positions=positions or [], is_paper=True, taken_at=T0,
    )


def test_a_fully_invested_book_refuses_rather_than_levering(monkeypatch):
    """Observed live: a $250 target became a $1.00 order because the book was
    fully invested and the holding policy would not release capital.

    The risk engine is what stops it — an order whose remaining room is below the
    minimum notional is rejected, not shrunk into dust.
    """
    b = AlpacaBroker(
        settings(), RiskEngine(RiskLimits(capital=500.0, min_notional=5.0)),
        dry_run=True, allocation=500.0,
    )
    full = [BrokerPosition("BTCUSD", 1.0, 498.0, 498.0, 0.0)]
    monkeypatch.setattr(b, "account", lambda: snapshot(positions=full))
    monkeypatch.setattr(b, "_latest_price", lambda s: 100.0)

    r = b.submit(OrderRequest("ETH/USD", Side.BUY, notional=250.0))
    assert not r
    assert "no leverage" in r.reason
    assert r.submitted_notional == 0.0


def test_a_fundable_order_reports_what_was_actually_committed(monkeypatch):
    b = AlpacaBroker(
        settings(), RiskEngine(RiskLimits(capital=500.0, max_position_pct=0.25)),
        dry_run=True, allocation=500.0,
    )
    monkeypatch.setattr(b, "account", lambda: snapshot())
    monkeypatch.setattr(b, "_latest_price", lambda s: 100.0)

    r = b.submit(OrderRequest("BTC/USD", Side.BUY, notional=400.0))
    assert r.accepted
    # Requested $400, capped to 25% of $500. The result must report the truth.
    assert r.submitted_notional == pytest.approx(125.0)
    assert r.submitted_notional < 400.0
