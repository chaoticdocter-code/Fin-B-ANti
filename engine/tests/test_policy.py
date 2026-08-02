"""Holding policy, and the ledger's accounting for observations.

These two encode the operator's decisions: keep crypto but only at multi-week
holds, and keep watching P&L daily but pay for it in the trial count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from finb.evaluation import PromotionGate, SearchLedger
from finb.sim.constraints import AssetClass
from finb.sim.costs import (
    ALPACA_CRYPTO,
    ALPACA_EQUITY,
    ALPACA_EQUITY_BROAD,
    COINBASE_ADVANCED,
)
from finb.sim.policy import HoldingPolicy

# --------------------------------------------------------------------------- #
#  Holding policy
# --------------------------------------------------------------------------- #


def test_minimum_hold_is_derived_from_cost_not_hardcoded():
    p = HoldingPolicy.from_costs(ALPACA_EQUITY_BROAD, ALPACA_CRYPTO)

    # Crypto's 57bps round trip forces a multi-week hold.
    assert p.min_hold_crypto.days >= 14
    assert p.min_hold_equity.days >= 5


def test_a_more_expensive_venue_forces_a_longer_hold():
    cheap = HoldingPolicy.from_costs(ALPACA_EQUITY_BROAD, ALPACA_CRYPTO)
    dear = HoldingPolicy.from_costs(ALPACA_EQUITY_BROAD, COINBASE_ADVANCED)
    assert dear.min_hold_crypto > cheap.min_hold_crypto


def test_equity_floor_prevents_cost_feasibility_becoming_a_licence_to_churn():
    """On megacaps, 3.4bps implies a half-day hold — cheap enough that pure cost
    feasibility would wave through daily churn. The floor is what stops that."""
    mega = HoldingPolicy.from_costs(ALPACA_EQUITY, ALPACA_CRYPTO, equity_floor_days=5)
    assert mega.min_hold_equity == timedelta(days=5)

    # A realistic broad universe at 13.4bps already demands longer than the
    # floor, so the floor stops binding and cost takes over.
    broad = HoldingPolicy.from_costs(ALPACA_EQUITY_BROAD, ALPACA_CRYPTO, equity_floor_days=5)
    assert broad.min_hold_equity > timedelta(days=5)


def test_exit_is_blocked_until_the_minimum_hold_elapses():
    p = HoldingPolicy(min_hold_crypto=timedelta(days=15))
    entry = datetime(2026, 8, 1, tzinfo=UTC)

    early = p.check_exit(entry, entry + timedelta(days=3), AssetClass.CRYPTO)
    assert not early.allowed
    assert early.days_remaining == 12.0
    assert "minimum hold for crypto is 15d" in early.reason

    late = p.check_exit(entry, entry + timedelta(days=15), AssetClass.CRYPTO)
    assert late.allowed
    assert late.days_remaining == 0.0


def test_equity_and_crypto_have_separate_clocks():
    p = HoldingPolicy(min_hold_equity=timedelta(days=5), min_hold_crypto=timedelta(days=15))
    entry = datetime(2026, 8, 1, tzinfo=UTC)
    at = entry + timedelta(days=7)

    assert p.check_exit(entry, at, AssetClass.EQUITY).allowed
    assert not p.check_exit(entry, at, AssetClass.CRYPTO).allowed


def test_breadth_is_what_makes_a_slow_crypto_strategy_measurable():
    """One pair at a 38-day hold can never be validated; twenty can."""
    p = HoldingPolicy(min_hold_crypto=timedelta(days=38))

    solo = p.years_to_observations(AssetClass.CRYPTO, n_symbols=1)
    broad = p.years_to_observations(AssetClass.CRYPTO, n_symbols=20)

    assert solo > 10          # a decade for 100 trades
    assert broad < 0.75       # under nine months
    assert p.trades_per_year(AssetClass.CRYPTO, 20) > 150


# --------------------------------------------------------------------------- #
#  The ledger charges for looking
# --------------------------------------------------------------------------- #


def test_observations_and_hypotheses_count_as_trials(tmp_path):
    led = SearchLedger(tmp_path / "l.jsonl")
    rng = np.random.default_rng(1)

    for i in range(5):
        led.record(f"v{i}", returns=rng.normal(0, 0.01, 200))
    for d in range(20):
        led.record_observation(f"2026-08-{d + 1:02d}", note="reviewed live equity curve")
    led.record_hypothesis("add a volatility regime filter")

    assert led.count_by_kind() == {"variant": 5, "observation": 20, "hypothesis": 1}
    assert led.n_trials == 26          # not 5
    assert len(led.variants) == 5


def test_daily_watching_raises_the_luck_hurdle(tmp_path):
    """The cost of the operator's chosen workflow, made explicit."""
    rng = np.random.default_rng(20260731)
    returns = rng.normal(0.0012, 0.01, size=1500)
    gate = PromotionGate()

    quiet = SearchLedger(tmp_path / "quiet.jsonl")
    watched = SearchLedger(tmp_path / "watched.jsonl")
    for led in (quiet, watched):
        for i in range(40):
            led.record(f"v{i}", returns=rng.normal(0, 0.01, 250))

    # A year of looking at P&L every session.
    for d in range(250):
        watched.record_observation(f"s{d}")

    a = gate.evaluate_ledger(returns, quiet)
    b = gate.evaluate_ledger(returns, watched)

    assert watched.n_trials > quiet.n_trials * 6
    assert b.expected_max_sr_annual > a.expected_max_sr_annual
    assert b.dsr < a.dsr


def test_sharpe_dispersion_ignores_non_variant_rows(tmp_path):
    led = SearchLedger(tmp_path / "l.jsonl")
    rng = np.random.default_rng(2)
    for i in range(20):
        led.record(f"v{i}", returns=rng.normal(0, 0.01, 300))
    before = led.sr_variance()

    for d in range(100):
        led.record_observation(f"s{d}")

    # Observations carry no Sharpe, so they must not drag the dispersion to zero
    # — that would collapse the luck hurdle exactly when it should be rising.
    assert led.sr_variance() == before
    assert led.n_trials == 120


def test_observations_survive_a_restart(tmp_path):
    p = tmp_path / "l.jsonl"
    led = SearchLedger(p)
    led.record_observation("2026-08-01")
    led.record_hypothesis("try dollar bars")

    reopened = SearchLedger(p)
    assert reopened.n_trials == 2
    assert reopened.count_by_kind()["hypothesis"] == 1
    assert "hypothesis" in reopened.summary()
