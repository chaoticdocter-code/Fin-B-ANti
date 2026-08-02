"""The null control arm, and the correlation correction it exposed.

The headline test is `test_correlated_trials_do_not_break_the_gate`. Before the
effective-N correction, 100 trials correlated at 0.9 pushed the gate's
false-promotion rate from 0% to 6% on real BTC returns — the failure mode a
population of near-identical variants produces by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from finb.evaluation.gate import PromotionGate, effective_n_trials, expected_max_sharpe
from finb.evaluation.null_cohort import (
    NullCohort,
    build_correlated_null_cohort,
    build_null_cohort,
    calibrate_gate,
    circular_block_bootstrap,
    mean_pairwise_correlation,
    optimal_block_size,
    shuffle_labels,
)


def fat_tailed_ar1(rng, n=1500, rho=0.15, vol=0.012):
    """A series with autocorrelation and fat tails, like a real return series."""
    shocks = rng.standard_t(df=4, size=n) * vol / np.sqrt(2.0)
    out = np.empty(n)
    out[0] = shocks[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + shocks[i]
    return out


# --------------------------------------------------------------------------- #
#  Bootstrap mechanics
# --------------------------------------------------------------------------- #


def test_block_bootstrap_preserves_fat_tails():
    rng = np.random.default_rng(1)
    r = fat_tailed_ar1(rng)
    s = circular_block_bootstrap(r, rng)

    def kurt(x):
        z = (x - x.mean()) / x.std(ddof=1)
        return float((z**4).mean())

    assert s.size == r.size
    assert kurt(s) > 4.0                      # tails survived
    assert abs(kurt(s) - kurt(r)) < 0.6 * kurt(r)


def test_block_size_grows_with_autocorrelation():
    rng = np.random.default_rng(2)
    weak = optimal_block_size(fat_tailed_ar1(rng, rho=0.02))
    strong = optimal_block_size(fat_tailed_ar1(rng, rho=0.75))
    assert strong > weak


def test_cohort_is_a_null_but_not_a_degenerate_one():
    """Every column must have zero *expected* Sharpe while still varying.

    Demeaning each column by its own mean would zero every Sharpe exactly,
    collapsing dispersion and silently reporting a 0% false-promotion rate.
    """
    rng = np.random.default_rng(3)
    r = fat_tailed_ar1(rng) + 0.0008          # a drifting market
    cohort = build_null_cohort(r, size=200, rng=rng)

    means = cohort.mean(axis=0)
    assert abs(means.mean()) < 5e-4           # no drift on average
    assert means.std(ddof=1) > 1e-5           # but individual luck remains
    assert np.ptp(means) > 0


def test_correlated_cohort_hits_its_target_correlation():
    rng = np.random.default_rng(4)
    r = fat_tailed_ar1(rng)
    for target in (0.0, 0.3, 0.6, 0.9):
        c = build_correlated_null_cohort(r, size=40, correlation=target, rng=rng)
        assert mean_pairwise_correlation(c) == pytest.approx(target, abs=0.06)


def test_shuffle_labels_keeps_the_marginal_distribution():
    rng = np.random.default_rng(5)
    y = np.array([1] * 30 + [0] * 70)
    s = shuffle_labels(y, rng)
    assert sorted(s) == sorted(y)
    assert not np.array_equal(s, y)


# --------------------------------------------------------------------------- #
#  Effective N
# --------------------------------------------------------------------------- #


def test_effective_n_interpolates_between_one_and_all():
    assert effective_n_trials(100, 0.0) == 100
    assert effective_n_trials(100, 1.0) == pytest.approx(1.0)
    assert effective_n_trials(100, 0.9) == pytest.approx(10.9)
    assert 1 < effective_n_trials(100, 0.5) < 100


def test_variance_floor_stops_the_hurdle_collapsing():
    """Correlated trials shrink observed Sharpe dispersion toward zero. Without
    a floor that drives SR* to zero and the deflation stops deflating."""
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.01, 1000)
    gate = PromotionGate()

    # Dispersion reported as essentially nil, as a highly correlated population
    # would report it.
    v = gate.evaluate(r, n_trials=100, sr_variance=1e-9, trial_correlation=0.9)
    assert v.expected_max_sr_annual > 0.3, "hurdle collapsed to nothing"

    # Raw SR* with no floor would be ~0.
    assert expected_max_sharpe(100, 1e-9) < 1e-3


# --------------------------------------------------------------------------- #
#  The regression test that matters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.9])
def test_correlated_trials_do_not_break_the_gate(rho):
    """Measured on real BTC returns before the fix: 0% / 4.3% / 6.0%."""
    rng = np.random.default_rng(20260801)
    r = fat_tailed_ar1(rng, n=1200)

    res = calibrate_gate(r, n_trials=60, n_repeats=80, seed=11, correlation=rho)

    assert res["false_promotion_rate"] <= 0.05, (
        f"rho={rho} gave FPR {res['false_promotion_rate']:.1%}"
    )


def test_calibration_reports_what_a_lucky_null_looks_like():
    rng = np.random.default_rng(8)
    r = fat_tailed_ar1(rng, n=1200)
    res = calibrate_gate(r, n_trials=60, n_repeats=60, seed=3)

    # The best of 60 zero-skill strategies still posts a respectable Sharpe.
    # That number is the reason the gate exists.
    assert res["median_winner_sharpe_annual"] > 0.3
    assert res["max_winner_sharpe_annual"] > res["median_winner_sharpe_annual"]


# --------------------------------------------------------------------------- #
#  Beating the best of the null
# --------------------------------------------------------------------------- #


def test_a_null_champion_does_not_beat_the_null_cohort():
    rng = np.random.default_rng(9)
    market = fat_tailed_ar1(rng, n=1200)
    champion = circular_block_bootstrap(market, rng) - market.mean()

    v = NullCohort(size=150, seed=1).judge(champion, market)

    assert not v.beats_best_of_null
    assert v.empirical_p_value > 0.01
    assert "does NOT clear" in v.summary


def test_a_genuinely_strong_champion_beats_the_cohort():
    rng = np.random.default_rng(10)
    market = fat_tailed_ar1(rng, n=1200)
    champion = market + 0.004                 # a large, real edge

    v = NullCohort(size=150, seed=1).judge(champion, market)

    assert v.beats_best_of_null
    assert v.empirical_p_value == 0.0
    assert v.champion > v.null_best > v.null_p95 > v.null_median
