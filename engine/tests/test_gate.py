"""The gate's job is to reject the luckiest variant of many.

The first test is the one that matters: 500 variants of pure noise, take the
best-looking one, and confirm the gate throws it out. Those 500 need not be
simultaneous — for an evolving bot they are 500 attempts spread over months —
but the arithmetic is identical. If this test ever goes green-to-red, the bot
has started laundering luck as skill.
"""

from __future__ import annotations

import numpy as np
import pytest

from finb.evaluation.gate import (
    PromotionGate,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _noise(rng, t, n, sd=0.01):
    return rng.normal(0.0, sd, size=(t, n))


# --------------------------------------------------------------------------- #
#  The headline property
# --------------------------------------------------------------------------- #


def test_luckiest_of_many_noise_strategies_is_rejected():
    rng = np.random.default_rng(20260730)
    T, N = 252, 500
    M = _noise(rng, T, N)

    sharpes = np.array([sharpe_ratio(M[:, j]) for j in range(N)])
    winner = int(np.argmax(sharpes))

    # The naive read of this "winner" is impressive...
    naive_annual = sharpe_ratio(M[:, winner], periods_per_year=252)
    assert naive_annual > 1.5, "expected the max of 500 noise draws to look good"

    # ...and the gate is not fooled by it.
    gate = PromotionGate()
    v = gate.evaluate(M[:, winner], n_trials=N, sr_variance=float(sharpes.var(ddof=1)))

    assert not v.passed
    assert v.dsr < 0.95
    assert any("deflated Sharpe" in r or "luck hurdle" in r for r in v.reasons)
    # The luck hurdle should be in the same league as the observed Sharpe.
    assert v.expected_max_sr_annual > 1.0


# The effect size below is deliberately larger than anything real markets offer.
# These tests check the arithmetic of the statistic, not a claim about achievable
# edge — sampling noise at a realistic 1.0-1.5 Sharpe would make them flaky.
_STRONG_EDGE = dict(loc=0.002, scale=0.01, size=2000)  # ~0.2 daily Sharpe


def test_genuine_edge_found_in_few_trials_passes():
    rng = np.random.default_rng(7)
    r = rng.normal(**_STRONG_EDGE)

    v = PromotionGate().evaluate(r, n_trials=10, sr_variance=0.0025)

    assert v.passed, v.reasons
    assert v.dsr > 0.95


def test_same_edge_fails_once_it_took_many_trials_to_find():
    """Identical returns, identical everything — only the search cost changes."""
    rng = np.random.default_rng(7)
    r = rng.normal(**_STRONG_EDGE)
    gate = PromotionGate()

    cheap = gate.evaluate(r, n_trials=10, sr_variance=0.0025)
    expensive = gate.evaluate(r, n_trials=100_000, sr_variance=0.0025)

    assert cheap.passed
    assert not expensive.passed
    assert expensive.expected_max_sr_annual > cheap.expected_max_sr_annual


def test_the_hurdle_creeps_up_as_the_bot_keeps_evolving():
    """One bot, evolving. The bar rises with every variant it has ever tried,
    which is why the trial count has to be persisted rather than remembered."""
    k = np.sqrt(252)
    session_10 = expected_max_sharpe(50, 0.0025) * k
    session_100 = expected_max_sharpe(800, 0.0025) * k
    session_500 = expected_max_sharpe(5000, 0.0025) * k

    assert session_10 < session_100 < session_500
    # Growth is sub-linear (~sqrt(log N)) — the search is affordable, but it is
    # emphatically not free.
    assert session_500 < 3 * session_10


# --------------------------------------------------------------------------- #
#  Component behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 10, 100, 10_000])
def test_luck_hurdle_grows_with_trial_count(n):
    prev = expected_max_sharpe(n // 2 or 2, 0.01)
    assert expected_max_sharpe(n, 0.01) >= prev


def test_luck_hurdle_is_zero_when_ungameable():
    assert expected_max_sharpe(1, 0.01) == 0.0     # a single trial cannot be cherry-picked
    assert expected_max_sharpe(500, 0.0) == 0.0    # identical strategies -> no dispersion to exploit


def test_psr_punishes_negative_skew():
    """Two series, same mean and volatility; one has a crash baked in."""
    rng = np.random.default_rng(11)
    base = rng.normal(0.001, 0.01, size=1000)

    skewed = base.copy()
    skewed[500] = -0.12                      # one bad day
    skewed = skewed - (skewed.mean() - base.mean())
    skewed = skewed * (base.std(ddof=1) / skewed.std(ddof=1))
    skewed = skewed - (skewed.mean() - base.mean())

    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(base)


def test_min_track_record_is_infinite_below_the_hurdle():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=500)
    assert min_track_record_length(r, benchmark_sr=0.5) == float("inf")


def test_dsr_is_a_probability():
    rng = np.random.default_rng(5)
    for mu in (-0.002, 0.0, 0.002):
        d = deflated_sharpe_ratio(rng.normal(mu, 0.01, size=800), 50, 0.01)
        assert 0.0 <= d <= 1.0


# --------------------------------------------------------------------------- #
#  PBO
# --------------------------------------------------------------------------- #


def test_pbo_on_pure_noise_is_near_coin_flip():
    rng = np.random.default_rng(42)
    res = pbo_cscv(_noise(rng, 1000, 20), n_partitions=10)

    assert 0.30 < res.pbo < 0.70, f"expected ~0.5 for noise, got {res.pbo}"
    assert res.n_splits_evaluated == 252  # C(10, 5)


def test_pbo_is_low_when_one_strategy_genuinely_wins():
    rng = np.random.default_rng(43)
    M = _noise(rng, 1000, 20)
    M[:, 7] = rng.normal(0.002, 0.01, size=1000)  # a real edge, hidden in the crowd

    res = pbo_cscv(M, n_partitions=10)

    assert res.pbo < 0.10, f"selection should transfer OOS, got PBO={res.pbo}"
    assert res.prob_oos_loss < 0.10
    assert res.median_oos_sharpe > 0
    # The same strategy wins essentially every split — that stability is the
    # real signature of skill, and it is what makes the negative OOS-vs-IS slope
    # here a complementary-partition artifact rather than a warning.
    assert res.selection_stability > 0.9
    assert res.performance_degradation < 0


def test_selection_churns_when_everything_is_noise():
    rng = np.random.default_rng(44)
    res = pbo_cscv(_noise(rng, 1000, 20), n_partitions=10)
    # No strategy can hold the crown, because there is nothing to hold it with.
    assert res.selection_stability < 0.5


def test_pbo_rejects_degenerate_inputs():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="at least 2 competing"):
        pbo_cscv(_noise(rng, 500, 1))
    with pytest.raises(ValueError, match="even"):
        pbo_cscv(_noise(rng, 500, 5), n_partitions=7)
    with pytest.raises(ValueError, match="observations"):
        pbo_cscv(_noise(rng, 10, 5), n_partitions=10)
