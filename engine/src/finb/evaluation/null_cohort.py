"""A permanent control arm: strategies with the signal destroyed.

Every statistical gate rests on assumptions about what noise looks like. Ours —
the Deflated Sharpe Ratio — assumes trials are roughly independent draws from a
fixed distribution. Real trading strategies violate that comprehensively: their
returns are autocorrelated, fat-tailed, and share factor exposure with each
other. When trials are correlated, the measured dispersion of Sharpe across them
*understates* the true spread, which lowers the luck hurdle, which lets noise
through.

The fix is to stop assuming and start measuring. Run a cohort of strategies with
the signal deliberately destroyed — but with every other statistical property of
the real data preserved — through the *identical* pipeline. Whatever the
pipeline reports about them is the noise floor. A champion must beat the best of
that cohort, not a threshold derived from theory.

Two ways to destroy signal while preserving structure:

- **Block bootstrap** resamples contiguous blocks, so autocorrelation, volatility
  clustering and fat tails survive while any relationship to a *specific* point
  in time is destroyed. Demeaning afterwards makes the expected Sharpe zero.
- **Label shuffling** permutes the targets, destroying the feature-label
  relationship while leaving both marginal distributions untouched.

This is empirical, cheap, and immune to our distributional assumptions. It is
the closest thing available to an honest answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from finb.evaluation.gate import PromotionGate, sharpe_ratio


def optimal_block_size(returns: np.ndarray) -> int:
    """A serviceable block length for the circular block bootstrap.

    Uses the n^(1/3) rule, widened when returns are strongly autocorrelated —
    blocks must be long enough to carry the dependence structure, or the
    bootstrap quietly produces something closer to IID noise and we are back to
    the assumption we were trying to escape.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        return max(1, n // 2)

    base = max(2, int(round(n ** (1 / 3))))
    if r.std(ddof=1) == 0:
        return base

    z = (r - r.mean()) / r.std(ddof=1)
    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    if not np.isfinite(rho):
        return base

    # Larger |rho| -> longer blocks, capped so blocks stay short vs the sample.
    scale = 1.0 / max(1e-3, 1.0 - abs(rho))
    return int(min(max(base, base * math.sqrt(scale)), max(2, n // 10)))


def circular_block_bootstrap(
    returns: np.ndarray,
    rng: np.random.Generator,
    *,
    block_size: int | None = None,
    length: int | None = None,
) -> np.ndarray:
    """One resample preserving serial dependence.

    Circular (wrapping at the end) so every observation has equal probability of
    appearing, rather than the start and end being under-sampled.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        raise ValueError("returns is empty")

    L = block_size or optimal_block_size(r)
    L = max(1, min(L, n))
    out_len = length or n
    n_blocks = int(np.ceil(out_len / L))

    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(L)[None, :]) % n
    return r[idx.ravel()[:out_len]]


def build_null_cohort(
    returns: np.ndarray,
    *,
    size: int = 200,
    rng: np.random.Generator | None = None,
    block_size: int | None = None,
    demean: bool = True,
) -> np.ndarray:
    """A (T, size) matrix of zero-skill strategies drawn from real data.

    `demean=True` subtracts the **source series'** mean, so the cohort has zero
    *expected* return while each individual resample still has its own realised
    drift. That residual variation is the entire point — it is what one lucky
    strategy looks like, and selecting the best of it is what we are trying to
    measure.

    Subtracting each column's *own* mean instead would force every null Sharpe
    to exactly zero, collapsing the dispersion to nothing and making the
    calibration silently report a 0% false-promotion rate.
    """
    rng = rng or np.random.default_rng()
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    L = block_size or optimal_block_size(r)
    drift = r.mean() if demean else 0.0

    cols = []
    for _ in range(size):
        cols.append(circular_block_bootstrap(r, rng, block_size=L) - drift)
    return np.column_stack(cols)


def build_correlated_null_cohort(
    returns: np.ndarray,
    *,
    size: int = 200,
    correlation: float = 0.0,
    rng: np.random.Generator | None = None,
    block_size: int | None = None,
) -> np.ndarray:
    """A null cohort whose members are correlated with each other.

    This is what a real search population looks like. A hundred variants of one
    strategy, all trading the same instruments with similar signals, are not a
    hundred independent trials — they share a common component and differ only
    at the margin. Pairwise correlations of 0.6-0.9 are ordinary.

    Constructed as ``sqrt(rho) * common + sqrt(1-rho) * idiosyncratic``, so every
    pair has correlation approximately `rho` while each column keeps the source
    series' volatility, autocorrelation and tails.

    This matters because the deflated Sharpe assumes roughly independent trials.
    Correlated trials shrink the observed dispersion of Sharpe, which lowers the
    luck hurdle — the mechanism by which a gate that looks calibrated on
    independent nulls can wave noise through on a real population.
    """
    rng = rng or np.random.default_rng()
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    L = block_size or optimal_block_size(r)
    drift = r.mean()
    rho = float(np.clip(correlation, 0.0, 1.0))

    common = circular_block_bootstrap(r, rng, block_size=L) - drift
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)

    cols = []
    for _ in range(size):
        idio = circular_block_bootstrap(r, rng, block_size=L) - drift
        cols.append(a * common + b * idio)
    return np.column_stack(cols)


def mean_pairwise_correlation(cohort: np.ndarray) -> float:
    """Average off-diagonal correlation of a (T, N) return matrix.

    Measure this on your actual variant population and feed it to the gate.
    Assuming independence is the assumption that breaks.
    """
    m = np.asarray(cohort, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return 0.0
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(m.T)
    off = c[np.triu_indices_from(c, k=1)]
    off = off[np.isfinite(off)]
    return float(off.mean()) if off.size else 0.0


def shuffle_labels(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute targets, destroying the feature-label link but not the marginals."""
    y = np.asarray(y)
    out = y.copy()
    rng.shuffle(out)
    return out


@dataclass(frozen=True, slots=True)
class NullVerdict:
    champion: float
    null_best: float
    null_p95: float
    null_median: float
    empirical_p_value: float
    """Fraction of the null cohort matching or beating the champion. This is the
    number to report — it makes no distributional assumptions at all."""

    beats_best_of_null: bool
    n_null: int

    @property
    def summary(self) -> str:
        verdict = "clears" if self.beats_best_of_null else "does NOT clear"
        return (
            f"champion Sharpe {self.champion:.3f} {verdict} the best of "
            f"{self.n_null} null strategies ({self.null_best:.3f}); "
            f"empirical p = {self.empirical_p_value:.4f}"
        )


@dataclass(frozen=True, slots=True)
class NullCohort:
    """The control arm. Keep one running permanently, alongside the real search."""

    size: int = 200
    seed: int = 0

    def judge(
        self,
        champion_returns: np.ndarray,
        market_returns: np.ndarray,
        *,
        periods_per_year: int = 252,
    ) -> NullVerdict:
        """Compare a champion against a cohort built from the same market.

        `market_returns` should be the return series the strategy traded on —
        the cohort must inherit *its* volatility clustering and tails, not some
        generic noise.
        """
        rng = np.random.default_rng(self.seed)
        cohort = build_null_cohort(market_returns, size=self.size, rng=rng)

        null_sr = np.array(
            [sharpe_ratio(cohort[:, j], periods_per_year) for j in range(cohort.shape[1])]
        )
        champ = sharpe_ratio(champion_returns, periods_per_year)

        return NullVerdict(
            champion=champ,
            null_best=float(np.max(null_sr)),
            null_p95=float(np.percentile(null_sr, 95)),
            null_median=float(np.median(null_sr)),
            empirical_p_value=float(np.mean(null_sr >= champ)),
            beats_best_of_null=bool(champ > np.max(null_sr)),
            n_null=int(cohort.shape[1]),
        )


def calibrate_gate(
    market_returns: np.ndarray,
    *,
    gate: PromotionGate | None = None,
    n_trials: int = 100,
    n_repeats: int = 200,
    seed: int = 0,
    periods_per_year: int = 252,
    correlation: float = 0.0,
) -> dict:
    """Measure the gate's real false-promotion rate on realistic null data.

    Repeats the whole search-and-select procedure on strategies that are
    guaranteed to have no skill but *do* have real market autocorrelation and
    tails, then counts how often the gate promotes the winner.

    Our synthetic-IID calibration gave 0.1%. The red-team review predicted 1-5%
    on realistic data. This function is how that gets settled with evidence
    rather than argument.
    """
    gate = gate or PromotionGate(periods_per_year=periods_per_year)
    rng = np.random.default_rng(seed)
    r = np.asarray(market_returns, dtype=float)
    r = r[np.isfinite(r)]
    L = optimal_block_size(r)

    promoted = 0
    dsrs, winners = [], []

    for _ in range(n_repeats):
        cohort = build_correlated_null_cohort(
            r, size=n_trials, rng=rng, block_size=L, correlation=correlation
        )
        sr = np.array([sharpe_ratio(cohort[:, j]) for j in range(n_trials)])
        best = int(np.argmax(sr))

        # Measure the population's correlation rather than assuming it, exactly
        # as a live system would.
        v = gate.evaluate(
            cohort[:, best],
            n_trials=n_trials,
            sr_variance=float(sr.var(ddof=1)),
            trial_correlation=mean_pairwise_correlation(cohort),
        )
        promoted += v.passed
        dsrs.append(v.dsr)
        winners.append(v.sharpe_annual)

    return {
        "false_promotion_rate": promoted / n_repeats,
        "n_repeats": n_repeats,
        "n_trials_per_repeat": n_trials,
        "correlation": correlation,
        "block_size": L,
        "median_dsr": float(np.median(dsrs)),
        "median_winner_sharpe_annual": float(np.median(winners)),
        "max_winner_sharpe_annual": float(np.max(winners)),
    }
