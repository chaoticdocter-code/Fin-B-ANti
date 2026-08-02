"""Selection statistics: what a strategy's track record is worth once you
account for how many strategies you tried.

Implements four things from the Bailey / López de Prado line of work:

- **PSR**  — Probabilistic Sharpe Ratio. Probability the true Sharpe exceeds a
  benchmark, correcting for skew, fat tails, and short samples.
- **SR\\***  — the Sharpe you should *expect* the best of N independent random
  strategies to show, purely by luck. This is the hurdle a farm must clear.
- **DSR**  — Deflated Sharpe Ratio: PSR measured against SR\\* instead of zero.
- **PBO**  — Probability of Backtest Overfitting via Combinatorially Symmetric
  Cross-Validation: how often the in-sample winner lands in the bottom half
  out-of-sample.

References
----------
Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier", J. Risk.
Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt.
Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest
Overfitting", J. Computational Finance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from finb.evaluation.ledger import SearchLedger

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
#  Basic moments
# --------------------------------------------------------------------------- #


def sharpe_ratio(returns: np.ndarray, periods_per_year: int | None = None) -> float:
    """Sharpe of a return series.

    Returns the **per-period** Sharpe by default. Pass `periods_per_year` to
    annualise. The deflation functions below all expect the per-period value,
    because they combine it with the per-period skew and kurtosis.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = float(r.mean() / sd)
    return sr * math.sqrt(periods_per_year) if periods_per_year else sr


def _moments(returns: np.ndarray) -> tuple[int, float, float, float]:
    """(n, per-period sharpe, skew, kurtosis) — kurtosis is non-excess."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return n, 0.0, 0.0, 3.0
    sd = r.std(ddof=1)
    if sd == 0:
        return n, 0.0, 0.0, 3.0
    z = (r - r.mean()) / sd
    skew = float((z**3).mean())
    kurt = float((z**4).mean())  # normal == 3.0
    return n, float(r.mean() / sd), skew, kurt


def _psr_denominator(sr: float, skew: float, kurt: float) -> float:
    """sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) — the standard error scaling.

    Negative skew and fat tails inflate this, which is exactly right: a strategy
    that grinds out small gains and occasionally blows up has a *less* reliable
    Sharpe than its point estimate suggests.
    """
    var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    return math.sqrt(max(var, 1e-12))


# --------------------------------------------------------------------------- #
#  Probabilistic / Deflated Sharpe
# --------------------------------------------------------------------------- #


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > `benchmark_sr`), in [0, 1]."""
    n, sr, skew, kurt = _moments(returns)
    if n < 3:
        return 0.0
    stat = (sr - benchmark_sr) * math.sqrt(n - 1) / _psr_denominator(sr, skew, kurt)
    return float(norm.cdf(stat))


def effective_n_trials(n_trials: int, correlation: float) -> float:
    """Independent-equivalent trial count for a correlated population.

    ``N_eff = rho + (1 - rho) * N``. A hundred variants correlated at 0.9 carry
    the information of about eleven independent ones.

    This cuts both ways and both errors are real: charging the raw count
    over-penalises a population of near-duplicates, while charging N=1 never
    penalises anything.
    """
    rho = float(min(max(correlation, 0.0), 1.0))
    return rho + (1.0 - rho) * n_trials


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """SR\\* — expected maximum per-period Sharpe across `n_trials` null strategies.

    `sr_variance` is the *variance of the Sharpe ratios across the trials you
    ran*. Estimate it from the farm's own leaderboard; do not guess it.

    This grows with the number of trials, which is the whole point: testing more
    strategies raises the bar that any of them must clear.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - g) * a + g * b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sr_variance: float,
) -> float:
    """P(true Sharpe > SR\\*) — the probability this strategy is not just the
    luckiest of `n_trials`.

    Read it as a confidence level. Below ~0.95 you have not demonstrated skill.
    """
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, sr_variance))


def min_track_record_length(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before the Sharpe is distinguishable from `benchmark_sr`.

    Returns ``inf`` when the observed Sharpe is already below the benchmark — no
    amount of further data would help.
    """
    n, sr, skew, kurt = _moments(returns)
    if n < 3 or sr <= benchmark_sr:
        return math.inf
    z = norm.ppf(confidence)
    return float(1.0 + (_psr_denominator(sr, skew, kurt) ** 2) * (z / (sr - benchmark_sr)) ** 2)


# --------------------------------------------------------------------------- #
#  Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Outcome of a combinatorially symmetric cross-validation sweep."""

    pbo: float
    """Fraction of splits where the in-sample winner ranked below median OOS.
    Above ~0.5 means your selection process is worse than random."""

    n_splits_evaluated: int
    logits: np.ndarray
    oos_performance: np.ndarray
    """OOS metric of the IS-selected strategy, per split."""

    is_performance: np.ndarray
    performance_degradation: float
    """OLS slope of OOS Sharpe on IS Sharpe across splits, for whichever
    strategy was selected.

    Only meaningful when read together with `pbo`. If selection is unstable —
    a different strategy winning each split, which is the overfitting signature
    — a negative slope means in-sample strength actively predicts out-of-sample
    weakness. But if one strategy dominates every split, this slope goes
    negative *mechanically*: IS and OOS are complementary halves of a single
    series, so blocks that flatter one starve the other. A negative slope
    alongside a near-zero PBO is that artifact, not a warning."""

    prob_oos_loss: float
    """Fraction of splits where the selected strategy lost money OOS."""

    median_oos_sharpe: float
    """Median OOS Sharpe of the selected strategy. The blunt question — did
    picking the in-sample winner actually make money afterwards?"""

    selection_stability: float
    """Fraction of splits won by the single most-frequently-selected strategy.
    Near 1/N means selection is churning through the population at random."""


def pbo_cscv(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
    max_combinations: int | None = 5000,
) -> PBOResult:
    """Run CSCV over a (T observations x N strategies) matrix of returns.

    Splits the timeline into `n_partitions` contiguous blocks, then for every way
    of choosing half of them as in-sample, picks the best strategy in-sample and
    measures where it ranks out-of-sample. If selection has no skill, that rank
    is uniform and PBO tends to 0.5.

    `n_partitions` must be even. C(16, 8) = 12,870 splits; `max_combinations`
    subsamples deterministically when that is too many.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T observations x N strategies)")
    if n_partitions % 2 != 0:
        raise ValueError("n_partitions must be even")

    T, N = M.shape
    if N < 2:
        raise ValueError("PBO needs at least 2 competing strategies")
    if n_partitions * 2 > T:
        raise ValueError(
            f"need at least {n_partitions * 2} observations for {n_partitions} partitions, got {T}"
        )

    block = T // n_partitions
    blocks = [M[i * block : (i + 1) * block] for i in range(n_partitions)]

    all_combos = list(combinations(range(n_partitions), n_partitions // 2))
    if max_combinations is not None and len(all_combos) > max_combinations:
        step = len(all_combos) / max_combinations
        all_combos = [all_combos[int(i * step)] for i in range(max_combinations)]

    logits, oos_perf, is_perf, winners = [], [], [], []

    for combo in all_combos:
        is_idx = set(combo)
        is_data = np.vstack([blocks[i] for i in range(n_partitions) if i in is_idx])
        oos_data = np.vstack([blocks[i] for i in range(n_partitions) if i not in is_idx])

        is_sr = np.array([sharpe_ratio(is_data[:, j]) for j in range(N)])
        oos_sr = np.array([sharpe_ratio(oos_data[:, j]) for j in range(N)])

        best = int(np.nanargmax(is_sr))

        # Relative rank of the chosen strategy OOS, in (0, 1).
        rank = float(np.sum(oos_sr <= oos_sr[best]))
        omega = rank / (N + 1.0)
        omega = min(max(omega, 1e-9), 1 - 1e-9)

        logits.append(math.log(omega / (1.0 - omega)))
        oos_perf.append(float(oos_sr[best]))
        is_perf.append(float(is_sr[best]))
        winners.append(best)

    lam = np.array(logits)
    oos_arr = np.array(oos_perf)
    is_arr = np.array(is_perf)

    # Degradation: does in-sample strength carry over at all?
    slope = float(np.polyfit(is_arr, oos_arr, 1)[0]) if np.ptp(is_arr) > 0 else 0.0

    counts = np.bincount(np.array(winners), minlength=N)

    return PBOResult(
        pbo=float(np.mean(lam <= 0.0)),
        n_splits_evaluated=len(all_combos),
        logits=lam,
        oos_performance=oos_arr,
        is_performance=is_arr,
        performance_degradation=slope,
        prob_oos_loss=float(np.mean(oos_arr <= 0.0)),
        median_oos_sharpe=float(np.median(oos_arr)),
        selection_stability=float(counts.max() / len(winners)),
    )


# --------------------------------------------------------------------------- #
#  The gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GateVerdict:
    passed: bool
    reasons: list[str]
    dsr: float
    psr: float
    sharpe_annual: float
    expected_max_sr_annual: float
    min_track_record: float
    n_observations: int


@dataclass(frozen=True, slots=True)
class PromotionGate:
    """The bar the bot must clear before a change is treated as an improvement.

    Defaults are deliberately unkind. The bot is evolving against a fixed
    history, so given enough attempts it will eventually produce a variant that
    looks excellent for reasons that have nothing to do with markets. This is
    what stands between that variant and your capital.
    """

    min_dsr: float = 0.95
    min_observations: int = 100
    max_pbo: float = 0.30
    periods_per_year: int = 252

    def evaluate(
        self,
        returns: np.ndarray,
        *,
        n_trials: int,
        sr_variance: float,
        trial_correlation: float = 0.0,
    ) -> GateVerdict:
        """Judge one variant.

        `n_trials` must be the number of variants **ever evaluated across the
        bot's whole life** — including every discarded mutation from every past
        session. Undercounting it is the single easiest way to fool yourself
        here, which is why `evaluate_ledger` exists and should be preferred.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = r.size
        reasons: list[str] = []

        # Correlated trials shrink the observed dispersion of Sharpe toward zero,
        # which would drive SR* to zero and stop the deflation deflating — while
        # the *shared* lucky drift they all inherit remains. Measured on real BTC
        # returns, 100 trials correlated at 0.9 took the false-promotion rate
        # from 0.0% to 6.0%.
        #
        # Two corrections. The floor: a single Sharpe estimated from n
        # observations has sampling variance ~1/n under the null, and no amount
        # of correlation between strategies makes any one of them better
        # estimated than that. And the effective count: near-duplicate trials
        # should not be charged as independent ones.
        n_eff = effective_n_trials(n_trials, trial_correlation)
        variance = max(sr_variance, 1.0 / n) if n > 1 else sr_variance

        sr_star = expected_max_sharpe(n_eff, variance)
        dsr = probabilistic_sharpe_ratio(r, sr_star)
        psr = probabilistic_sharpe_ratio(r, 0.0)
        mtrl = min_track_record_length(r, sr_star)
        ann = sharpe_ratio(r, self.periods_per_year)
        k = math.sqrt(self.periods_per_year)

        if n < self.min_observations:
            reasons.append(f"only {n} observations, need {self.min_observations}")
        if dsr < self.min_dsr:
            eff = (
                f"{n_trials} trials"
                if trial_correlation <= 0
                else f"{n_trials} trials (~{n_eff:.0f} independent at rho={trial_correlation:.2f})"
            )
            reasons.append(
                f"deflated Sharpe {dsr:.3f} < {self.min_dsr:.2f} "
                f"(after {eff} the luck hurdle is SR* {sr_star * k:.2f} annualised)"
            )
        if math.isinf(mtrl):
            reasons.append("Sharpe does not exceed the luck hurdle at any sample size")
        elif mtrl > n:
            reasons.append(f"needs ~{mtrl:.0f} observations to be significant, has {n}")

        return GateVerdict(
            passed=not reasons,
            reasons=reasons,
            dsr=dsr,
            psr=psr,
            sharpe_annual=ann,
            expected_max_sr_annual=sr_star * k,
            min_track_record=mtrl,
            n_observations=n,
        )

    def evaluate_ledger(self, returns: np.ndarray, ledger: SearchLedger) -> GateVerdict:
        """Judge a variant, charging it for the bot's entire search history.

        Prefer this over `evaluate`. It reads the trial count from the ledger
        instead of asking you to remember it, which removes the only input you
        are likely to get wrong.
        """
        return self.evaluate(
            returns, n_trials=ledger.n_trials, sr_variance=ledger.sr_variance()
        )
