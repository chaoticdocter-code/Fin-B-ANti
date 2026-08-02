---
type: concept
aliases:
  - SR*
  - expected maximum Sharpe
  - deflated Sharpe
---

# Luck Hurdle

The Sharpe ratio you should **expect to see from pure chance**, given how many
variants you tried. Anything below it is not evidence of anything.

## Why it exists

Try one strategy and its Sharpe is an unbiased estimate of its skill. Try 400
and keep the best, and you are no longer measuring skill — you are measuring
the maximum of 400 draws from a distribution. That maximum is large even when
every draw has mean zero. Selection turns noise into an impressive number, and
it does so silently.

## The formula

$$SR^* = \sqrt{V[SR_n]}\;\Big[(1-\gamma)\,Z^{-1}\!\left(1-\tfrac{1}{N}\right) + \gamma\, Z^{-1}\!\left(1-\tfrac{1}{N e}\right)\Big]$$

where $N$ is the number of variants tried, $V[SR_n]$ is the variance of Sharpe
across those variants, and $\gamma \approx 0.5772$ is Euler–Mascheroni.

The **Deflated Sharpe Ratio** then asks: what is the probability the true Sharpe
exceeds $SR^*$, given this variant's observations, skew, and kurtosis? Below
0.95 and nothing has been demonstrated.

## What it means in practice

$SR^*$ scales roughly as $\sqrt{\log N / T}$ — sub-linear in the number of
trials, inverse-square-root in the length of the evaluation. Two consequences,
and the second is the useful one:

- **Trying more variants is cheap.** Going from 50 to 5,000 trials raises the
  bar by well under 3×. Search freely.
- **Evaluating for longer is powerful.** Doubling the evaluation window lowers
  the bar by ~1.41×, and unlike edge, time is free.

> The lever is patience, not restraint. Explore widely; judge slowly.

## Measured here

2,000 simulated years, 100 zero-skill strategies each, best one selected:

| Rule | Promotes the null winner |
|---|---|
| Highest return wins | 100% |
| Deflated Sharpe ≥ 0.95 | 0.1% |

Median deflated Sharpe for a null winner came out at **0.481** — almost exactly
the 0.5 a well-calibrated statistic should give.

## Where it lives

`engine/src/finb/evaluation/gate.py`. Fed by the [[Search Ledger]].
See [[0003 The search ledger]].

Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
