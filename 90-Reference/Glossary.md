---
type: reference
updated: 2026-07-30
---

# Glossary

Terms used across the vault. Kept short on purpose.

## Statistics

**[[Luck Hurdle]]** (SR*) — the Sharpe pure chance produces given how many
variants you tried.

**Deflated Sharpe Ratio (DSR)** — probability the true Sharpe beats the luck
hurdle. Our promotion threshold is 0.95.

**Probabilistic Sharpe Ratio (PSR)** — same idea against a benchmark of zero.
Penalises negative skew and fat tails, which is why a strategy that grinds out
gains and occasionally blows up scores worse than its point estimate suggests.

**PBO** — Probability of Backtest Overfitting. How often the in-sample winner
lands in the bottom half out-of-sample. Above ~0.5 your selection process is
worse than random.

**Minimum track record length** — how many observations before a Sharpe is
distinguishable from the hurdle. Often embarrassingly large.

**Purged / embargoed CV** — cross-validation that removes training samples
overlapping the test window. Ordinary k-fold leaks the future into the past on
financial data and inflates every score.

## Mechanics

**PDT** — Pattern Day Trader. FINRA rule: 4+ day trades in 5 business days
flags an account, requiring $25,000 equity. Applies to US equities, **not** to
crypto. Binding at $500. See [[Open Questions]].

**T+1** — US equity trade settlement, one business day. In a cash account
unsettled proceeds cannot be reused without a good-faith violation, which caps
how many round trips $500 can do per week.

**Notional order** — an order in dollars rather than shares. What makes a $500
account able to hold more than one position.

**Shadow book** — a fully simulated $500 account used to evaluate a variant
without touching the broker. See [[0004 Full book per variant not split capital]].

**Slippage** — the gap between the price you modelled and the price you got.
At $500 the dominant cost, and the easiest thing to under-model into a
profitable-looking backtest.

## This project

**Variant** — one version of the bot: a specific genome, model, and parameter
set. Every one ever evaluated is counted in the [[Search Ledger]].

**Genome** — the parameterisation the bot evolves over.

**Promotion** — a variant clearing the gate and becoming the production bot.
Rare by design.

**The gate** — `finb.evaluation.gate`. The thing standing between a
good-looking variant and your capital.
