---
type: decision
id: "0006"
status: accepted
date: 2026-07-31
amends: "0005"
---

# 0006 — Multi-day horizons, and breadth instead of frequency

## Context

[[0005 Crypto takes intraday equities take multi-day]] concluded that crypto
should carry the intraday work, because PDT rules it out for equities. That was
right about PDT and wrong about crypto, because it reasoned about regulation
without reasoning about cost.

Modelled in `finb.sim.costs`, round-trip cost on $500:

| Venue | Taker | Maker | Move needed to break even |
|---|---|---|---|
| alpaca-equity | 3.3 bps | −1.7 bps | 0.033% |
| binance-spot | 24.0 bps | 18.0 bps | 0.240% |
| alpaca-crypto | 57.0 bps | 45.0 bps | 0.570% |
| kraken-spot | 57.0 bps | 49.0 bps | 0.570% |
| coinbase-advanced | 125.0 bps | 117.0 bps | 1.250% |

Crypto costs **17×** what equities cost per round trip at Alpaca, and **38×** at
Coinbase. The venue with no frequency limit is the venue that charges most for
using it.

Set that against a typical crypto daily move of ~200 bps: a single round trip
consumes **~28% of a whole day's range**. Trading 20 times a day requires a
57 bps net edge on every trade. That is not a modelling challenge — it is
arithmetic, and it does not have a solution.

## Decision

**Multi-day holding periods on both venues.** No intraday strategies.

Cost is fixed per round trip while the expected move grows with √time. The same
57 bps that eats 28% of a one-day move eats under 9% of a ten-day move.

**And: buy observations with breadth, not frequency.** Twenty crypto pairs held
for days produces the same number of labelled observations as one pair traded
intraday, at a twentieth of the fee drag.

## Consequences

- Bar frequency for *features* stays high (hourly or better). Only the *holding
  period* lengthens. Feature resolution and trade frequency are separate
  choices, and conflating them is what makes fee drag invisible.
- Triple-barrier `max_bars` should be set in days, not minutes.
- This works against us on one axis and it should be said plainly: fewer trades
  per year means fewer observations, and the [[Luck Hurdle]] scales as
  √(log N / T). Breadth is what repays that — a cross-section of 20 instruments
  recovers the sample size that lower frequency gave up.
- Maker orders are worth real money in crypto (57 → 45 bps) but only when they
  fill. The simulator must not assume a resting order always fills, or it will
  hand back the entire saving as fiction.
- The equity leg is nearly free at 3.3 bps and PDT does not bind at multi-day
  horizons. Equities are the *cheap* venue here, which inverts the intuition
  that crypto is the natural home for a small account.

## Status of 0005

Amended, not withdrawn. Its analysis of PDT and settlement stands and is
enforced in `finb.sim.constraints`. Only its conclusion about intraday crypto is
replaced by this record.
