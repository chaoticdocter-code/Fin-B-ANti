---
type: decision
id: "0005"
status: amended
amended_by: "0006"
date: 2026-07-30
---

> [!warning] Amended by [[0006 Multi-day horizons and breadth instead of frequency]]
> The PDT and settlement analysis below is correct and is enforced in code.
> The conclusion that **crypto should carry intraday trading is wrong** — a
> round trip there costs 57 bps, about 28% of a typical daily move. Costs rule
> out intraday on both venues, not just equities.

# 0005 — Crypto takes intraday, equities take multi-day

## Context

You are a US tax resident, so FINRA's Pattern Day Trader rule applies. Now
modelled and tested in `finb.sim.constraints`. What it actually permits at $500:

| Venue / account | Round-trip capacity |
|---|---|
| Equities, **margin** | **3 day trades per rolling 5 business days.** A 4th flags the account and demands $25,000 equity. |
| Equities, **cash** | No PDT — but T+1 settlement means ~1 full-balance round trip per day, and reusing unsettled proceeds risks a good-faith violation. 3 of those in 12 months = 90-day restriction. |
| **Crypto** | Unlimited. No PDT, instant settlement, 24/7. |

Three day trades per week is roughly 0.6 per day. No intraday equity strategy
survives that — not because the edge is absent, but because the account is not
allowed to act on it.

The rolling window is also longer in wall-clock terms than it sounds. A day
trade on Wed 14 Jan 2026 does not leave the window until Thu 22 Jan, because
the weekend and MLK Day do not count. Eight calendar days, from five business
days.

## Decision

Split the strategy space by venue:

- **Crypto — intraday.** Where the bot's fast-horizon work happens. This is the
  only place a $500 account can trade often enough to accumulate the
  observations that the [[Luck Hurdle]] demands.
- **US equities — multi-day.** Overnight-and-longer horizons only, where PDT
  simply does not bite. Signals evaluated once per session rather than
  continuously.

## Consequences

- **This is good news for statistics, not just compliance.** The hurdle scales
  as √(log N / T). Crypto's 24/7 clock generates observations several times
  faster than an equity session, so the same calendar time buys a much lower
  hurdle.
- The equity leg will accumulate evidence slowly. Expect it to sit below the
  gate for a long time, and do not "fix" that by shortening its evaluation
  window.
- The simulator must enforce these rules rather than merely report them, or the
  bot will evolve straight into strategies it cannot legally run.
- Crypto's fee structure is much worse than commission-free equities. The
  freedom to trade often is not the same as the ability to profit from it —
  see [[Execution Realism]] once costs are modelled.

## Open

- Whether to run the equity leg as cash or margin. Cash removes PDT entirely at
  the cost of settlement friction; margin keeps 3 day trades in reserve. Leaning
  cash, pending the research brief on Alpaca's account options.
