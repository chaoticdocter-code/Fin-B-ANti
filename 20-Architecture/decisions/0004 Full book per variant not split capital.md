---
type: decision
id: "0004"
status: accepted
date: 2026-07-30
---

# 0004 — Each variant gets the full $500, not a slice

## Context

When several variants are evaluated in parallel, the intuitive move is to split
the $500 between them. With even 20 variants that is $25 each; with 100, $5.

At $5 a variant cannot buy one share of most US equities, sits below minimum
notional on several venues, and pays fee and spread drag that swamps any edge.
Worse statistically: it generates too few trades for its Sharpe to mean
anything, so the evaluation is noise regardless of the strategy.

## Decision

Every variant is simulated against its own full **$500 shadow book**, in
parallel. Identical starting capital, directly comparable records, no
fragmentation.

Only the variant currently in production touches the actual Alpaca paper
account.

## Consequences

- Shadow books are simulation, not broker state — so the [[Execution Realism]]
  model (fees, spread, slippage, partial fills) is what makes them meaningful.
  A naive fill-at-mid simulator would make every variant look good.
- $500 remains the real constraint on the production variant: position sizing,
  minimum notional, and fee drag as a percentage of capital are all modelled
  against that number, not against the broker's $100k default.
