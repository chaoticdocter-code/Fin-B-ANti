---
type: decision
id: "0001"
status: accepted
date: 2026-07-30
---

# 0001 — Paper only, and the live guard

## Context

The project will eventually want a live path. Live trading code that exists
"but is switched off" is the standard way people lose money by accident: a
stray config, a copied notebook, a default that flipped.

## Decision

Two independent locks, both of which must open:

1. `FINB_ALLOW_LIVE` must equal the literal string
   `I_UNDERSTAND_THIS_TRADES_REAL_MONEY`. Nothing shorter, no booleans, no `1`.
   It cannot be set by a typo or a truthy default.
2. The variant must have passed the [[0003 The search ledger|gate]]. A human
   deciding to go live is not sufficient; the statistics must agree.

`Settings.assert_paper_only()` is called on every path that could reach a real
order book, and raises if lock 1 is open.

## Consequences

- Enabling live is deliberately annoying. That is the feature.
- `finb doctor` reports the safety posture first, before anything else.
- Paper results still need discounting — see [[Open Questions]] on Alpaca's
  fill model, which is more generous than a real book.
