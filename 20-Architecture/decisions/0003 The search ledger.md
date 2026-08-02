---
type: decision
id: "0003"
status: accepted
date: 2026-07-30
---

# 0003 — The search ledger

## Context

The [[Luck Hurdle]] depends on how many variants were tried. For a farm that
number is obvious: count the bots. For one bot evolving over months it is
invisible — and nobody remembers that sessions 1–39 quietly tried 800 variants
before session 40 produced something that looked good.

Undercounting trials is the single easiest way to promote noise. It requires no
mistake in the maths, only a lapse in memory.

## Decision

An append-only JSONL ledger at `data/artifacts/`, recording every variant ever
evaluated: id, timestamp, per-period Sharpe, observation count, genome hash.
It survives restarts and only grows.

`PromotionGate.evaluate_ledger(returns, ledger)` reads the trial count from the
ledger rather than asking the caller to supply it — removing the one input most
likely to be wrong.

## Consequences

- Deleting the ledger destroys evidence; it does not reset the odds. Treat it
  as the most valuable file in the project.
- With fewer than 10 trials the dispersion estimate falls back to the
  theoretical null (1/T) rather than an empirical variance from three samples,
  which would understate the hurdle and wave through noise.
- A torn final line from an interrupted write costs one trial, not the history.

## Verification

400 variants of pure noise, best one selected: apparent annualised Sharpe above
1.5, deflated Sharpe below 0.95, rejected. Test:
`tests/test_ledger.py::test_ledger_drives_the_gate_without_the_caller_tracking_trials`.
