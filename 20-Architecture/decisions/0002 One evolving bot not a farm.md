---
type: decision
id: "0002"
status: accepted
date: 2026-07-30
supersedes: farm-of-100-bots
---

# 0002 — One evolving bot, not a farm

## Context

The original plan was ~100 bots in an elimination bracket, culled until one
remained. Built and measured before being dropped, which is why the numbers
below are ours rather than borrowed.

**100 bots with provably zero skill, 60 days, $500 each:** the survivor turned
$500 into $661 (+32%), showing an annualised Sharpe of **5.07**. In 100% of
2,000 simulated tournaments the survivor "beat" Sharpe 2.0. In the following
period it was profitable **49.2%** of the time.

An elimination bracket always crowns someone. When the field is worthless it
crowns the luckiest coin — and hands you a track record that looks like genius.

## Decision

One bot that evolves continuously. No population, no bracket, no elimination.

## Consequences

- **The statistics do not get easier.** A bot that evolves is still searching;
  its trials are spread across sessions instead of across machines. The
  arithmetic is unchanged, only the bookkeeping is harder. Hence
  [[0003 The search ledger]].
- Simpler operationally: one config, one model, one equity curve to reason
  about during a supervised session.
- We give up the natural diversity a population provides. Mitigation: the bot's
  evolution must explore genuinely different hypotheses, not just re-tune the
  same one, or it will converge and stay converged.

## Measured, then deleted

The tournament and its simulation were removed from the codebase. What survived
is the part that mattered: the gate, and the ledger that feeds it.

Over a full year, per 2,000 simulated null tournaments:

| Rule | Promoted a zero-skill winner |
|---|---|
| "Crown the survivor" | 100% |
| Deflated-Sharpe gate | 0.1% |
