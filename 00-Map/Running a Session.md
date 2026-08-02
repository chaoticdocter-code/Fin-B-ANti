---
type: guide
updated: 2026-08-02
---

# Running a Session

Everything below is paper. No command here can place a real order —
see [[0001 Paper only and the live guard]].

Your window is **08:30–13:30 Pacific** (11:30–16:30 ET), ending half an hour
after the US close so the last act runs on final bars.

## Every day — the one that matters

```bash
cd "D:\Fin B\engine"; uv run finb session
```

Takes about a minute. It snapshots the tradeable universe, refreshes bars,
health-checks the data, reads the account, and writes today's note into
`40-Sessions`.

**Run it even on days you do nothing else.** The universe snapshot is the only
part of this project that cannot be recovered later — a missed day is a
permanent hole in the survivorship record, and no vendor sells it back to you.

## Reading the output

| Line | What to look at |
|---|---|
| `universe` | Should say "assets captured". If it failed, that day is gone. |
| `data … mean completeness` | Below ~95% means gaps are being trained on. |
| `budget $500.00 of $95,468.36` | Ours vs the broker's. See [[0011 The broker balance is not the budget]]. |
| `ledger: N trials` | Rises every session. This is the cost of looking. |
| `risk:` | `HALTED` means the kill switch fired and needs a written reason to clear. |

Warnings are printed but not fatal. A stablecoin showing low completeness is
expected; a major showing it is not.

## The session note

Each run writes `40-Sessions/YYYY-MM-DD Session.md`. Anything you type outside
the `finb:begin`/`finb:end` markers survives every regeneration, so write freely
under **My notes** — that section is yours permanently.

## Other commands

```bash
uv run finb doctor
```
Credentials, paths, and safety posture. Run after editing `.env`.

```bash
uv run finb costs
```
Round-trip cost and the minimum holding period per venue. This is the
feasibility map — check it before designing anything.

```bash
uv run finb account
```
Read-only account and positions.

```bash
uv run finb map
```
Regenerates [[System Map]] from the code's real state. Green means built *and*
tested.

## Experiments

```bash
uv run python scripts/baseline.py
uv run python scripts/ml_baseline.py
```

Both currently produce honest negatives — see [[2026-08-01 Baseline]] and
[[2026-08-01 ML baseline]]. They are the reference implementations to copy when
testing a new idea.

## Before you get attached to a result

Three questions, in order:

1. **How many things did I try?** Not "how many did I keep". The
   [[Search Ledger]] knows; it counts hypotheses and sessions, not just code.
2. **Does it beat the null cohort?** Theory can be argued with. Two hundred
   block-bootstrapped zero-skill strategies from the same market cannot.
3. **Does raising the confidence threshold improve gross edge?** If not, there
   is no edge to tighten toward — this is the cheapest diagnostic available and
   it caught the ML baseline immediately.

> The expected outcome of any given experiment is a negative. That is not the
> project failing; it is the only setting in which a positive would mean
> anything.
