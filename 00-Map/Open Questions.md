---
type: map
status: living
updated: 2026-07-31
---

# Open Questions

Things we do not know or have not built, in the order they will hurt.

## Urgent — gets worse every day we wait

- [ ] **Point-in-time universe snapshots.** Start writing `/v2/assets` (equities
      and crypto pairs) to dated Parquet **today**. Roughly twenty lines. Every
      day of delay is survivorship bias that can never be reconstructed —
      backfilling today's surviving symbols into last year is not a bias, it is
      a fabrication. Highest urgency-per-line in the project.

## Needed before the first strategy means anything

- [ ] **A permanent null control cohort.** ~200 label-shuffled strategies flowing
      through the identical pipeline, forever. The champion must beat the *best
      of the null cohort*, not a threshold. Empirical, cheap, and immune to our
      distributional assumptions. The [[Red Team Review]] calls this the
      highest-value statistical component available and it is not built.
- [ ] **Recalibrate the gate on real data.** Our 0.1% false-promotion rate came
      from IID synthetic nulls. Real strategies have autocorrelation, fat tails,
      and shared factor exposure, all of which inflate it. Re-run against
      block-bootstrapped real returns with shuffled labels. Expect 1–5%. See
      [[Luck Hurdle]].
- [ ] **A sealed forward holdout** with a pre-registered unlock date and a query
      budget that decrements in the [[Search Ledger]]. [[0009 Looking at P&L is a logged trial]]
      makes the trial count honest but does not solve sequential adaptivity —
      only unseen data does.
- [ ] **The leakage test suite as CI**: truncation-equality, extra-shift, and
      future-noise tests. ~200 lines, permanently closes the bug classes most
      likely to manufacture a fake winner. See [[Leakage]].

## Surfaced by the first live run

- [ ] **A fully-invested book cannot rebalance under a 38-day minimum hold.**
      Observed live: with $499 of $500 committed and every hold unexpired, a
      signal change produced zero executable orders. The holding period is a
      liquidity constraint, not only a cost rule. Options: stagger entries so
      holds expire in a rolling fashion, keep a cash buffer, or accept that the
      rebalance cadence *is* the holding period and score accordingly. Needs
      designing rather than patching — see [[2026-08-02 First live bot run]].
- [ ] **Long-only cross-sectional momentum buys negative-momentum assets** when
      the whole universe is falling — two of the first four picks had negative
      scores. Decide whether an absolute-momentum filter (hold cash when nothing
      is rising) belongs in the strategy, and remember that adding it is a trial.

## Decisions owed

- [ ] **Train/serve feed policy, in writing, before any model is fit.** Either
      train on IEX history and infer on IEX, or train on 15-minute-delayed SIP
      and infer on delayed SIP. Mixing them is train/serve skew and the research
      flags it as the single most likely silent killer. Pick one.
- [ ] **The 5-hour session window and the operator's timezone.** Still owed by
      me — I said I would propose one from the research rather than ask.
- [ ] **The bot's first falsifiable hypothesis.** Not "predict the market." One
      specific edge, one instrument set, one horizon. Cross-sectional momentum on
      a 20-pair crypto cross-section at a 38-day hold is the current front-runner
      because [[0008 Crypto stays at cost-derived multi-week holds]] forces
      breadth anyway.

## Settled, with reasoning worth revisiting later

- **Obsidian as the interface.** The review recommends cutting the Canvas map and
  Bases dashboards as "the project becoming its own documentation." Kept: it was
  explicitly requested, it is already built, and the generated map reads its
  state from the code so it cannot drift into lying. Revisit if it ever starts
  consuming time that belongs to the scorer.
- **Crypto stays**, at 38-day holds with mandatory breadth —
  [[0008 Crypto stays at cost-derived multi-week holds]].
- **Daily P&L watching stays**, at a measured 32% cost to the luck hurdle —
  [[0009 Looking at P&L is a logged trial]].

## Answered by the research

- ~~Does Alpaca's free IEX feed carry enough of the tape?~~ **No.** 2.36% of
  trades. It biases every volume feature and understates range volatility.
- ~~Alpaca's paper fill model?~~ Fills beyond displayed NBBO size, random 10%
  partials, no queue, no impact. Our own cost model is the scorer.
- ~~Does PDT make intraday equities unworkable at $500?~~ **PDT no longer
  exists** — [[0007 The PDT rule was repealed]].
