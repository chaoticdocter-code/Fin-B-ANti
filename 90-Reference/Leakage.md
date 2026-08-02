---
type: concept
aliases:
  - purging
  - embargo
  - overlapping labels
---

# Leakage

The second way this project could fool itself. The [[Luck Hurdle]] guards
against picking the luckiest variant; this guards against every variant's score
being inflated in the first place.

## The measured result

A pure random walk — 3,000 bars from a random number generator, no signal of any
kind. Labels: the sign of the 100-bar forward return. Features: price and its
200-bar moving average. Model: 1-nearest-neighbour.

| Cross-validation | Accuracy |
|---|---|
| Ordinary shuffled 5-fold | **0.955** |
| Purged + embargoed 5-fold | **0.495** |

Shuffled k-fold reports 95.5% accuracy on noise. That is not a subtle bias — it
is the difference between "we found something" and "we found nothing", and it is
the default behaviour of every tutorial.

## Why it happens

Two ordinary choices combine:

1. **Overlapping labels.** A label opened at bar 100 resolving at bar 200, and
   one opened at 101 resolving at 201, share 99% of their outcome window. They
   are not two observations. They are one observation counted twice.
2. **Smooth features.** Price and a moving average change slowly, so
   consecutive events sit almost on top of each other in feature space, and the
   pair is nearly unique per point in time.

Shuffle those across folds and the model never has to generalise. For each test
sample its temporal twin is sitting in the training set, carrying the same
features and the same answer. The model looks it up.

> Worth noting what *didn't* leak: trailing returns as features gave 0.514 vs
> 0.502 — no leak at all. Shifting a window of iid noise by one bar lands
> somewhere completely different, so no near-duplicates form. **Smoothness is
> the ingredient**, and smooth features are exactly what technical analysis
> produces.

## The fix

- **Purge** — drop any training sample whose label window overlaps the test
  window.
- **Embargo** — additionally drop a small block immediately after the test
  window, since serial correlation leaks forward even without literal overlap.

Both in `finb.models.cv.PurgedKFold`. It takes the `touch_idx` that
[[Triple Barrier|triple-barrier labelling]] reports — which is the reason that
labelling scheme records when each label resolves.

`leakage_report()` quantifies how much ordinary k-fold would have dropped. If
that fraction is large, any un-purged score in the project is meaningless.

López de Prado, *Advances in Financial Machine Learning*, ch. 7.
