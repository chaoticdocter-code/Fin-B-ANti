"""Purged and embargoed cross-validation.

The problem, concretely. A triple-barrier label opened at bar 100 may not
resolve until bar 150. A label opened at bar 140 resolves at 190. Those two
labels are built from an overlapping stretch of price history, so they are not
independent observations — they are partly the same observation counted twice.

Put one in training and the other in test and the model is scored on data it has
effectively already seen. The result is a backtest that looks strong and a live
system that does not work.

Two corrections, both from López de Prado ch. 7:

- **Purging** removes any training sample whose label window overlaps the test
  window.
- **Embargo** additionally removes a small block of training samples
  *immediately after* the test window, because serial correlation in features
  leaks in that direction too even when label windows do not literally overlap.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PurgedKFold:
    """K-fold over time-ordered samples, with purging and an embargo.

    Samples must be in chronological order. Test folds are contiguous blocks —
    shuffling would be meaningless here, and is the mistake this class exists to
    prevent.
    """

    n_splits: int = 5
    embargo_pct: float = 0.01
    """Embargo length as a fraction of the sample count. 1% is a common
    default; raise it when features use long lookbacks."""

    def split(self, t1: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx).

        `t1[i]` is the index at which sample *i*'s label resolves — exactly the
        `touch_idx` returned by `finb.features.triple_barrier`. Passing
        ``arange(n)`` (labels that resolve instantly) reduces this to ordinary
        contiguous k-fold.
        """
        t1 = np.asarray(t1, dtype=int)
        n = t1.size
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n < self.n_splits:
            raise ValueError(f"need at least {self.n_splits} samples, got {n}")

        embargo = int(n * self.embargo_pct)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1] - 1  # inclusive test block
            test_idx = np.arange(lo, hi + 1)

            all_idx = np.arange(n)
            # Purge: drop training samples whose label window [j, t1[j]]
            # intersects the test block [lo, hi].
            overlaps = (all_idx <= hi) & (t1 >= lo)
            # Embargo: drop the block immediately following the test fold.
            embargoed = (all_idx > hi) & (all_idx <= hi + embargo)

            train_idx = all_idx[~(overlaps | embargoed)]
            yield train_idx, test_idx

    def get_n_splits(self, *_args, **_kwargs) -> int:
        return self.n_splits


def leakage_report(t1: np.ndarray, n_splits: int = 5, embargo_pct: float = 0.01) -> dict:
    """Quantify how much ordinary k-fold would have leaked.

    Useful as a sanity check before trusting any cross-validated score: if the
    dropped fraction is large, your labels overlap heavily and any un-purged
    result is meaningless.
    """
    t1 = np.asarray(t1, dtype=int)
    n = t1.size
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    naive_train = 0
    purged_train = 0
    for train_idx, test_idx in cv.split(t1):
        purged_train += train_idx.size
        naive_train += n - test_idx.size

    dropped = naive_train - purged_train
    return {
        "n_samples": n,
        "naive_train_samples": naive_train,
        "purged_train_samples": purged_train,
        "dropped": dropped,
        "dropped_fraction": dropped / naive_train if naive_train else 0.0,
        "mean_label_span": float(np.mean(t1 - np.arange(n))),
    }
