"""Triple-barrier labelling and purged CV.

The last test is the point of both modules: it shows ordinary k-fold scoring a
model at 90%+ on data that contains no signal whatsoever, and purged CV
returning it to chance.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from finb.features.labeling import ewm_volatility, triple_barrier
from finb.models.cv import PurgedKFold, leakage_report

# --------------------------------------------------------------------------- #
#  Labelling
# --------------------------------------------------------------------------- #


def test_upper_barrier_touched_first_is_labelled_plus_one():
    close = np.array([100.0, 101, 102, 103, 110, 90])
    vol = np.full(6, 0.05)
    r = triple_barrier(close, event_idx=np.array([0]), volatility=vol, pt=1.0, sl=1.0, max_bars=5)

    assert r.label[0] == 1
    assert r.barrier[0] == "pt"
    assert r.touch_idx[0] == 4      # 110 >= 105
    assert r.ret[0] == pytest.approx(0.10)


def test_lower_barrier_touched_first_is_labelled_minus_one():
    close = np.array([100.0, 99, 94, 120, 120, 120])
    vol = np.full(6, 0.05)
    r = triple_barrier(close, event_idx=np.array([0]), volatility=vol, pt=1.0, sl=1.0, max_bars=5)

    assert r.label[0] == -1
    assert r.barrier[0] == "sl"
    assert r.touch_idx[0] == 2


def test_the_path_decides_not_the_endpoint():
    """Price dips through the stop and recovers. Endpoint says win; reality says
    the position was already closed at a loss."""
    close = np.array([100.0, 94.0, 100.0, 106.0])
    vol = np.full(4, 0.05)
    r = triple_barrier(close, event_idx=np.array([0]), volatility=vol, pt=1.0, sl=1.0, max_bars=3)

    assert r.label[0] == -1
    assert r.touch_idx[0] == 1
    assert (close[-1] / close[0]) - 1 > 0  # naive forward return would say +6%


def test_no_barrier_touched_falls_to_the_vertical():
    close = np.array([100.0, 100.2, 99.8, 100.1, 100.0])
    vol = np.full(5, 0.05)
    r = triple_barrier(close, event_idx=np.array([0]), volatility=vol, pt=1.0, sl=1.0, max_bars=4)

    assert r.label[0] == 0
    assert r.barrier[0] == "vertical"
    assert r.touch_idx[0] == 4


def test_barriers_scale_with_volatility():
    """The same 3% move is a win in a calm regime and noise in a violent one."""
    close = np.array([100.0, 103.0, 103.0, 103.0])

    calm = triple_barrier(close, event_idx=np.array([0]), volatility=np.full(4, 0.01), max_bars=3)
    wild = triple_barrier(close, event_idx=np.array([0]), volatility=np.full(4, 0.10), max_bars=3)

    assert calm.label[0] == 1
    assert wild.label[0] == 0


def test_meta_labelling_asks_whether_the_bet_was_right():
    close = np.array([100.0, 94.0, 94.0, 94.0])
    vol = np.full(4, 0.05)

    # A short position in a falling market is correct.
    short = triple_barrier(
        close, event_idx=np.array([0]), volatility=vol, side=np.array([-1.0]), max_bars=3
    )
    assert short.label[0] == 1
    assert short.ret[0] > 0

    # The same market, betting long, is wrong.
    long = triple_barrier(
        close, event_idx=np.array([0]), volatility=vol, side=np.array([1.0]), max_bars=3
    )
    assert long.label[0] == 0


def test_disabled_barriers():
    close = np.array([100.0, 80.0, 130.0, 100.0])
    vol = np.full(4, 0.05)
    r = triple_barrier(close, event_idx=np.array([0]), volatility=vol, sl=0.0, max_bars=3)
    assert r.barrier[0] == "pt"  # the stop was switched off, so the drop is ignored


def test_ewm_volatility_is_positive_and_tracks_regime():
    rng = np.random.default_rng(1)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.001, 500))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, 500))

    assert np.all(ewm_volatility(calm) > 0)
    assert ewm_volatility(wild)[-1] > 10 * ewm_volatility(calm)[-1]


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length as close"):
        triple_barrier(np.arange(10.0), volatility=np.ones(5))
    with pytest.raises(ValueError, match="out of range"):
        triple_barrier(np.arange(10.0), event_idx=np.array([99]))


# --------------------------------------------------------------------------- #
#  Purged CV
# --------------------------------------------------------------------------- #


def test_purging_removes_every_overlapping_training_sample():
    n = 100
    t1 = np.minimum(np.arange(n) + 10, n - 1)   # each label spans 10 bars
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)

    for train_idx, test_idx in cv.split(t1):
        lo, hi = test_idx[0], test_idx[-1]
        for j in train_idx:
            overlaps = j <= hi and t1[j] >= lo
            assert not overlaps, f"train sample {j} (resolves {t1[j]}) overlaps test [{lo},{hi}]"


def test_embargo_removes_the_block_after_the_test_fold():
    n = 100
    t1 = np.arange(n)  # instantaneous labels: only the embargo can remove anything
    no_emb = list(PurgedKFold(n_splits=5, embargo_pct=0.0).split(t1))
    with_emb = list(PurgedKFold(n_splits=5, embargo_pct=0.10).split(t1))

    for k, ((tr_a, te), (tr_b, _)) in enumerate(zip(no_emb, with_emb, strict=True)):
        removed = set(tr_a) - set(tr_b)
        assert all(i > te[-1] for i in removed), "embargo must only look forward"
        if k < len(no_emb) - 1:
            assert tr_b.size < tr_a.size
        else:
            # The final fold ends at the last sample, so there is nothing after
            # it to embargo.
            assert removed == set()


def test_train_and_test_never_intersect():
    t1 = np.minimum(np.arange(200) + 25, 199)
    for train_idx, test_idx in PurgedKFold(n_splits=4, embargo_pct=0.02).split(t1):
        assert not (set(train_idx) & set(test_idx))
        assert test_idx.size > 0 and train_idx.size > 0


def test_instant_labels_reduce_to_plain_contiguous_kfold():
    n = 50
    splits = list(PurgedKFold(n_splits=5, embargo_pct=0.0).split(np.arange(n)))
    assert sum(len(te) for _, te in splits) == n
    for train_idx, test_idx in splits:
        assert train_idx.size == n - test_idx.size


def test_leakage_report_scales_with_label_span():
    short = leakage_report(np.minimum(np.arange(500) + 2, 499))
    long = leakage_report(np.minimum(np.arange(500) + 100, 499))

    assert long["dropped_fraction"] > short["dropped_fraction"]
    assert long["mean_label_span"] > short["mean_label_span"]


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="n_splits must be"):
        list(PurgedKFold(n_splits=1).split(np.arange(10)))
    with pytest.raises(ValueError, match="at least 5 samples"):
        list(PurgedKFold(n_splits=5).split(np.arange(3)))


# --------------------------------------------------------------------------- #
#  Why any of this matters
# --------------------------------------------------------------------------- #


def test_naive_kfold_finds_strong_signal_in_pure_noise_and_purged_cv_does_not():
    """A random walk. No signal exists. Ordinary shuffled k-fold says otherwise.

    Two ingredients make the leak, and both are ordinary modelling choices:

    - Labels span 100 bars with the barriers switched off, so consecutive events
      share 99% of their outcome window and almost always share a label.
    - Features are price and a 200-bar moving average — both vary *smoothly*, so
      consecutive events sit almost on top of each other in feature space, and
      the pair is nearly unique per point in time.

    Shuffle those across folds and a 1-nearest-neighbour model learns nothing.
    It finds each test sample's temporal twin sitting in the training set and
    reads the answer off it.

    Note the feature choice matters: trailing *returns* do not produce this,
    because shifting a window of iid noise by one bar lands somewhere completely
    different. Smoothness is what creates near-duplicates.
    """
    rng = np.random.default_rng(20260731)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 3000))

    horizon, warm = 100, 250

    def moving_average(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full_like(a, np.nan, dtype=float)
        out[w - 1 :] = (c[w:] - c[:-w]) / w
        return out

    events = np.arange(warm, len(close) - horizon)
    r = triple_barrier(
        close,
        event_idx=events,
        pt=0.0,                      # barriers off: every label runs the full
        sl=0.0,                      # horizon, so windows overlap heavily
        max_bars=horizon,
        label_vertical_by_sign=True,
    )
    X = np.column_stack([close[r.event_idx], moving_average(close, 200)[r.event_idx]])
    y = r.label

    model = KNeighborsClassifier(n_neighbors=1)

    naive = cross_val_score(
        model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0)
    ).mean()

    purged = PurgedKFold(n_splits=5, embargo_pct=0.01)
    scores = []
    for tr, te in purged.split(r.touch_idx):
        if tr.size < 10 or len(np.unique(y[tr])) < 2:
            continue
        scores.append(model.fit(X[tr], y[tr]).score(X[te], y[te]))
    honest = float(np.mean(scores))

    # Measured: naive 0.955, purged 0.495 — i.e. shuffled k-fold reports 95%
    # accuracy on data generated by a random number generator.
    assert naive > 0.90, f"expected the leak to be blatant, got {naive:.3f}"
    assert honest < 0.60, f"purged CV should return to chance, got {honest:.3f}"
    assert naive - honest > 0.35, f"gap collapsed: {naive:.3f} -> {honest:.3f}"
