"""Panel dataset construction and purged training.

The most important assertion in this file is
`test_a_model_trained_on_noise_scores_at_chance`: if the pipeline can find
signal in a random walk, every result it produces later is worthless.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from finb.models.dataset import build_panel_dataset
from finb.models.gbdt import signal_to_returns, train_purged

T0 = datetime(2020, 1, 1, tzinfo=UTC)


def synthetic_bars(n_symbols=4, n=700, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n_symbols):
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, n))
        noise = rng.uniform(0.002, 0.02, n)
        out[f"SYM{k}"] = pl.DataFrame(
            {
                "ts": [T0 + timedelta(days=i) for i in range(n)],
                "open": close,
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "volume": rng.uniform(1e5, 1e6, n),
            }
        )
    return out


@pytest.fixture(scope="module")
def dataset():
    return build_panel_dataset(synthetic_bars(seed=20260801), horizon=20, warmup=130)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #


def test_dataset_builds_and_is_time_sorted(dataset):
    assert len(dataset) > 500
    assert (np.diff(dataset.ts.astype("datetime64[ns]").astype(np.int64)) >= 0).all()
    assert dataset.X.shape[1] == len(dataset.feature_names)


def test_no_nulls_survive_into_the_feature_matrix(dataset):
    assert np.isfinite(dataset.X).all()


def test_labels_are_binary_and_not_degenerate(dataset):
    assert set(np.unique(dataset.y)) <= {0, 1}
    assert 0.15 < dataset.y.mean() < 0.85


def test_label_resolution_index_never_points_backwards(dataset):
    """t1 must be at or after the sample itself, or purging is meaningless."""
    rows = np.arange(len(dataset))
    assert (dataset.t1_idx >= rows).all()
    assert (dataset.t1_idx < len(dataset)).all()


def test_labels_span_many_rows_so_purging_actually_matters(dataset):
    span = dataset.t1_idx - np.arange(len(dataset))
    assert span.mean() > 5, "overlapping labels are the reason purged CV exists"


def test_multiple_symbols_are_interleaved_by_time(dataset):
    """Stacked per-symbol then globally sorted — not concatenated in blocks."""
    first_100 = set(dataset.symbol[:100].tolist())
    assert len(first_100) > 1


def test_rejects_input_with_no_usable_history():
    with pytest.raises(ValueError, match="no symbol had enough history"):
        build_panel_dataset(synthetic_bars(n=50), horizon=20, warmup=130)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #


def test_training_produces_out_of_fold_predictions_everywhere(dataset):
    res = train_purged(
        dataset.X, dataset.y, dataset.t1_idx,
        n_splits=4, feature_names=dataset.feature_names,
    )
    covered = np.isfinite(res.oof)
    assert covered.mean() > 0.9
    assert len(res.models) >= 3
    assert res.importance and abs(sum(res.importance.values()) - 1.0) < 1e-6


def test_a_model_trained_on_noise_scores_at_chance(dataset):
    """The load-bearing test. Synthetic bars are a random walk; there is nothing
    to learn. An AUC meaningfully above 0.5 here means the pipeline leaks."""
    res = train_purged(dataset.X, dataset.y, dataset.t1_idx, n_splits=4)
    assert 0.42 < res.mean_auc < 0.58, f"suspicious AUC on pure noise: {res.mean_auc}"
    assert abs(res.information_coefficient(dataset.y)) < 0.10


def test_purging_actually_removes_training_rows(dataset):
    res = train_purged(dataset.X, dataset.y, dataset.t1_idx, n_splits=4)
    assert res.leakage["dropped_fraction"] > 0
    assert res.leakage["mean_label_span"] > 0


def test_a_learnable_signal_is_actually_learned():
    """Sanity in the other direction: if a real signal exists, find it.

    A pipeline that scores everything at chance is just as broken as one that
    scores everything highly.
    """
    rng = np.random.default_rng(7)
    n = 4000
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    t1 = np.minimum(np.arange(n) + 3, n - 1)

    res = train_purged(X, y, t1, n_splits=4)
    assert res.mean_auc > 0.85


# --------------------------------------------------------------------------- #
#  Costs
# --------------------------------------------------------------------------- #


def test_costs_are_charged_on_every_taken_trade():
    oof = np.array([0.9, 0.9, 0.2, 0.2])
    ret = np.array([0.01, 0.01, 0.01, 0.01])

    net = signal_to_returns(oof, ret, threshold=0.5, cost_bps=57.0)
    assert net[0] == pytest.approx(0.01 - 0.0057)
    assert net[2] == 0.0, "no trade taken, no cost"


def test_a_higher_threshold_takes_fewer_trades():
    rng = np.random.default_rng(1)
    oof = rng.uniform(0, 1, 1000)
    ret = rng.normal(0, 0.05, 1000)

    loose = signal_to_returns(oof, ret, threshold=0.5)
    tight = signal_to_returns(oof, ret, threshold=0.8)
    assert (tight != 0).sum() < (loose != 0).sum()


def test_a_zero_edge_signal_loses_exactly_the_cost():
    rng = np.random.default_rng(2)
    n = 20_000
    oof = np.full(n, 0.9)
    ret = rng.normal(0.0, 0.03, n)

    net = signal_to_returns(oof, ret, threshold=0.5, cost_bps=57.0)
    assert net.mean() * 1e4 == pytest.approx(-57.0, abs=5.0)
