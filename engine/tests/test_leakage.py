"""The leakage suite, tested against feature functions that genuinely leak.

Each leaky example below is a real mistake people make, and none of them raise
an exception on their own — they just quietly improve the backtest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from finb.features.leakage import assert_causal, audit_features


@pytest.fixture
def bars():
    rng = np.random.default_rng(20260801)
    n = 300
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [t0 + timedelta(hours=i) for i in range(n)],
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.uniform(1e3, 1e4, n),
        }
    )


# --------------------------------------------------------------------------- #
#  Causal features must pass
# --------------------------------------------------------------------------- #


def clean_features(df: pl.DataFrame) -> pl.DataFrame:
    """Trailing windows only, and returns computed from past bars."""
    return df.select(
        sma_20=pl.col("close").rolling_mean(window_size=20),
        ret_1=pl.col("close").pct_change(),
        vol_20=pl.col("close").pct_change().rolling_std(window_size=20),
        hl_range=(pl.col("high") - pl.col("low")) / pl.col("close"),
    )


def test_a_causal_feature_set_passes(bars):
    report = audit_features(clean_features, bars, warmup=25)
    assert report.passed, str(report)
    assert report.checks_run > 0
    assert_causal(clean_features, bars, warmup=25)


def test_lagged_features_pass(bars):
    def lagged(df):
        return df.select(prev_close=pl.col("close").shift(1))

    assert_causal(lagged, bars, warmup=2)


# --------------------------------------------------------------------------- #
#  Leaky features must be caught
# --------------------------------------------------------------------------- #


def test_centered_rolling_window_is_caught(bars):
    """A centred window averages bars from the future. Extremely common."""

    def leaky(df):
        return df.select(
            sma_centered=pl.col("close").rolling_mean(window_size=21, center=True)
        )

    report = audit_features(leaky, bars, warmup=25)
    assert not report.passed
    assert "sma_centered" in report.leaking_columns
    assert any(f.check == "truncation" for f in report.findings)


def test_whole_series_normalisation_is_caught(bars):
    """Z-scoring against the full series' mean and std uses every future bar."""

    def leaky(df):
        c = pl.col("close")
        return df.select(z=(c - c.mean()) / c.std())

    report = audit_features(leaky, bars, warmup=25)
    assert not report.passed
    assert "z" in report.leaking_columns


def test_forward_looking_return_is_caught(bars):
    """A negative shift is the target, not a feature."""

    def leaky(df):
        return df.select(fwd_ret=pl.col("close").shift(-1) / pl.col("close") - 1)

    report = audit_features(leaky, bars, warmup=5)
    assert not report.passed
    assert "fwd_ret" in report.leaking_columns


def test_backfill_is_caught(bars):
    """Back-filling a gap copies a later value into an earlier row."""

    def leaky(df):
        gapped = df.with_columns(
            pl.when(pl.int_range(pl.len()) % 17 == 0)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        return gapped.select(filled=pl.col("close").fill_null(strategy="backward"))

    report = audit_features(leaky, bars, warmup=5)
    assert not report.passed
    assert "filled" in report.leaking_columns


def test_expanding_max_of_the_whole_series_is_caught(bars):
    """Dividing by the series maximum is normalisation by hindsight."""

    def leaky(df):
        return df.select(pct_of_peak=pl.col("close") / pl.col("close").max())

    assert not audit_features(leaky, bars, warmup=5).passed


def test_future_noise_check_catches_what_truncation_might_miss(bars):
    """A feature reading only the final bar keeps the same *length* under
    truncation, so corrupting the tail is what exposes it."""

    def leaky(df):
        return df.select(vs_last=pl.col("close") / pl.col("close").last())

    report = audit_features(leaky, bars, warmup=5)
    assert not report.passed
    assert any(f.check == "future-noise" for f in report.findings)


# --------------------------------------------------------------------------- #
#  Warmup
# --------------------------------------------------------------------------- #


def test_missing_warmup_nulls_are_flagged(bars):
    """Back-filling the warmup period drags future values into the first rows.

    Note that forward-fill would be fine here — it cannot fill *leading* nulls
    at all, and copies only past values. Backward-fill is the leaky direction.
    """

    def suspicious(df):
        return df.select(
            sma=pl.col("close").rolling_mean(window_size=20).fill_null(strategy="backward")
        )

    report = audit_features(suspicious, bars, warmup=25, expect_warmup=20)
    assert not report.passed
    assert any(f.check == "warmup" for f in report.findings)


def test_forward_fill_is_causal_and_passes(bars):
    def fine(df):
        gapped = df.with_columns(
            pl.when(pl.int_range(pl.len()) % 17 == 0)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        return gapped.select(filled=pl.col("close").fill_null(strategy="forward"))

    assert_causal(fine, bars, warmup=25)


def test_legitimate_warmup_nulls_pass(bars):
    def fine(df):
        return df.select(sma=pl.col("close").rolling_mean(window_size=20))

    report = audit_features(fine, bars, warmup=25, expect_warmup=20)
    assert report.passed, str(report)


# --------------------------------------------------------------------------- #


def test_report_reads_usefully(bars):
    def leaky(df):
        return df.select(bad=pl.col("close").shift(-3))

    report = audit_features(leaky, bars, warmup=5)
    text = str(report)
    assert "LOOKAHEAD DETECTED" in text
    assert "bad" in text

    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        assert_causal(leaky, bars, warmup=5)
