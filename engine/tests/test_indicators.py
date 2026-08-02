"""Every indicator is run through the leakage audit.

This is the point of writing them by hand rather than importing a library: the
causality guarantee is tested, not assumed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from finb.features import indicators as ind
from finb.features.leakage import assert_causal


@pytest.fixture
def bars():
    rng = np.random.default_rng(20260801)
    n = 400
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.02, n))
    noise = rng.uniform(0.001, 0.01, n)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [t0 + timedelta(days=i) for i in range(n)],
            "open": close * (1 - noise / 2),
            "high": close * (1 + noise),
            "low": close * (1 - noise),
            "close": close,
            "volume": rng.uniform(1e5, 1e6, n),
        }
    )


ALL_EXPRS = ind.DEFAULT + ind.VOLUME


@pytest.mark.parametrize("expr", ALL_EXPRS, ids=lambda e: ind.compute(
    pl.DataFrame({"close": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5, "volume": [1.0] * 5}),
    [e],
).columns[0])
def test_every_indicator_is_causal(bars, expr):
    assert_causal(lambda df: ind.compute(df, [expr]), bars, warmup=110)


def test_the_whole_default_set_is_causal(bars):
    assert_causal(lambda df: ind.compute(df), bars, warmup=110)


def test_feature_names_are_unique_and_descriptive():
    names = ind.feature_names(ALL_EXPRS)
    assert len(names) == len(set(names))
    assert all(any(ch.isdigit() for ch in n) for n in names), "windows should be in the name"


# --------------------------------------------------------------------------- #
#  Values behave as documented
# --------------------------------------------------------------------------- #


def frame(closes, highs=None, lows=None, vols=None):
    n = len(closes)
    return pl.DataFrame(
        {
            "close": closes,
            "high": highs or [c * 1.01 for c in closes],
            "low": lows or [c * 0.99 for c in closes],
            "volume": vols or [1000.0] * n,
        }
    )


def test_momentum_sign_follows_the_trend():
    up = frame(list(np.linspace(100, 200, 100)))
    down = frame(list(np.linspace(200, 100, 100)))
    assert ind.compute(up, [ind.momentum(20)])["mom_20"][-1] > 0
    assert ind.compute(down, [ind.momentum(20)])["mom_20"][-1] < 0


def test_momentum_skip_ignores_the_most_recent_bars():
    # Rises for 80 bars, then collapses over the last 5.
    path = list(np.linspace(100, 200, 80)) + list(np.linspace(200, 120, 20))
    df = frame(path)
    plain = ind.compute(df, [ind.momentum(30)])["mom_30"][-1]
    skipped = ind.compute(df, [ind.momentum(30, skip=20)])["mom_30_20"][-1]
    assert plain < skipped, "skipping the crash should leave a stronger signal"


def test_rsi_is_bounded_and_directional():
    up = ind.compute(frame(list(np.linspace(100, 200, 100))), [ind.rsi(14)])["rsi_14"]
    down = ind.compute(frame(list(np.linspace(200, 100, 100))), [ind.rsi(14)])["rsi_14"]
    vals = [v for v in up if v is not None] + [v for v in down if v is not None]
    assert all(0.0 <= v <= 100.0 for v in vals)
    assert up[-1] > 95
    assert down[-1] < 5


def test_zscore_is_scale_free():
    rng = np.random.default_rng(2)
    path = 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))
    a = ind.compute(frame(list(path)), [ind.zscore(20)])["z_20"][-1]
    b = ind.compute(frame(list(path * 1000)), [ind.zscore(20)])["z_20"][-1]
    assert a == pytest.approx(b, rel=1e-9)


def test_volatility_rises_with_volatility():
    rng = np.random.default_rng(3)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.002, 300))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.04, 300))
    v_calm = ind.compute(frame(list(calm)), [ind.realized_vol(20)])["vol_20"][-1]
    v_wild = ind.compute(frame(list(wild)), [ind.realized_vol(20)])["vol_20"][-1]
    assert v_wild > 5 * v_calm


def test_parkinson_uses_the_range_not_the_close():
    """Two series with identical closes but different intrabar ranges."""
    closes = [100.0] * 100
    tight = frame(closes, highs=[100.1] * 100, lows=[99.9] * 100)
    wide = frame(closes, highs=[105.0] * 100, lows=[95.0] * 100)

    assert ind.compute(tight, [ind.realized_vol(20)])["vol_20"][-1] == pytest.approx(0.0, abs=1e-12)
    assert (
        ind.compute(wide, [ind.parkinson_vol(20)])["parkinson_20"][-1]
        > ind.compute(tight, [ind.parkinson_vol(20)])["parkinson_20"][-1]
    )


def test_distance_from_high_is_zero_at_a_new_high():
    rising = frame(list(np.linspace(100, 200, 100)))
    assert ind.compute(rising, [ind.distance_from_high(20)])["from_high_20"][-1] == pytest.approx(0.0)


def test_warmup_nulls_are_present_and_correctly_sized():
    df = frame(list(np.linspace(100, 200, 100)))
    out = ind.compute(df, [ind.price_to_sma(20)])
    assert out["px_sma_20"].null_count() == 19
    assert out["px_sma_20"][19] is not None
