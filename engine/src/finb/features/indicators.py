"""Causal technical features.

Written rather than pulled from TA-Lib or pandas-ta for three reasons: the
Windows build story for TA-Lib is miserable, most libraries return
silently-centred or back-filled series somewhere, and every function here is
covered by `finb.features.leakage.assert_causal` — which is a guarantee no
third-party indicator library offers.

Every expression is trailing-window only. No centring, no back-fill, no
whole-series statistics. If you add one, add it to the audited set too.

> A caveat that matters more than the maths: on Alpaca's free equity feed these
> are computed from IEX prints, about 2.4% of the tape. Volume features there
> measure IEX's routing share rather than market activity, and range-based
> volatility comes out biased low. On crypto, where the feed is complete, they
> mean what they say.
"""

from __future__ import annotations

import math

import polars as pl

C = pl.col("close")
H = pl.col("high")
L = pl.col("low")
V = pl.col("volume")


# --------------------------------------------------------------------------- #
#  Returns and trend
# --------------------------------------------------------------------------- #


def log_return(n: int = 1) -> pl.Expr:
    """Log return over the trailing n bars."""
    return (C / C.shift(n)).log().alias(f"logret_{n}")


def momentum(lookback: int, skip: int = 0) -> pl.Expr:
    """Return over `lookback` bars, ending `skip` bars ago.

    The skip exists because short-horizon reversal is well documented; the 12-1
    convention in equities is precisely a one-month skip.
    """
    return (C.shift(skip) / C.shift(skip + lookback) - 1.0).alias(
        f"mom_{lookback}_{skip}" if skip else f"mom_{lookback}"
    )


def price_to_sma(n: int) -> pl.Expr:
    """Price relative to its trailing simple moving average. >1 is above trend."""
    return (C / C.rolling_mean(window_size=n)).alias(f"px_sma_{n}")


def sma_ratio(fast: int, slow: int) -> pl.Expr:
    """Fast MA over slow MA — a scale-free trend measure."""
    return (
        C.rolling_mean(window_size=fast) / C.rolling_mean(window_size=slow)
    ).alias(f"sma_{fast}_{slow}")


def distance_from_high(n: int) -> pl.Expr:
    """Drawdown from the trailing n-bar high, in [-1, 0]."""
    return (C / C.rolling_max(window_size=n) - 1.0).alias(f"from_high_{n}")


def distance_from_low(n: int) -> pl.Expr:
    return (C / C.rolling_min(window_size=n) - 1.0).alias(f"from_low_{n}")


# --------------------------------------------------------------------------- #
#  Volatility
# --------------------------------------------------------------------------- #


def realized_vol(n: int) -> pl.Expr:
    """Standard deviation of trailing log returns."""
    return (C / C.shift(1)).log().rolling_std(window_size=n).alias(f"vol_{n}")


def parkinson_vol(n: int) -> pl.Expr:
    """Range-based volatility. Uses high/low, so it is far more efficient than
    close-to-close — and correspondingly more damaged by a partial feed."""
    hl2 = (H / L).log() ** 2
    return (
        (hl2.rolling_mean(window_size=n) / (4.0 * math.log(2.0))).sqrt()
    ).alias(f"parkinson_{n}")


def atr(n: int) -> pl.Expr:
    """Average true range, normalised by price so it is comparable across assets."""
    prev_close = C.shift(1)
    tr = pl.max_horizontal(
        H - L,
        (H - prev_close).abs(),
        (L - prev_close).abs(),
    )
    return (tr.rolling_mean(window_size=n) / C).alias(f"atr_{n}")


def vol_of_vol(n: int) -> pl.Expr:
    """Instability of volatility itself — a decent regime marker."""
    v = (C / C.shift(1)).log().rolling_std(window_size=n)
    return (v.rolling_std(window_size=n) / v).alias(f"volvol_{n}")


def chandelier_exit(n: int = 22, mult: float = 3.0) -> pl.Expr:
    """Chandelier Exit: Trailing stop level derived from trailing high minus ATR multiple."""
    prev_close = C.shift(1)
    tr = pl.max_horizontal(
        H - L,
        (H - prev_close).abs(),
        (L - prev_close).abs(),
    )
    raw_atr = tr.rolling_mean(window_size=n)
    return (H.rolling_max(window_size=n) - (raw_atr * mult)).alias(f"chandelier_{n}_{int(mult)}")


def bollinger_width(n: int = 20, mult: float = 2.0) -> pl.Expr:
    """Bollinger Band width relative to rolling SMA."""
    sma = C.rolling_mean(window_size=n)
    std = C.rolling_std(window_size=n)
    return ((2.0 * mult * std) / (sma + 1e-12)).alias(f"bb_width_{n}")


def keltner_width(n: int = 20, mult: float = 1.5) -> pl.Expr:
    """Keltner Channel width relative to rolling SMA."""
    sma = C.rolling_mean(window_size=n)
    prev_close = C.shift(1)
    tr = pl.max_horizontal(H - L, (H - prev_close).abs(), (L - prev_close).abs())
    raw_atr = tr.rolling_mean(window_size=n)
    return ((2.0 * mult * raw_atr) / (sma + 1e-12)).alias(f"kc_width_{n}")


def volatility_squeeze(n: int = 20) -> pl.Expr:
    """True if Bollinger Bands are contracting inside Keltner Channels (Squeeze)."""
    return (bollinger_width(n, 2.0) < keltner_width(n, 1.5)).alias(f"vol_squeeze_{n}")


# --------------------------------------------------------------------------- #
#  Mean reversion / oscillators
# --------------------------------------------------------------------------- #


def zscore(n: int) -> pl.Expr:
    """Price z-score against its own trailing window.

    Trailing mean and std, never the whole series — the whole-series version is
    the leak that `test_whole_series_normalisation_is_caught` exists for.
    """
    return (
        (C - C.rolling_mean(window_size=n)) / C.rolling_std(window_size=n)
    ).alias(f"z_{n}")


def rsi(n: int = 14) -> pl.Expr:
    """Cutler's RSI — simple moving averages of gains and losses.

    Preferred over Wilder's smoothing here because a plain rolling mean has an
    unambiguous warmup, which makes the causality audit meaningful.
    """
    delta = C.diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0).rolling_mean(window_size=n)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0).rolling_mean(window_size=n)
    rs = gain / (loss + 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{n}")


# --------------------------------------------------------------------------- #
#  Volume
# --------------------------------------------------------------------------- #


def volume_ratio(n: int) -> pl.Expr:
    """Current volume against its trailing average.

    On IEX equity data this measures IEX's share of routing, not market
    activity, and that share varies by symbol, time of day and regime. Reliable
    on crypto; treat with suspicion on free equity data.
    """
    return (V / V.rolling_mean(window_size=n)).alias(f"volratio_{n}")


def rvol(n: int = 20) -> pl.Expr:
    """Relative Volume: current bar volume relative to trailing average."""
    return (V / (V.rolling_mean(window_size=n) + 1e-12)).alias(f"rvol_{n}")


def vwap_deviation(n: int = 24) -> pl.Expr:
    """Percentage deviation of close price from rolling VWAP over trailing n bars."""
    typical = (H + L + C) / 3.0
    vwap = (typical * V).rolling_sum(window_size=n) / (V.rolling_sum(window_size=n) + 1e-12)
    return (C / vwap - 1.0).alias(f"vwap_dev_{n}")


def dollar_volume(n: int) -> pl.Expr:
    return ((V * C).rolling_mean(window_size=n)).log1p().alias(f"dollarvol_{n}")


# --------------------------------------------------------------------------- #
#  Presets
# --------------------------------------------------------------------------- #

TREND = [
    momentum(20), momentum(60), momentum(90, skip=7),
    price_to_sma(20), price_to_sma(60),
    sma_ratio(10, 50),
    distance_from_high(60), distance_from_low(60),
]

VOLATILITY = [realized_vol(20), realized_vol(60), parkinson_vol(20), atr(14), vol_of_vol(20)]

REVERSION = [zscore(20), zscore(60), rsi(14), log_return(1), log_return(5)]

VOLUME = [volume_ratio(20), dollar_volume(20)]

DEFAULT = TREND + VOLATILITY + REVERSION


def compute(bars: pl.DataFrame, exprs: list[pl.Expr] | None = None) -> pl.DataFrame:
    """Evaluate feature expressions against a bar frame.

    Bars must be sorted ascending by `ts`. Nothing here sorts for you, because
    silently reordering a frame is how a subtly wrong index becomes a subtly
    wrong backtest.
    """
    return bars.select(exprs if exprs is not None else DEFAULT)


def feature_names(exprs: list[pl.Expr] | None = None) -> list[str]:
    return compute(
        pl.DataFrame(
            {
                "close": [1.0] * 5,
                "high": [1.0] * 5,
                "low": [1.0] * 5,
                "volume": [1.0] * 5,
            }
        ),
        exprs,
    ).columns
