"""Unit tests for Squeeze indicators, VolSqueezeStrategy, PairSpreadStrategy, and RegimeRouter."""

from __future__ import annotations

import numpy as np
import polars as pl

from finb.features.indicators import bollinger_width, keltner_width, volatility_squeeze
from finb.models.regime_router import MarketRegime, RegimeRouter
from finb.models.strategies import PairSpreadStrategy, VolSqueezeStrategy


def test_squeeze_indicators():
    df = pl.DataFrame({
        "close": [10.0 + i * 0.1 for i in range(30)],
        "high": [10.2 + i * 0.1 for i in range(30)],
        "low": [9.8 + i * 0.1 for i in range(30)],
    })

    res = df.select([bollinger_width(20), keltner_width(20), volatility_squeeze(20)])
    assert "bb_width_20" in res.columns
    assert "kc_width_20" in res.columns
    assert "vol_squeeze_20" in res.columns


def test_vol_squeeze_strategy_signal_generation():
    strat = VolSqueezeStrategy()
    symbols = ["BTC/USD", "ETH/USD"]

    closes = np.array([[100.0, 200.0] for _ in range(60)])
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.array([[1000.0, 2000.0] for _ in range(60)])

    res = strat.generate_signals(closes, highs, lows, volumes, symbols)
    assert res.strategy_name == "VolSqueeze"
    assert "BTC/USD" in res.target_weights


def test_pair_spread_strategy_signal_generation():
    strat = PairSpreadStrategy(window=20)
    symbols = ["ETH/USD", "BTC/USD"]

    closes = np.array([[1800.0 + (i * 2.0), 60000.0] for i in range(50)])
    res = strat.generate_signals(closes, symbols)
    assert res.strategy_name == "PairSpread"
    assert len(res.target_weights) == 2


def test_regime_router_classification():
    router = RegimeRouter()
    closes = np.array([[100.0, 200.0] for _ in range(50)])
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.array([[1000.0, 2000.0] for _ in range(50)])

    regime = router.classify_regime(closes, highs, lows, volumes)
    assert isinstance(regime, MarketRegime)
