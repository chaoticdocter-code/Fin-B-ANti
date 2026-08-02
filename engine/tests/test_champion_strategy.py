"""Tests for Champion Strategy Archetype (TrendVolTargetStrategy & RegimeRouter)."""

from __future__ import annotations

import numpy as np
import pytest

from finb.models.regime_router import MarketRegime, RegimeRouter
from finb.models.strategies import TrendVolTargetStrategy


def test_trend_vol_target_strategy_signal_generation():
    strat = TrendVolTargetStrategy(sma_period=20, target_vol=0.20)
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    
    # 50 bars of synthetic prices trending up for BTC and ETH
    np.random.seed(42)
    n_bars = 50
    closes = np.zeros((n_bars, len(symbols)))
    for j in range(len(symbols)):
        base = 100.0 * (j + 1)
        trend = np.linspace(0, 20.0, n_bars)
        noise = np.random.normal(0, 1.0, n_bars)
        closes[:, j] = base + trend + noise

    res = strat.generate_signals(closes, symbols)
    assert res.strategy_name == "TrendVolTarget"
    # DOGE/USD should be 0 because it's not in MAJORS
    assert res.target_weights["DOGE/USD"] == 0.0
    # Liquid majors should have positive target weights
    assert res.target_weights["BTC/USD"] > 0.0
    assert res.target_weights["ETH/USD"] > 0.0
    assert res.target_weights["SOL/USD"] > 0.0


def test_regime_router_routes_to_trend():
    router = RegimeRouter()
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    n_bars = 50
    closes = np.tile(np.linspace(100, 150, n_bars)[:, None], (1, 3))
    highs = closes + 2.0
    lows = closes - 2.0
    volumes = np.ones((n_bars, 3)) * 1000.0

    res = router.route(closes, highs, lows, volumes, symbols, {"BTC/USD": 0.33, "ETH/USD": 0.33, "SOL/USD": 0.33})
    assert res.active_regime in [MarketRegime.TREND, MarketRegime.SQUEEZE, MarketRegime.RANGEBOUND]
    assert res.signal_result.strategy_name in ["TrendVolTarget", "VolSqueeze", "PairSpread"]
