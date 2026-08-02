"""Unit tests for QuantConnect Crypto Intraday Mean Reversion Strategy."""

from __future__ import annotations

import numpy as np
import pytest

from finb.models.qc_mean_reversion import QCIntradayMeanReversion


def test_qc_mean_reversion_long_signal():
    strat = QCIntradayMeanReversion(bb_period=20, rsi_period=14)
    symbols = ["BTC/USD", "ETH/USD"]
    n_bars = 30
    
    # Normal flat series around 100.0
    closes = np.ones((n_bars, len(symbols))) * 100.0
    highs = np.ones((n_bars, len(symbols))) * 101.0
    lows = np.ones((n_bars, len(symbols))) * 99.0

    # Drop BTC/USD significantly to trigger %B < 0.05 & oversold RSI
    closes[-5:, 0] = np.linspace(100.0, 75.0, 5)
    lows[-5:, 0] = np.linspace(99.0, 74.0, 5)

    signals = strat.generate_signals(closes, highs, lows, symbols)
    assert "BTC/USD" in signals
    sig = signals["BTC/USD"]
    assert sig.action == "BUY"
    assert sig.target_weight > 0
    assert sig.target_price > closes[-1, 0]


def test_qc_mean_reversion_short_signal():
    strat = QCIntradayMeanReversion(bb_period=20, rsi_period=14)
    symbols = ["ETH/USD"]
    n_bars = 30
    
    closes = np.ones((n_bars, len(symbols))) * 100.0
    highs = np.ones((n_bars, len(symbols))) * 101.0
    lows = np.ones((n_bars, len(symbols))) * 99.0

    # Spike ETH/USD upward to trigger %B > 0.95 & overbought RSI
    closes[-5:, 0] = np.linspace(100.0, 130.0, 5)
    highs[-5:, 0] = np.linspace(101.0, 131.0, 5)

    signals = strat.generate_signals(closes, highs, lows, symbols)
    assert "ETH/USD" in signals
    sig = signals["ETH/USD"]
    assert sig.action == "SELL"
    assert sig.target_weight < 0
    assert sig.target_price < closes[-1, 0]
