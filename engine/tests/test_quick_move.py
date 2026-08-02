"""Unit tests for Quick-Move Strategy & Adaptive Tweaker."""

from __future__ import annotations

import numpy as np
import pytest

from finb.evaluation.tweaker import AdaptiveTweaker, StrategyParameters, TradeOutcome
from finb.models.quick_move import QuickMoveStrategy


def test_quick_move_strategy_trigger():
    strat = QuickMoveStrategy(lookback_bars=20, volume_threshold=1.2, take_profit_pct=0.02)
    symbols = ["BTC/USD", "ETH/USD"]
    n_bars = 30
    
    closes = np.ones((n_bars, len(symbols))) * 100.0
    highs = np.ones((n_bars, len(symbols))) * 101.0
    lows = np.ones((n_bars, len(symbols))) * 99.0
    volumes = np.ones((n_bars, len(symbols))) * 100.0

    # Breakout on bar -1 for BTC/USD with 2x volume surge
    closes[-1, 0] = 105.0
    highs[-1, 0] = 106.0
    volumes[-1, 0] = 250.0

    signals = strat.generate_signals(closes, highs, lows, volumes, symbols)
    assert "BTC/USD" in signals
    sig = signals["BTC/USD"]
    assert sig.action == "BUY"
    assert sig.take_profit_price == pytest.approx(105.0 * 1.02)
    assert sig.stop_loss_price == pytest.approx(105.0 * (1.0 - 0.012))


def test_quick_move_strategy_short_trigger():
    strat = QuickMoveStrategy(lookback_bars=20, volume_threshold=1.2, take_profit_pct=0.02, stop_loss_pct=0.012)
    symbols = ["ETH/USD"]
    n_bars = 30
    
    closes = np.ones((n_bars, len(symbols))) * 100.0
    highs = np.ones((n_bars, len(symbols))) * 101.0
    lows = np.ones((n_bars, len(symbols))) * 99.0
    volumes = np.ones((n_bars, len(symbols))) * 100.0

    # Breakdown on bar -1 for ETH/USD with volume surge
    closes[-1, 0] = 95.0
    lows[-1, 0] = 94.0
    volumes[-1, 0] = 300.0

    signals = strat.generate_signals(closes, highs, lows, volumes, symbols)
    assert "ETH/USD" in signals
    sig = signals["ETH/USD"]
    assert sig.action == "SELL"
    assert sig.target_weight < 0
    assert sig.take_profit_price == pytest.approx(95.0 * (1.0 - 0.02))
    assert sig.stop_loss_price == pytest.approx(95.0 * (1.0 + 0.012))



def test_adaptive_tweaker_parameter_adjustment():
    tweaker = AdaptiveTweaker(StrategyParameters(volume_threshold=1.3, take_profit_pct=0.02))
    
    # Record 5 winning trades
    for i in range(5):
        tweaker.record_trade(TradeOutcome("BTC/USD", 100.0, 103.0, 0.20, 4.0, 2.80))

    params = tweaker.evaluate_and_tweak()
    # High win rate should trigger take profit expansion
    assert params.take_profit_pct > 0.02
