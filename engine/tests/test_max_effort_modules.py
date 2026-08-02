"""Unit tests for Max Effort modules: CompositeAlphaEngine, EquityCurveGuard, and BotDaemon."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from finb.bot_daemon import BotDaemon
from finb.models.composite_alpha import CompositeAlphaEngine, FactorWeights
from finb.risk.equity_curve_guard import EquityCurveGuard


def test_factor_weights_validation():
    fw = FactorWeights(trend=0.4, mean_reversion=0.2, volatility=0.2, volume=0.2)
    assert abs(fw.trend + fw.mean_reversion + fw.volatility + fw.volume - 1.0) < 1e-4

    with pytest.raises(ValueError, match="Factor weights must sum to 1.0"):
        FactorWeights(trend=0.5, mean_reversion=0.5, volatility=0.5, volume=0.5)


def test_composite_alpha_engine_scoring():
    engine = CompositeAlphaEngine()
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    closes = np.array([[100.0, 200.0, 50.0] for _ in range(120)])
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = np.array([[1000.0, 2000.0, 500.0] for _ in range(120)])

    res = engine.score_panel(closes, highs, lows, volumes, symbols, lookback=100)
    assert len(res.symbols) == 3
    assert len(res.rankings) == 3
    assert res.composite_scores.shape == (3,)


def test_equity_curve_guard_deleverage_and_halt():
    guard = EquityCurveGuard(deleverage_dd_pct=0.10, halt_dd_pct=0.15, initial_capital=500.0)
    now = datetime.now(UTC)

    # Normal state ($500 -> $500)
    d1 = guard.update(now, 500.0)
    assert d1.allowed_leverage_scale == 1.0
    assert not d1.halted

    # High-water mark update ($500 -> $600)
    d2 = guard.update(now, 600.0)
    assert d2.allowed_leverage_scale == 1.0
    assert guard.peak_equity == 600.0

    # 11% drawdown from peak $600 -> $534 (534-600)/600 = -11%
    d3 = guard.update(now, 534.0)
    assert d3.allowed_leverage_scale == 0.5
    assert not d3.halted
    assert "De-leverage active" in d3.reason

    # 16.7% drawdown from peak $600 -> $500 (500-600)/600 = -16.7%
    d4 = guard.update(now, 500.0)
    assert d4.allowed_leverage_scale == 0.0
    assert d4.halted
    assert "Emergency Halt" in d4.reason


def test_bot_daemon_health_initialization():
    daemon = BotDaemon(cycle_interval_seconds=3600.0, max_retries=2)
    assert daemon.health.cycles_completed == 0
    assert daemon.health.errors_encountered == 0
    assert daemon.health.last_cycle_time is None
