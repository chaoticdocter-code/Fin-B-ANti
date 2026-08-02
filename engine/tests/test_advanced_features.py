"""Unit tests for Advanced Features: Chandelier Exit, TrailingStopEngine, TWAPExecutor, and MLSignalFilter."""

from __future__ import annotations

import polars as pl

from finb.execution.stops import TrailingStopEngine
from finb.execution.twap import TWAPConfig
from finb.features.indicators import chandelier_exit


def test_chandelier_exit_indicator():
    df = pl.DataFrame({
        "close": [10.0, 12.0, 11.0, 13.0, 14.0],
        "high": [10.5, 12.5, 11.5, 13.5, 14.5],
        "low": [9.5, 11.5, 10.5, 12.5, 13.5],
    })

    res = df.select(chandelier_exit(n=3, mult=2.0))
    assert "chandelier_3_2" in res.columns
    assert res.height == 5


def test_trailing_stop_engine_hwm_and_trigger():
    engine = TrailingStopEngine(atr_multiplier=2.0)

    # Initial price update
    hwm = engine.update_high_water_mark("BTC/USD", 100.0)
    assert hwm == 100.0

    # New high
    hwm = engine.update_high_water_mark("BTC/USD", 120.0)
    assert hwm == 120.0

    # Check stop (HWM 120 - 2.0 * ATR 5.0 = stop 110)
    res = engine.check_stop("BTC/USD", current_price=115.0, atr_value=5.0)
    assert not res.triggered
    assert res.stop_price == 110.0

    # Price drops to 105 -> Triggered!
    res = engine.check_stop("BTC/USD", current_price=105.0, atr_value=5.0)
    assert res.triggered
    assert "breached" in res.reason


def test_twap_config_defaults():
    cfg = TWAPConfig()
    assert cfg.slices == 3
    assert cfg.min_slice_notional == 10.0
