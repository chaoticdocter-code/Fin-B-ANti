"""Unit tests for engine enhancements: VWAP deviation, RVOL, sentiment fetcher, and PositionState."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from finb.bot import PositionState
from finb.data.sources.sentiment import CryptoSentiment
from finb.features.indicators import rvol, vwap_deviation


def test_vwap_deviation_and_rvol_expressions():
    df = pl.DataFrame({
        "close": [10.0, 12.0, 11.0, 13.0, 14.0],
        "high": [10.5, 12.5, 11.5, 13.5, 14.5],
        "low": [9.5, 11.5, 10.5, 12.5, 13.5],
        "volume": [100.0, 200.0, 150.0, 300.0, 250.0],
    })

    res = df.select([vwap_deviation(3), rvol(3)])
    assert "vwap_dev_3" in res.columns
    assert "rvol_3" in res.columns
    assert res.height == 5


def test_position_state_last_entry_time(tmp_path: Path):
    file = tmp_path / "positions.json"
    state = PositionState(file)

    assert state.last_entry_time() is None

    t1 = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 8, 2, 14, 0, tzinfo=UTC)

    state.opened("BTC/USD", t1)
    assert state.last_entry_time() == t1

    state.opened("ETH/USD", t2)
    assert state.last_entry_time() == t2


def test_crypto_sentiment_dataclass():
    ts = datetime.now(UTC)
    s = CryptoSentiment(timestamp=ts, fear_greed_index=20, classification="Extreme Fear")
    assert s.is_extreme_fear
    assert not s.is_extreme_greed
