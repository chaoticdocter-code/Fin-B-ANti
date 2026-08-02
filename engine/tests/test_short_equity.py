"""Unit tests for the Equity Long/Short strategy module."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from finb.short_equity import filter_shortable_universe, generate_long_short_signals


def test_filter_shortable_universe_filters_price_and_borrow():
    data = {
        "snapshot_date": [date(2026, 8, 2)] * 4,
        "symbol": ["AAPL", " cheap1", " cheap2", " unborrowable"],
        "asset_class": ["us_equity"] * 4,
        "tradable": [True] * 4,
        "shortable": [True, True, True, False],
        "easy_to_borrow": [True, True, True, False],
    }
    df = pl.DataFrame(data)
    prices = {
        "AAPL": 300.0,
        " cheap1": 50.0,
        " cheap2": 110.0,
        " unborrowable": 20.0,
    }

    filtered = filter_shortable_universe(df, prices, max_price=125.0)
    assert " cheap1" in filtered
    assert " cheap2" in filtered
    assert "AAPL" not in filtered
    assert " unborrowable" not in filtered


def test_generate_long_short_signals_assigns_signed_weights():
    symbols = ["A", "B", "C", "D", "E", "F"]
    closes = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [15.0, 14.0, 12.0, 8.0, 6.0, 5.0],
    ])

    sig = generate_long_short_signals(
        closes, symbols, lookback=1, top_n_long=2, top_n_short=2, max_long_pct=0.25, max_short_pct=0.10
    )

    assert sig.longs == ["A", "B"]
    assert set(sig.shorts) == {"E", "F"}
    assert sig.weights["A"] == 0.25
    assert sig.weights["B"] == 0.25
    assert sig.weights["F"] == -0.10
    assert sig.weights["E"] == -0.10
