"""Equity Long/Short Strategy for Small-Capital Accounts.

Adheres strictly to Decision 0014:
- Both Long and Short legs are drawn from the same shortable universe
  (shortable, easy_to_borrow, price <= $125 under $500 budget).
- Tighter short risk limits: 10% per short position, 30% max gross short.
- Signed position sizing so closing longs reduces risk and opening shorts
  consumes short risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finb.log import get_logger

log = get_logger("short_equity")


def filter_shortable_universe(
    df_universe: pl.DataFrame,
    latest_prices: dict[str, float],
    max_price: float = 125.0,
) -> list[str]:
    """Filter candidate equity symbols for the long/short universe.

    Both legs must come from the same shortable, easy-to-borrow universe priced
    below max_price to prevent price-level confounding (Decision 0014).
    """
    candidates = df_universe.filter(
        (pl.col("asset_class") == "us_equity")
        & pl.col("tradable")
        & pl.col("shortable")
        & pl.col("easy_to_borrow")
    )["symbol"].to_list()

    eligible = []
    for sym in candidates:
        price = latest_prices.get(sym)
        if price is not None and 0.0 < price <= max_price:
            eligible.append(sym)

    log.info(f"Shortable equity screen: {len(candidates)} candidates -> {len(eligible)} under ${max_price:.0f}")
    return eligible


@dataclass
class EquityLongShortSignal:
    longs: list[str]
    shorts: list[str]
    weights: dict[str, float]


def generate_long_short_signals(
    closes: np.ndarray,
    symbols: list[str],
    lookback: int = 60,
    top_n_long: int = 3,
    top_n_short: int = 3,
    max_long_pct: float = 0.25,
    max_short_pct: float = 0.10,
) -> EquityLongShortSignal:
    """Generate signed target weights (+ for long, - for short)."""
    if closes.shape[0] < lookback + 1:
        return EquityLongShortSignal([], [], {})

    moms = closes[-1] / closes[-1 - lookback] - 1.0
    valid = np.isfinite(moms)

    if valid.sum() < top_n_long + top_n_short:
        return EquityLongShortSignal([], [], {})

    order_idx = np.argsort(np.where(valid, moms, -np.inf))[::-1]
    sorted_syms = [symbols[j] for j in order_idx if valid[j]]

    longs = sorted_syms[:top_n_long]
    shorts = sorted_syms[-top_n_short:]

    weights = {}
    for sym in longs:
        weights[sym] = max_long_pct
    for sym in shorts:
        weights[sym] = -max_short_pct  # Signed negative weight for short

    return EquityLongShortSignal(longs, shorts, weights)
