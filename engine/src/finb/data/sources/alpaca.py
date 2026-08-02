"""Alpaca market data.

Two things to keep in mind, both from the research and both easy to forget:

1. **The free equity feed is IEX only — about 2.4% of consolidated volume.**
   That is not a scaled-down copy of the tape, it is a biased sample. Volume
   features built on it measure IEX's routing share, and range-based volatility
   (ATR, Parkinson, Garman-Klass) comes out systematically *low*, which makes
   volatility targeting oversize every position. `feed` is therefore explicit on
   every equity call — never defaulted, so the choice is always visible at the
   call site.

2. **Crypto data is the good stuff**: real-time, complete, and free, with no
   delay and no subscription. It is the highest-fidelity free surface available
   here, which is a large part of why crypto survived as a venue at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import polars as pl

from finb.config import Settings
from finb.data.lake import BAR_SCHEMA, Timeframe
from finb.log import get_logger

log = get_logger("alpaca")

_TF = {
    Timeframe.M1: ("Min", 1),
    Timeframe.M5: ("Min", 5),
    Timeframe.M15: ("Min", 15),
    Timeframe.H1: ("Hour", 1),
    Timeframe.D1: ("Day", 1),
}


def _timeframe(tf: Timeframe):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    unit, amount = _TF[tf]
    return TimeFrame(amount, getattr(TimeFrameUnit, unit))


def _to_frame(bars_by_symbol: dict, symbol: str) -> pl.DataFrame:
    rows = bars_by_symbol.get(symbol, [])
    if not rows:
        return pl.DataFrame(schema=BAR_SCHEMA)
    return pl.DataFrame(
        {
            "ts": [b.timestamp for b in rows],
            "open": [float(b.open) for b in rows],
            "high": [float(b.high) for b in rows],
            "low": [float(b.low) for b in rows],
            "close": [float(b.close) for b in rows],
            "volume": [float(b.volume) for b in rows],
            "trade_count": [int(b.trade_count) if b.trade_count is not None else None for b in rows],
            "vwap": [float(b.vwap) if b.vwap is not None else None for b in rows],
        },
        schema=BAR_SCHEMA,
    )


def fetch_crypto_bars(
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Crypto OHLCV. Needs no credentials — Alpaca's crypto data is public."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest

    client = CryptoHistoricalDataClient()
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=_timeframe(timeframe),
        start=start,
        end=end,
    )
    resp = client.get_crypto_bars(req)
    df = _to_frame(resp.data, symbol)
    log.info(f"{symbol} {timeframe.value}: {len(df)} bars")
    return df


def fetch_equity_bars(
    settings: Settings,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime | None = None,
    *,
    feed: Literal["iex", "sip"] = "iex",
    adjustment: Literal["raw", "split", "dividend", "all"] = "raw",
) -> pl.DataFrame:
    """Equity OHLCV.

    `feed` is required rather than defaulted silently. `iex` is free but partial;
    `sip` is complete but, on the free tier, only available with a 15-minute
    delay — and mixing the two between training and inference is train/serve
    skew, which the research flags as the single most likely silent killer.

    `adjustment` defaults to `raw` deliberately. Split/dividend-adjusted history
    bakes *future* corporate actions into past prices, so an executor trained on
    adjusted bars has been shown the future. Store raw; adjust for features only.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    if not (settings.alpaca_api_key_id and settings.alpaca_api_secret_key):
        raise RuntimeError("Alpaca credentials are not configured — see .env")

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key_id,
        secret_key=settings.alpaca_api_secret_key,
    )
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=_timeframe(timeframe),
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
    )
    resp = client.get_stock_bars(req)
    df = _to_frame(resp.data, symbol)
    log.info(f"{symbol} {timeframe.value} feed={feed} adj={adjustment}: {len(df)} bars")
    return df
