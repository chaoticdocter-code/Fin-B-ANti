"""US Equities data fetching."""

import httpx
import polars as pl
import yfinance as yf
from datetime import datetime, timezone

from finb.data.lake import BarLake, Timeframe, AssetClass
from finb.config import get_settings

def fetch_yfinance_history(
    symbol: str, 
    timeframe: Timeframe, 
    start: datetime, 
    end: datetime
) -> pl.DataFrame:
    """Fetch history using yfinance and conform to BarLake schema."""
    interval_map = {
        Timeframe.M1: "1m",
        Timeframe.M5: "5m",
        Timeframe.M15: "15m",
        Timeframe.H1: "1h",
        Timeframe.D1: "1d",
    }
    interval = interval_map.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")

    # yfinance expects YYYY-MM-DD
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
    )
    
    if df.empty:
        return pl.DataFrame()

    df = df.reset_index()
    # Handle both 'Date' (daily) and 'Datetime' (intraday)
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    
    # Map to schema
    # yf columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    pldf = pl.DataFrame({
        "ts": df[time_col],
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "close": df["Close"],
        "volume": df["Volume"],
    })

    # Add missing columns with nulls
    pldf = pldf.with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.lit(None, dtype=pl.Int64).alias("trade_count"),
        pl.lit(None, dtype=pl.Float64).alias("vwap")
    ])
    
    # Ensure timezone is UTC
    return pldf


def backfill_equities(
    lake: BarLake, 
    symbols: list[str], 
    timeframe: Timeframe, 
    start: datetime, 
    end: datetime
) -> None:
    """Backfill missing data for equities in the given time range."""
    for sym in symbols:
        missing = lake.missing(sym, timeframe, start, end, asset=AssetClass.EQUITY)
        if not missing:
            continue
            
        print(f"[{sym}] Fetching {len(missing)} missing bars for {timeframe}...")
        df = fetch_yfinance_history(sym, timeframe, start, end)
        if not df.is_empty():
            count = lake.write(sym, timeframe, df, asset=AssetClass.EQUITY)
            print(f"[{sym}] Lake now holds {count} bars.")
        else:
            print(f"[{sym}] No data returned from provider.")
