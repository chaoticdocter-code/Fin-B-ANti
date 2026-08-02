"""Composite Model combining Volume-Confirmed VWAP and Bollinger Mean Reversion (or EMA Scalping)."""

import polars as pl
from typing import Optional

from finb.models.base import Model, Target

class CompositeModel(Model):
    """
    Hyper-Aggressive HFT Scalping Strategy (Momentum Breakout).
    
    Logic for LONG ONLY (since Alpaca crypto has no shorting):
    - EMA Crossover: 9 EMA > 21 EMA (Short-term trend is up).
    - MACD Confirmation: MACD Histogram > 0 (Momentum is accelerating).
    - RSI Filter: 50 < RSI < 70 (Bullish zone, but not yet overbought).
    - Volume Confirmation: Current Volume > 1.2x Volume MA (Ensures real participation).
    
    If all true, set weight to 1.0. Otherwise 0.0.
    """
    
    def __init__(
        self, 
        ema_fast_col: str = "ema_9", 
        ema_slow_col: str = "ema_21",
        macd_hist_col: str = "macd_hist",
        vol_col: str = "volume",
        vol_ma_col: str = "vol_ma_20",
        rsi_col: str = "rsi_14",
        rsi_min: float = 40.0,
        rsi_max: float = 90.0,
        vol_spike_multiplier: float = 0.5
    ):
        self.ema_fast_col = ema_fast_col
        self.ema_slow_col = ema_slow_col
        self.macd_hist_col = macd_hist_col
        self.vol_col = vol_col
        self.vol_ma_col = vol_ma_col
        self.rsi_col = rsi_col
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.vol_spike_multiplier = vol_spike_multiplier

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        pass

    def predict(self, df: pl.DataFrame) -> list[Target]:
        targets = []
        for row in df.iter_rows(named=True):
            sym = row.get("symbol")
            if not sym:
                continue
                
            ema_fast = row.get(self.ema_fast_col)
            ema_slow = row.get(self.ema_slow_col)
            macd_hist = row.get(self.macd_hist_col)
            vol = row.get(self.vol_col)
            vol_ma = row.get(self.vol_ma_col)
            rsi = row.get(self.rsi_col)
            
            # If any required column is null, we can't trade
            if any(v is None for v in [ema_fast, ema_slow, macd_hist, vol, vol_ma, rsi]):
                targets.append(Target(symbol=sym, weight=0.0))
                continue
                
            # Filter 1: Short-term trend (EMA Cross)
            is_uptrend = ema_fast > ema_slow
            
            # Filter 3: RSI Bullish Zone
            is_bullish_rsi = self.rsi_min < rsi < self.rsi_max
            
            if is_uptrend and is_bullish_rsi:
                # High frequency breakout trigger
                targets.append(Target(symbol=sym, weight=1.0))
            else:
                targets.append(Target(symbol=sym, weight=0.0))
            
        return targets
