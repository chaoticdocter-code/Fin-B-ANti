"""QuantConnect Crypto Intraday Mean Reversion Strategy Engine.

Ingested from QuantConnect open-source algorithmic strategy archetype:
1. Identifies short-term price over-extensions using Bollinger Bands (%B) and RSI(14).
2. Executes Long entries on capitulation bounces (%B < 0.05, RSI < 35).
3. Executes Short entries on climax pull-backs (%B > 0.95, RSI > 65).
4. Targets mean reversion back to the 20-period Moving Average with 1.5x ATR trailing stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finb.log import get_logger

log = get_logger("qc_mean_reversion")


@dataclass
class MeanReversionSignal:
    symbol: str
    action: str              # BUY | SELL | HOLD
    target_weight: float
    target_price: float      # SMA_20 mean reversion target
    stop_loss_price: float
    rationale: str


class QCIntradayMeanReversion:
    """QuantConnect Intraday Crypto Mean Reversion Engine."""

    TARGET_SYMBOLS = {"BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"}

    def __init__(
        self,
        bb_period: int = 20,
        rsi_period: int = 14,
        max_position_pct: float = 0.33,
    ) -> None:
        self.bb_period = bb_period
        self.rsi_period = rsi_period
        self.max_position_pct = max_position_pct

    def _compute_rsi(self, prices: np.ndarray) -> float:
        """Calculates 14-period RSI."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-self.rsi_period :])
        avg_loss = np.mean(losses[-self.rsi_period :])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def generate_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        symbols: list[str],
    ) -> dict[str, MeanReversionSignal]:
        n_bars, n_sym = closes.shape
        signals: dict[str, MeanReversionSignal] = {}

        if n_bars < self.bb_period:
            return signals

        for j, sym in enumerate(symbols):
            if sym not in self.TARGET_SYMBOLS:
                continue

            c_win = closes[-self.bb_period :, j]
            h_win = highs[-self.bb_period :, j]
            l_win = lows[-self.bb_period :, j]

            curr_price = float(closes[-1, j])
            sma_20 = float(np.mean(c_win))
            std_dev = float(np.std(c_win))

            upper_bb = sma_20 + 2.0 * std_dev
            lower_bb = sma_20 - 2.0 * std_dev
            pct_b = (curr_price - lower_bb) / (upper_bb - lower_bb + 1e-12)

            rsi_val = self._compute_rsi(closes[:, j])

            prev_c = np.roll(c_win, 1)
            prev_c[0] = c_win[0]
            tr = np.maximum(h_win - l_win, np.abs(h_win - prev_c))
            atr_val = float(np.mean(tr))

            # LONG Signal: Extreme Oversold (%B < 0.05 & RSI < 35) -> Revert to SMA_20
            if pct_b < 0.05 and rsi_val < 35.0:
                sl_price = curr_price - 1.5 * atr_val
                signals[sym] = MeanReversionSignal(
                    symbol=sym,
                    action="BUY",
                    target_weight=self.max_position_pct,
                    target_price=sma_20,
                    stop_loss_price=sl_price,
                    rationale=f"Oversold Capitulation (%B={pct_b:.2f}, RSI={rsi_val:.1f}); Reversion target ${sma_20:.2f}",
                )
                log.info(f"QC MeanReversion LONG for {sym}: Target=${sma_20:.2f}, SL=${sl_price:.2f}")

            # SHORT Signal: Extreme Overbought (%B > 0.95 & RSI > 65) -> Revert to SMA_20
            elif pct_b > 0.95 and rsi_val > 65.0:
                sl_price = curr_price + 1.5 * atr_val
                signals[sym] = MeanReversionSignal(
                    symbol=sym,
                    action="SELL",
                    target_weight=-self.max_position_pct,
                    target_price=sma_20,
                    stop_loss_price=sl_price,
                    rationale=f"Overbought Climax (%B={pct_b:.2f}, RSI={rsi_val:.1f}); Reversion target ${sma_20:.2f}",
                )
                log.info(f"QC MeanReversion SHORT for {sym}: Target=${sma_20:.2f}, SL=${sl_price:.2f}")

        return signals
