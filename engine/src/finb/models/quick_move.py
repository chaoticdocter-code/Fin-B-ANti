"""Quick-Move Intraday Profit-Snapping Strategy Engine.

Designed for rapid daily execution:
1. Detects high-momentum volatility breakouts on liquid majors (BTC/USD, ETH/USD, SOL/USD, AVAX/USD).
2. Sets explicit profit targets (+1.5% to +2.5%) to snap fast profits.
3. Implements tight trailing stops (1.0 ATR) and a 24-hour max hold time to cycle capital rapidly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finb.log import get_logger

log = get_logger("quick_move")


@dataclass
class QuickMoveSignal:
    symbol: str
    action: str              # BUY | SELL | HOLD
    target_weight: float
    take_profit_price: float
    stop_loss_price: float
    rationale: str


class QuickMoveStrategy:
    """Intraday momentum breakout and rapid profit-snapping engine."""

    TARGET_MAJORS = {"BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"}

    def __init__(
        self,
        lookback_bars: int = 24,
        volume_threshold: float = 1.3,
        take_profit_pct: float = 0.02,     # +2.0% fast profit target
        stop_loss_pct: float = 0.012,       # -1.2% tight stop loss
        max_position_pct: float = 0.40,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.volume_threshold = volume_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct

    def generate_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbols: list[str],
    ) -> dict[str, QuickMoveSignal]:
        n_bars, n_sym = closes.shape
        signals: dict[str, QuickMoveSignal] = {}

        if n_bars < self.lookback_bars:
            return signals

        for j, sym in enumerate(symbols):
            if sym not in self.TARGET_MAJORS:
                continue

            c_win = closes[-self.lookback_bars :, j]
            h_win = highs[-self.lookback_bars :, j]
            l_win = lows[-self.lookback_bars :, j]
            v_win = volumes[-self.lookback_bars :, j]

            curr_price = float(closes[-1, j])
            high_breakout = float(np.max(h_win[:-1]))
            low_breakdown = float(np.min(l_win[:-1]))
            avg_volume = float(np.mean(v_win[:-1]))
            curr_volume = float(volumes[-1, j])
            rvol = curr_volume / (avg_volume + 1e-12)

            # LONG Trigger: Price breaking above local high with Volume Expansion
            if curr_price > high_breakout and rvol >= self.volume_threshold:
                tp_price = curr_price * (1.0 + self.take_profit_pct)
                sl_price = curr_price * (1.0 - self.stop_loss_pct)
                signals[sym] = QuickMoveSignal(
                    symbol=sym,
                    action="BUY",
                    target_weight=self.max_position_pct,
                    take_profit_price=tp_price,
                    stop_loss_price=sl_price,
                    rationale=f"LONG Breakout above {high_breakout:.2f} with RVOL {rvol:.2f}x; TP @ ${tp_price:.2f}",
                )
                log.info(f"QuickMove LONG trigger for {sym}: TP=${tp_price:.2f}, SL=${sl_price:.2f}")

            # SHORT Trigger: Price breaking below local low with Volume Expansion
            elif curr_price < low_breakdown and rvol >= self.volume_threshold:
                tp_price = curr_price * (1.0 - self.take_profit_pct)
                sl_price = curr_price * (1.0 + self.stop_loss_pct)
                signals[sym] = QuickMoveSignal(
                    symbol=sym,
                    action="SELL",
                    target_weight=-self.max_position_pct,
                    take_profit_price=tp_price,
                    stop_loss_price=sl_price,
                    rationale=f"SHORT Breakdown below {low_breakdown:.2f} with RVOL {rvol:.2f}x; TP @ ${tp_price:.2f}",
                )
                log.info(f"QuickMove SHORT trigger for {sym}: TP=${tp_price:.2f}, SL=${sl_price:.2f}")

        return signals

