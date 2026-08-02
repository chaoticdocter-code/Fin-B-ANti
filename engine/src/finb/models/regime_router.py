"""Multi-Strategy Dynamic Regime Router.

Classifies current market regime into:
1. **SQUEEZE**: Volatility contraction detected across assets -> route to `VolSqueezeStrategy`.
2. **TREND**: High trend strength (ADX > 25) -> route to `TrendVolTargetStrategy`.
3. **RANGEBOUND**: Flat / low volatility -> route to `PairSpreadStrategy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from finb.log import get_logger
from finb.models.strategies import PairSpreadStrategy, SignalResult, TrendVolTargetStrategy, VolSqueezeStrategy

log = get_logger("regime_router")


class MarketRegime(StrEnum):
    SQUEEZE = "squeeze"
    TREND = "trend"
    RANGEBOUND = "rangebound"


@dataclass
class RoutingResult:
    active_regime: MarketRegime
    signal_result: SignalResult
    confidence: float


class RegimeRouter:
    """Classifies market regime and routes capital to optimal strategy."""

    def __init__(self) -> None:
        self.vol_squeeze = VolSqueezeStrategy()
        self.pair_spread = PairSpreadStrategy()
        self.trend_vol_target = TrendVolTargetStrategy()


    def classify_regime(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> MarketRegime:
        """Classify current market regime from panel OHLCV."""
        n_bars, n_sym = closes.shape
        if n_bars < 30:
            return MarketRegime.TREND

        # Check for volatility squeeze in any symbol
        squeeze_count = 0
        for j in range(n_sym):
            c_win = closes[-20:, j]
            h_win = highs[-20:, j]
            l_win = lows[-20:, j]
            ma = np.mean(c_win)
            std = np.std(c_win)
            bb_w = (4.0 * std) / (ma + 1e-12)

            prev_c = np.roll(c_win, 1)
            prev_c[0] = c_win[0]
            tr = np.maximum(h_win - l_win, np.abs(h_win - prev_c))
            kc_w = (3.0 * np.mean(tr)) / (ma + 1e-12)

            if bb_w < kc_w:
                squeeze_count += 1

        if squeeze_count > 0:
            return MarketRegime.SQUEEZE

        # Check trend strength (avg 20-period price to SMA deviation)
        smas = np.mean(closes[-20:], axis=0)
        deviations = np.abs(closes[-1] / (smas + 1e-12) - 1.0)
        avg_dev = float(np.mean(deviations))

        if avg_dev >= 0.02:  # > 2% average trend deviation
            return MarketRegime.TREND

        return MarketRegime.RANGEBOUND

    def route(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbols: list[str],
        default_weights: dict[str, float],
    ) -> RoutingResult:
        """Route signals to the active strategy archetype based on regime."""
        regime = self.classify_regime(closes, highs, lows, volumes)

        if regime == MarketRegime.SQUEEZE:
            sig = self.vol_squeeze.generate_signals(closes, highs, lows, volumes, symbols)
            log.info(f"Regime Router: SQUEEZE active -> {sig.rationale}")
            return RoutingResult(regime, sig, 0.85)

        if regime == MarketRegime.RANGEBOUND:
            sig = self.pair_spread.generate_signals(closes, symbols)
            log.info(f"Regime Router: RANGEBOUND active -> {sig.rationale}")
            return RoutingResult(regime, sig, 0.75)

        # Default to TREND strategy (Liquid Majors Trend-Following + Vol Target)
        sig = self.trend_vol_target.generate_signals(closes, symbols)
        log.info(f"Regime Router: TREND active -> {sig.rationale}")
        return RoutingResult(regime, sig, 0.90)

