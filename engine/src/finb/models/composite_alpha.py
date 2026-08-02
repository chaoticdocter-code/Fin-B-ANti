"""Multi-Factor Composite Alpha Engine.

Combines four orthogonal quantitative factor families into a unified
cross-sectional rank score:
1. **Trend Factor**: Trailing momentum and price relative to SMA.
2. **Mean-Reversion Factor**: Inverted VWAP deviation and RSI z-score.
3. **Volatility-Regime Factor**: Inverse realized volatility (vol-scaling).
4. **Volume Accumulation Factor**: Relative volume (RVOL).

Each factor is standardized cross-sectionally to zero mean and unit variance
before weighting, preventing high-variance factors from dominating the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finb.log import get_logger

log = get_logger("composite_alpha")


@dataclass(frozen=True, slots=True)
class FactorWeights:
    trend: float = 0.40
    mean_reversion: float = 0.20
    volatility: float = 0.20
    volume: float = 0.20

    def __post_init__(self) -> None:
        total = self.trend + self.mean_reversion + self.volatility + self.volume
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Factor weights must sum to 1.0, got {total:.4f}")


@dataclass
class AlphaScoreResult:
    symbols: list[str]
    composite_scores: np.ndarray
    rankings: list[tuple[str, float]]


class CompositeAlphaEngine:
    """Standardizes and combines multi-factor quantitative signals."""

    def __init__(self, weights: FactorWeights | None = None) -> None:
        self.weights = weights or FactorWeights()

    @staticmethod
    def _zscore_cross_section(scores: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score standardization per factor."""
        valid = np.isfinite(scores)
        if valid.sum() < 2:
            return np.zeros_like(scores)

        mean = float(scores[valid].mean())
        std = float(scores[valid].std(ddof=1))
        if std == 0.0:
            return np.zeros_like(scores)

        out = np.zeros_like(scores)
        out[valid] = (scores[valid] - mean) / std
        return out

    def score_panel(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbols: list[str],
        lookback: int = 100,
    ) -> AlphaScoreResult:
        """Compute composite multi-factor score per symbol."""
        n_bars, n_sym = closes.shape
        if n_bars < lookback + 1:
            return AlphaScoreResult(symbols, np.zeros(n_sym), [])

        # 1. Trend factor: 100h momentum + price-to-SMA
        mom = closes[-1] / closes[-1 - lookback] - 1.0
        sma = np.mean(closes[-lookback:], axis=0)
        px_sma = closes[-1] / (sma + 1e-12) - 1.0
        f_trend = self._zscore_cross_section(mom + px_sma)

        # 2. Mean-reversion factor: VWAP deviation & RSI
        typical = (highs[-24:] + lows[-24:] + closes[-24:]) / 3.0
        vwap = np.sum(typical * volumes[-24:], axis=0) / (np.sum(volumes[-24:], axis=0) + 1e-12)
        vwap_dev = -(closes[-1] / (vwap + 1e-12) - 1.0)  # Inverted for mean reversion
        f_reversion = self._zscore_cross_section(vwap_dev)

        # 3. Volatility factor: Inverse realized vol
        rets = np.diff(closes[-24:], axis=0) / (closes[-24:-1] + 1e-12)
        vols = np.std(rets, axis=0)
        f_volatility = self._zscore_cross_section(1.0 / (vols + 1e-6))

        # 4. Volume factor: RVOL (current volume vs 20h mean volume)
        rvol_val = volumes[-1] / (np.mean(volumes[-20:], axis=0) + 1e-12)
        f_volume = self._zscore_cross_section(rvol_val)

        # Composite weighted sum
        composite = (
            self.weights.trend * f_trend
            + self.weights.mean_reversion * f_reversion
            + self.weights.volatility * f_volatility
            + self.weights.volume * f_volume
        )

        valid = np.isfinite(composite)
        order = np.argsort(np.where(valid, composite, -np.inf))[::-1]
        rankings = [(symbols[j], float(composite[j])) for j in order if valid[j]]

        return AlphaScoreResult(symbols, composite, rankings)
