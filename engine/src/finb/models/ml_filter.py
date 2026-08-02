"""LightGBM Machine Learning Signal Filter under Purged Cross-Validation.

Filters out low-confidence trade signals where out-of-fold predicted
probability of outperformance is below the probability threshold (P < 0.55).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finb.log import get_logger
from finb.models.dataset import build_panel_dataset
from finb.models.gbdt import train_purged

log = get_logger("ml_filter")


@dataclass
class MLFilterResult:
    allowed: bool
    confidence: float
    reason: str = ""


class MLSignalFilter:
    """Evaluates LightGBM out-of-fold probability predictions on bar features."""

    def __init__(self, min_confidence: float = 0.55) -> None:
        self.min_confidence = min_confidence

    def evaluate_signal(
        self,
        bars: dict,
        symbol: str,
        horizon: int = 24,
    ) -> MLFilterResult:
        """Evaluate machine learning probability filter on symbol bars."""
        if symbol not in bars or len(bars[symbol]) < 200:
            return MLFilterResult(allowed=True, confidence=0.50, reason="insufficient history for ML filter")

        try:
            ds = build_panel_dataset({symbol: bars[symbol]}, horizon=horizon)
            if ds.X.shape[0] < 100:
                return MLFilterResult(allowed=True, confidence=0.50, reason="dataset too short")

            res = train_purged(ds.X, ds.y, ds.t1_idx, params={"num_leaves": 15, "max_depth": 4})
            prob = float(res.oof[np.isfinite(res.oof)][-1]) if res.oof.size > 0 else 0.50

            allowed = prob >= self.min_confidence
            reason = (
                f"ML Filter APPROVED (confidence {prob:.1%} >= {self.min_confidence:.1%})"
                if allowed
                else f"ML Filter BLOCKED (confidence {prob:.1%} < {self.min_confidence:.1%})"
            )
            log.info(f"{symbol}: {reason}")
            return MLFilterResult(allowed=allowed, confidence=prob, reason=reason)
        except Exception as e:  # noqa: BLE001
            log.warning(f"ML filter error for {symbol}: {type(e).__name__} ({e})")
            return MLFilterResult(allowed=True, confidence=0.50, reason=f"ML filter fallback ({type(e).__name__})")
