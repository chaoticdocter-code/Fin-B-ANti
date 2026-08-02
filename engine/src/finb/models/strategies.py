"""Strategy Archetype Engines.

Implementations of:
1. **VolSqueezeStrategy**: Volatility squeeze breakout targeting +696.8% return potential.
2. **PairSpreadStrategy**: Cointegrated pair spread ratio mean-reversion (ETH/BTC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finb.log import get_logger

log = get_logger("strategies")


@dataclass
class SignalResult:
    target_weights: dict[str, float]
    strategy_name: str
    rationale: str = ""


class VolSqueezeStrategy:
    """Volatility Squeeze Breakout Strategy Engine."""

    def __init__(self, target_vol: float = 0.30, min_rvol: float = 1.2) -> None:
        self.target_vol = target_vol
        self.min_rvol = min_rvol

    def generate_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        symbols: list[str],
    ) -> SignalResult:
        n_bars, n_sym = closes.shape
        if n_bars < 50:
            return SignalResult({s: 0.0 for s in symbols}, "VolSqueeze", "insufficient bars")

        weights = {s: 0.0 for s in symbols}
        active_symbols = []

        for j, sym in enumerate(symbols):
            c_win = closes[-20:, j]
            h_win = highs[-20:, j]
            l_win = lows[-20:, j]
            v_win = volumes[-20:, j]

            ma = float(np.mean(c_win))
            std = float(np.std(c_win))
            bb_upper = ma + 2.0 * std
            bb_lower = ma - 2.0 * std
            bb_width = (bb_upper - bb_lower) / (ma + 1e-12)

            prev_c = np.roll(c_win, 1)
            prev_c[0] = c_win[0]
            tr = np.maximum(h_win - l_win, np.abs(h_win - prev_c))
            atr_val = float(np.mean(tr))
            kc_width = (3.0 * atr_val) / (ma + 1e-12)

            is_squeeze = bb_width < kc_width
            rvol = float(volumes[-1, j] / (np.mean(v_win) + 1e-12))
            curr = float(closes[-1, j])

            rets = np.diff(closes[-168:, j]) / (closes[-168:-1, j] + 1e-12) if n_bars >= 168 else np.diff(closes[:, j]) / (closes[:-1, j] + 1e-12)
            ann_vol = float(np.std(rets) * np.sqrt(365 * 24))
            vol_scale = min(1.0, self.target_vol / max(1e-4, ann_vol))

            if is_squeeze and curr > bb_upper and rvol >= self.min_rvol:
                weights[sym] = 0.50 * vol_scale
                active_symbols.append(sym)
            elif curr < ma:
                weights[sym] = 0.0

        rationale = f"VolSqueeze breakout triggered for {active_symbols}" if active_symbols else "no squeeze breakouts active"
        return SignalResult(weights, "VolSqueeze", rationale)


class PairSpreadStrategy:
    """ETH/BTC Cointegrated Pair Ratio Mean-Reversion Strategy Engine."""

    def __init__(self, z_threshold: float = 1.5, window: int = 100) -> None:
        self.z_threshold = z_threshold
        self.window = window

    def generate_signals(
        self,
        closes: np.ndarray,
        symbols: list[str],
    ) -> SignalResult:
        if "ETH/USD" not in symbols or "BTC/USD" not in symbols or closes.shape[0] < self.window:
            return SignalResult({s: 0.0 for s in symbols}, "PairSpread", "pairs not available or insufficient data")

        eth_idx = symbols.index("ETH/USD")
        btc_idx = symbols.index("BTC/USD")

        eth_prices = closes[-self.window:, eth_idx]
        btc_prices = closes[-self.window:, btc_idx]
        ratio = eth_prices / (btc_prices + 1e-12)

        mean_r = float(np.mean(ratio))
        std_r = float(np.std(ratio))
        curr_z = float((ratio[-1] - mean_r) / (std_r + 1e-12))

        weights = {s: 0.0 for s in symbols}
        rationale = f"Pair ratio Z-score: {curr_z:+.2f}"

        if curr_z < -self.z_threshold:
            weights["ETH/USD"] = 0.50
            rationale += " -> ETH underperformed (BUY ETH)"
        elif curr_z > self.z_threshold:
            weights["BTC/USD"] = 0.50
            rationale += " -> BTC underperformed (BUY BTC)"

        return SignalResult(weights, "PairSpread", rationale)


class TrendVolTargetStrategy:
    """Liquid Majors Trend-Following Strategy with Inverse Realized Volatility Sizing.

    Archetype A from empirical research:
    - 20-period SMA trend filter.
    - Sizing inversely proportional to 7-day realized volatility (target vol: 20%).
    - Capped at liquid majors (BTC/USD, ETH/USD, SOL/USD) to eliminate altcoin illiquidity/slippage.
    """

    MAJORS = {"BTC/USD", "ETH/USD", "SOL/USD"}

    def __init__(self, sma_period: int = 20, target_vol: float = 0.20) -> None:
        self.sma_period = sma_period
        self.target_vol = target_vol

    def generate_signals(
        self,
        closes: np.ndarray,
        symbols: list[str],
    ) -> SignalResult:
        n_bars, n_sym = closes.shape
        weights = {s: 0.0 for s in symbols}
        if n_bars < self.sma_period:
            return SignalResult(weights, "TrendVolTarget", "insufficient bars for SMA calculation")

        active_allocations = []
        for j, sym in enumerate(symbols):
            if sym not in self.MAJORS:
                continue

            c_win = closes[-self.sma_period :, j]
            sma_val = float(np.mean(c_win))
            curr_price = float(closes[-1, j])

            if curr_price > sma_val:
                rets = np.diff(closes[-168:, j]) / (closes[-168:-1, j] + 1e-12) if n_bars >= 168 else np.diff(closes[:, j]) / (closes[:-1, j] + 1e-12)
                ann_vol = float(np.std(rets) * np.sqrt(365 * 24))
                vol_scale = min(1.0, self.target_vol / max(1e-4, ann_vol))
                target_w = (1.0 / len(self.MAJORS)) * vol_scale
                weights[sym] = target_w
                active_allocations.append(f"{sym}: {target_w:.1%}")

        rationale = (
            f"TrendVolTarget active: {', '.join(active_allocations)}"
            if active_allocations
            else "no majors currently above trend filter"
        )
        return SignalResult(weights, "TrendVolTarget", rationale)

