"""Adaptive Performance Feedback & Strategy Parameter Tweaker.

Monitors win rate, profit factor, and transaction fee drag, dynamically adjusting
strategy parameters (volume thresholds, profit targets, and trailing stops) to maximize net profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finb.log import get_logger

log = get_logger("tweaker")


@dataclass
class TradeOutcome:
    symbol: str
    entry_price: float
    exit_price: float
    fees_paid: float
    hold_hours: float
    net_pnl: float


@dataclass
class StrategyParameters:
    volume_threshold: float = 1.3
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.012
    max_hold_hours: float = 24.0


class AdaptiveTweaker:
    """Feedback loop that auto-tweaks strategy parameters based on empirical results."""

    def __init__(self, initial_params: StrategyParameters | None = None) -> None:
        self.params = initial_params or StrategyParameters()
        self.history: list[TradeOutcome] = []

    def record_trade(self, outcome: TradeOutcome) -> None:
        self.history.append(outcome)
        log.info(f"Recorded trade for {outcome.symbol}: Net PnL = ${outcome.net_pnl:+.2f}")

    def evaluate_and_tweak(self) -> StrategyParameters:
        """Analyzes recent trades and adjusts parameters to optimize net profit."""
        if len(self.history) < 3:
            return self.params

        recent = self.history[-10:]
        wins = [t for t in recent if t.net_pnl > 0]
        losses = [t for t in recent if t.net_pnl <= 0]
        win_rate = len(wins) / len(recent)

        total_gross_profit = sum(t.net_pnl for t in wins)
        total_fees = sum(t.fees_paid for t in recent)

        # Rule 1: High fee drag (> 20% of gross profits) -> tighten volume threshold to filter out weak setups
        if total_gross_profit > 0 and (total_fees / total_gross_profit) > 0.20:
            self.params.volume_threshold = min(2.0, self.params.volume_threshold + 0.1)
            log.info(f"Tweaker: Fee drag high ({total_fees:.2f}); increased volume threshold to {self.params.volume_threshold:.2f}x")

        # Rule 2: Win rate > 65% -> expand profit target to capture larger moves
        if win_rate >= 0.65:
            self.params.take_profit_pct = min(0.05, self.params.take_profit_pct + 0.003)
            log.info(f"Tweaker: Win rate high ({win_rate:.0%}); expanded take profit target to {self.params.take_profit_pct:.1%}")

        # Rule 3: Win rate < 40% -> tighten stop loss and lower hold time to exit faster
        elif win_rate < 0.40:
            self.params.stop_loss_pct = max(0.008, self.params.stop_loss_pct - 0.002)
            self.params.max_hold_hours = max(12.0, self.params.max_hold_hours - 2.0)
            log.info(f"Tweaker: Win rate low ({win_rate:.0%}); tightened stop loss to {self.params.stop_loss_pct:.1%} and hold time to {self.params.max_hold_hours:.0f}h")

        return self.params
