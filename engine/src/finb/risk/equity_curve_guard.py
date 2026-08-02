"""Portfolio Drawdown Control & Equity Curve Stop-Loss Guard.

Monitors portfolio equity curve drawdowns and enforces dynamic capital protection:
- **Drawdown >= 10%**: De-leverage scaling (positions scaled down by 50%).
- **Drawdown >= 15%**: Emergency Halt — force 100% cash until equity curve recovers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from finb.log import get_logger

log = get_logger("equity_curve_guard")


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed_leverage_scale: float  # 1.0 (normal), 0.5 (scaled down), 0.0 (halted)
    drawdown_pct: float
    halted: bool
    reason: str = ""


class EquityCurveGuard:
    """Monitors peak equity and enforces portfolio-level drawdown limits."""

    def __init__(
        self,
        deleverage_dd_pct: float = 0.10,
        halt_dd_pct: float = 0.15,
        initial_capital: float = 500.0,
    ) -> None:
        self.deleverage_dd_pct = deleverage_dd_pct
        self.halt_dd_pct = halt_dd_pct
        self.peak_equity = initial_capital
        self.equity_history: list[float] = [initial_capital]

    def update(self, ts: datetime, current_equity: float) -> GuardDecision:
        """Update equity state and return sizing multiplier / halt decision."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.equity_history.append(current_equity)
        dd = (current_equity - self.peak_equity) / self.peak_equity  # <= 0.0

        if abs(dd) >= self.halt_dd_pct:
            reason = f"Emergency Halt: portfolio drawdown {abs(dd):.1%} >= {self.halt_dd_pct:.1%}"
            log.warning(reason)
            return GuardDecision(
                allowed_leverage_scale=0.0,
                drawdown_pct=abs(dd),
                halted=True,
                reason=reason,
            )

        if abs(dd) >= self.deleverage_dd_pct:
            reason = f"De-leverage active: portfolio drawdown {abs(dd):.1%} >= {self.deleverage_dd_pct:.1%}"
            log.info(reason)
            return GuardDecision(
                allowed_leverage_scale=0.5,
                drawdown_pct=abs(dd),
                halted=False,
                reason=reason,
            )

        return GuardDecision(
            allowed_leverage_scale=1.0,
            drawdown_pct=abs(dd),
            halted=False,
            reason="Normal operation",
        )
