"""ATR Trailing Stop & Chandelier Exit Engine.

Tracks position high-water mark (HWM) since entry and triggers stop-loss
exits when price drops below (HWM - multiplier * ATR).
"""

from __future__ import annotations

from dataclasses import dataclass

from finb.log import get_logger

log = get_logger("stops")


@dataclass(slots=True)
class StopCheckResult:
    triggered: bool
    stop_price: float
    reason: str = ""


class TrailingStopEngine:
    """Manages high-water mark tracking and trailing stop triggers."""

    def __init__(self, atr_multiplier: float = 3.0) -> None:
        self.atr_multiplier = atr_multiplier
        self._hwm: dict[str, float] = {}
        self._lwm: dict[str, float] = {}  # for short positions

    def update_high_water_mark(self, symbol: str, current_price: float, is_short: bool = False) -> float:
        """Update and return the position's high/low water mark."""
        if not is_short:
            prev = self._hwm.get(symbol, current_price)
            new_hwm = max(prev, current_price)
            self._hwm[symbol] = new_hwm
            return new_hwm

        prev = self._lwm.get(symbol, current_price)
        new_lwm = min(prev, current_price)
        self._lwm[symbol] = new_lwm
        return new_lwm

    def check_stop(
        self,
        symbol: str,
        current_price: float,
        atr_value: float,
        is_short: bool = False,
    ) -> StopCheckResult:
        """Check if current_price breaches the trailing ATR stop."""
        if is_short:
            lwm = self.update_high_water_mark(symbol, current_price, is_short=True)
            stop_price = lwm + (atr_value * self.atr_multiplier)
            triggered = current_price >= stop_price
            reason = (
                f"Short ATR stop breached: price ${current_price:,.2f} >= stop ${stop_price:,.2f}"
                if triggered
                else ""
            )
            return StopCheckResult(triggered, stop_price, reason)

        hwm = self.update_high_water_mark(symbol, current_price, is_short=False)
        stop_price = hwm - (atr_value * self.atr_multiplier)
        triggered = current_price <= stop_price
        reason = (
            f"Long ATR Chandelier stop breached: price ${current_price:,.2f} <= stop ${stop_price:,.2f}"
            if triggered
            else ""
        )
        return StopCheckResult(triggered, stop_price, reason)

    def position_closed(self, symbol: str) -> None:
        """Reset state when position closes."""
        self._hwm.pop(symbol, None)
        self._lwm.pop(symbol, None)
