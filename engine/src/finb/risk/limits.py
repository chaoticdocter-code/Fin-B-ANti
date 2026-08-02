"""Position sizing, exposure limits, and the kill switch."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from finb.sim.constraints import Side


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard bounds. Deliberately tight — $500 has no capacity for a bad day.

    `max_position_pct` at 0.25 means at least four positions, which is not
    diversification so much as an upper bound on how wrong any single call can
    be.
    """

    capital: float = 500.0
    max_position_pct: float = 0.25
    max_gross_exposure: float = 1.0
    """1.0 = fully invested, never levered."""

    max_short_position_pct: float = 0.10
    """Tighter than the long cap, deliberately. A long can lose 100%; a short's
    loss is unbounded, and the position grows against you exactly as it moves
    against you. Half the long limit is not conservatism, it is the same amount
    of risk."""

    max_gross_short_pct: float = 0.30
    """Ceiling on total short exposure. Caps the damage from a market-wide
    squeeze, which is the scenario where every short goes wrong at once and
    per-position limits do not help."""

    short_stop_loss_pct: float = 0.15
    """A short with no stop is an unbounded liability. Advisory here — the
    execution layer is responsible for placing it."""

    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    max_trades_per_day: int = 25
    """A runaway-loop backstop. A strategy on multi-week holds should never
    approach this; if it does, something is broken rather than aggressive."""

    min_notional: float = 1.0
    vol_target_annual: float = 0.20
    """Annualised volatility the sizing aims at. Crypto routinely runs 60-80%,
    so this scales positions *down* hard by default."""


@dataclass
class RiskState:
    peak_equity: float
    equity: float
    day: date | None = None
    day_start_equity: float = 0.0
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    history: list[str] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.equity / self.day_start_equity - 1.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    qty: float
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class RiskEngine:
    """Gatekeeper for every order. Nothing reaches a venue without passing here."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            peak_equity=self.limits.capital,
            equity=self.limits.capital,
            day_start_equity=self.limits.capital,
        )

    # ------------------------------------------------------------------ #

    def update(self, ts: datetime, equity: float) -> None:
        """Mark the account and trip the kill switch if a limit is breached."""
        d = ts.date()
        if self.state.day != d:
            self.state.day = d
            self.state.day_start_equity = equity
            self.state.trades_today = 0

        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

        if self.state.halted:
            return

        if self.state.drawdown <= -self.limits.max_drawdown_pct:
            self._halt(
                f"drawdown {self.state.drawdown:.1%} breached the "
                f"{-self.limits.max_drawdown_pct:.0%} limit"
            )
        elif self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            self._halt(
                f"daily loss {self.state.daily_pnl_pct:.1%} breached the "
                f"{-self.limits.max_daily_loss_pct:.0%} limit"
            )

    @property
    def max_short_position_pct_effective(self) -> float:
        """Never looser than the long cap, whatever the config says."""
        return min(self.limits.max_short_position_pct, self.limits.max_position_pct)

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.history.append(reason)

    def reset_halt(self, acknowledgement: str) -> None:
        """Clear the kill switch. Requires a written reason, which is the point.

        Recovery is manual by design: an automatic reset would let the system
        resume trading into whatever condition tripped it.
        """
        if not acknowledgement.strip():
            raise ValueError("a written acknowledgement is required to resume trading")
        self.state.history.append(f"resumed: {acknowledgement}")
        self.state.halted = False
        self.state.halt_reason = ""

    # ------------------------------------------------------------------ #

    def position_size(
        self,
        equity: float,
        price: float,
        *,
        asset_volatility: float,
        periods_per_year: int = 365,
        conviction: float = 1.0,
    ) -> float:
        """Volatility-targeted size, capped by the concentration limit.

        `asset_volatility` is the per-period standard deviation of returns. A
        more volatile asset gets a smaller position for the same risk, which is
        what stops a crypto book being sized like an equity one.
        """
        if price <= 0 or equity <= 0 or asset_volatility <= 0:
            return 0.0

        annual_vol = asset_volatility * math.sqrt(periods_per_year)
        scale = min(1.0, self.limits.vol_target_annual / annual_vol)
        target = equity * self.limits.max_position_pct * scale * max(0.0, min(1.0, conviction))
        return target / price if target >= self.limits.min_notional else 0.0

    def check(
        self,
        *,
        side: Side,
        symbol: str,
        qty: float,
        price: float,
        equity: float,
        current_position_value: float = 0.0,
        gross_exposure: float = 0.0,
        current_qty: float = 0.0,
        gross_short: float = 0.0,
        allow_short: bool = False,
    ) -> RiskDecision:
        """Approve, shrink, or reject a proposed order.

        `current_qty` is signed: positive long, negative short. It is what
        distinguishes *selling to close* from *selling to open a short* — two
        orders that look identical on the wire and have opposite risk. Treating
        every sell as risk-reducing would wave through unlimited short exposure.
        """
        notional = qty * price

        if side is Side.SELL:
            closing = min(qty, max(0.0, current_qty))
            opening_short = qty - closing

            # Closing is always permitted, including while halted — otherwise the
            # kill switch would trap the very positions it fired over.
            if opening_short <= 1e-12:
                return RiskDecision(True, qty)

            if not allow_short:
                if closing <= 0:
                    return RiskDecision(False, 0.0, "short selling is not enabled")
                return RiskDecision(True, closing, "reduced to closing size; shorts disabled")

            if self.state.halted:
                if closing <= 0:
                    return RiskDecision(False, 0.0, f"halted: {self.state.halt_reason}")
                return RiskDecision(True, closing, "halted — closing only")

            short_notional = opening_short * price
            cap = equity * self.max_short_position_pct_effective
            existing_short = max(0.0, -current_qty) * price
            if existing_short + short_notional > cap:
                room = cap - existing_short
                if room < self.limits.min_notional:
                    return RiskDecision(
                        True, closing,
                        f"{symbol} already at the "
                        f"{self.max_short_position_pct_effective:.0%} short limit",
                    ) if closing > 0 else RiskDecision(
                        False, 0.0,
                        f"{symbol} already at the "
                        f"{self.max_short_position_pct_effective:.0%} short limit",
                    )
                opening_short = room / price
                short_notional = room

            gross_room = equity * self.limits.max_gross_short_pct - gross_short
            if short_notional > gross_room:
                if gross_room < self.limits.min_notional:
                    msg = (
                        f"gross short exposure at the "
                        f"{self.limits.max_gross_short_pct:.0%} ceiling"
                    )
                    return (
                        RiskDecision(True, closing, msg) if closing > 0
                        else RiskDecision(False, 0.0, msg)
                    )
                opening_short = gross_room / price

            return RiskDecision(True, closing + opening_short)

        if self.state.halted:
            return RiskDecision(False, 0.0, f"halted: {self.state.halt_reason}")

        if self.state.trades_today >= self.limits.max_trades_per_day:
            return RiskDecision(
                False, 0.0,
                f"daily trade cap of {self.limits.max_trades_per_day} reached — "
                "this is a runaway-loop backstop, not a strategy limit",
            )

        if notional < self.limits.min_notional:
            return RiskDecision(False, 0.0, f"below ${self.limits.min_notional} minimum notional")

        # Concentration.
        cap = equity * self.limits.max_position_pct
        if current_position_value + notional > cap:
            room = cap - current_position_value
            if room < self.limits.min_notional:
                return RiskDecision(
                    False, 0.0,
                    f"{symbol} already at the {self.limits.max_position_pct:.0%} "
                    "position limit",
                )
            qty = room / price
            notional = room

        # Gross exposure — no leverage, ever.
        room = equity * self.limits.max_gross_exposure - gross_exposure
        if notional > room:
            if room < self.limits.min_notional:
                return RiskDecision(False, 0.0, "fully invested; no leverage permitted")
            qty = room / price
            notional = room

        return RiskDecision(True, qty)

    def record_fill(self) -> None:
        self.state.trades_today += 1

    # ------------------------------------------------------------------ #

    def status(self) -> str:
        s = self.state
        if s.halted:
            return f"HALTED — {s.halt_reason}"
        return (
            f"ok — equity ${s.equity:,.2f}, drawdown {s.drawdown:.1%}, "
            f"{s.trades_today}/{self.limits.max_trades_per_day} trades today"
        )
