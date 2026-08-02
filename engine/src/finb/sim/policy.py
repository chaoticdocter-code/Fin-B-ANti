"""Holding-period policy — the one design parameter that decides feasibility.

The reviewer's sharpest point about the original plan was that holding period
was not a design parameter at all: it was whatever the search happened to
converge on. That is backwards. Cost is fixed per round trip and expected edge
grows with √time, so the holding period *is* the cost ratio. It should be set by
fiat, from the cost model, before any search runs.

`HoldingPolicy.from_costs` derives the minimum hold from the venue's actual
round-trip cost and the instrument's volatility, so the number moves when the
fee assumptions move instead of sitting in the code as a magic constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from finb.sim.constraints import AssetClass
from finb.sim.costs import CostModel, min_hold_days


@dataclass(frozen=True, slots=True)
class ExitCheck:
    allowed: bool
    days_held: float
    days_remaining: float
    reason: str


@dataclass(frozen=True, slots=True)
class HoldingPolicy:
    """Minimum holding period per asset class."""

    min_hold_equity: timedelta = timedelta(days=5)
    min_hold_crypto: timedelta = timedelta(days=15)

    def minimum(self, asset: AssetClass) -> timedelta:
        return self.min_hold_crypto if asset is AssetClass.CRYPTO else self.min_hold_equity

    @classmethod
    def from_costs(
        cls,
        equity_venue: CostModel,
        crypto_venue: CostModel,
        *,
        equity_daily_vol: float = 0.018,
        crypto_daily_vol: float = 0.035,
        coverage: float = 2.0,
        ic: float = 0.03,
        equity_floor_days: int = 5,
    ) -> HoldingPolicy:
        """Derive minimum holds from real cost assumptions.

        `equity_floor_days` exists because cheap equity costs imply a sub-day
        hold, and permitting that reintroduces the turnover drag that makes a
        $500 book unviable. Cost feasibility is a floor on the holding period,
        not a licence to trade at the floor.
        """
        eq = min_hold_days(
            equity_venue.round_trip_bps(500.0), equity_daily_vol, ic=ic, coverage=coverage
        )
        cr = min_hold_days(
            crypto_venue.round_trip_bps(500.0), crypto_daily_vol, ic=ic, coverage=coverage
        )
        return cls(
            min_hold_equity=timedelta(days=max(eq, equity_floor_days)),
            min_hold_crypto=timedelta(days=cr),
        )

    def earliest_exit(self, entry: datetime, asset: AssetClass) -> datetime:
        return entry + self.minimum(asset)

    def check_exit(self, entry: datetime, now: datetime, asset: AssetClass) -> ExitCheck:
        """Whether a position opened at `entry` may be closed at `now`.

        Stop-losses and risk kill-switches are expected to override this — the
        policy governs *discretionary* exits, not risk ones. A minimum hold that
        also traps a losing position is a way to turn a cost rule into a
        drawdown.
        """
        required = self.minimum(asset)
        held = now - entry
        remaining = required - held
        if held >= required:
            return ExitCheck(True, held.days, 0.0, "")
        return ExitCheck(
            allowed=False,
            days_held=held.total_seconds() / 86400.0,
            days_remaining=remaining.total_seconds() / 86400.0,
            reason=(
                f"minimum hold for {asset.value} is {required.days}d to cover "
                f"round-trip costs; {remaining.total_seconds() / 86400.0:.1f}d remaining"
            ),
        )

    def trades_per_year(self, asset: AssetClass, n_symbols: int = 1) -> float:
        """Observations per year at this policy — the sample-size budget.

        Breadth is the multiplier. One crypto pair at a 38-day hold yields ~9
        trades a year and can never be validated; twenty pairs yield ~180 and
        reach 100 observations in about seven months.
        """
        days = 365.0 if asset is AssetClass.CRYPTO else 252.0
        hold = self.minimum(asset).total_seconds() / 86400.0
        return (days / hold) * n_symbols if hold > 0 else float("inf")

    def years_to_observations(
        self, asset: AssetClass, n_symbols: int = 1, target: int = 100
    ) -> float:
        per_year = self.trades_per_year(asset, n_symbols)
        return target / per_year if per_year > 0 else float("inf")
