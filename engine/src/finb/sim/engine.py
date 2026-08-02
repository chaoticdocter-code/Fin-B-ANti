"""The shadow book: a simulated $500 account, marked bar by bar.

Bar-driven rather than event-driven, deliberately. At the holding periods
[[0006]] forces — 8 days for equities, 38 for crypto — an event loop simulating
queue position would be precision theatre on top of a decision made once a
month. What actually determines whether the result is honest at this size is
whether costs, minimum notional, and the holding policy are enforced. Those are.

Everything charged here is a cost Alpaca's paper engine does **not** charge:
spread, slippage, regulatory fees, and crypto commission. Paper fills happen at
NBBO with effectively unlimited size and no queue, so a strategy scored on
Alpaca's own P&L is being scored against a simulator it can farm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from finb.sim.constraints import AssetClass, Side
from finb.sim.costs import CostModel
from finb.sim.policy import HoldingPolicy


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float
    entry_ts: datetime
    entry_price: float
    cost_basis: float

    def value(self, price: float) -> float:
        return self.qty * price


@dataclass(slots=True)
class Trade:
    ts: datetime
    symbol: str
    side: Side
    qty: float
    price: float
    notional: float
    cost: float
    reason: str = ""


@dataclass
class ShadowBook:
    """One simulated account. Long-only, fully-funded, no leverage.

    Long-only because a $500 account cannot short: equities need $2,000 minimum
    margin equity, and crypto here is spot. A simulator that permits shorts will
    happily evolve strategies that cannot be run.
    """

    capital: float
    cost_model: CostModel
    holding_policy: HoldingPolicy = field(default_factory=HoldingPolicy)
    asset_class: AssetClass = AssetClass.CRYPTO
    min_notional: float = 1.0

    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    trades: list[Trade] = field(default_factory=list, init=False)
    equity_history: list[tuple[datetime, float]] = field(default_factory=list, init=False)
    blocked_by_hold: int = field(default=0, init=False)
    blocked_by_notional: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.capital)

    # ------------------------------------------------------------------ #

    def equity(self, prices: dict[str, float]) -> float:
        held = sum(
            p.value(prices[s]) for s, p in self.positions.items() if s in prices
        )
        return self.cash + held

    def mark(self, ts: datetime, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.equity_history.append((ts, eq))
        return eq

    # ------------------------------------------------------------------ #

    def rebalance(
        self,
        ts: datetime,
        prices: dict[str, float],
        target_weights: dict[str, float],
    ) -> None:
        """Move toward `target_weights` (fractions of equity), paying costs.

        Sells run before buys so proceeds fund purchases — the same ordering a
        cash account is forced into by settlement.
        """
        total = sum(max(0.0, w) for w in target_weights.values())
        if total > 1.0 + 1e-9:
            # Never lever up. Scale down instead of silently borrowing.
            target_weights = {k: v / total for k, v in target_weights.items()}

        equity = self.equity(prices)
        if equity <= 0:
            return

        desired = {
            s: max(0.0, w) * equity / prices[s]
            for s, w in target_weights.items()
            if s in prices and prices[s] > 0
        }

        # --- sells -------------------------------------------------------
        for symbol in list(self.positions):
            if symbol not in prices:
                continue
            pos = self.positions[symbol]
            want = desired.get(symbol, 0.0)
            if want >= pos.qty - 1e-12:
                continue

            check = self.holding_policy.check_exit(pos.entry_ts, ts, self.asset_class)
            if not check.allowed:
                self.blocked_by_hold += 1
                continue

            qty = pos.qty - want
            if qty * prices[symbol] < self.min_notional:
                self.blocked_by_notional += 1
                continue
            self._execute(ts, symbol, Side.SELL, qty, prices[symbol])

        # --- buys --------------------------------------------------------
        for symbol, want in desired.items():
            have = self.positions[symbol].qty if symbol in self.positions else 0.0
            qty = want - have
            if qty <= 1e-12:
                continue
            notional = qty * prices[symbol]
            if notional < self.min_notional:
                self.blocked_by_notional += 1
                continue
            # Estimate cost so the buy does not overdraw the account.
            est = self.cost_model.fill_cost(notional, side=Side.BUY, qty=qty).total
            if notional + est > self.cash:
                affordable = max(0.0, self.cash - est) / prices[symbol]
                if affordable * prices[symbol] < self.min_notional:
                    continue
                qty = affordable
            self._execute(ts, symbol, Side.BUY, qty, prices[symbol])

    def _execute(
        self, ts: datetime, symbol: str, side: Side, qty: float, price: float
    ) -> None:
        notional = qty * price
        cost = self.cost_model.fill_cost(notional, side=side, qty=qty).total

        if side is Side.BUY:
            self.cash -= notional + cost
            if symbol in self.positions:
                p = self.positions[symbol]
                p.cost_basis += notional + cost
                p.qty += qty
            else:
                self.positions[symbol] = Position(symbol, qty, ts, price, notional + cost)
        else:
            self.cash += notional - cost
            p = self.positions[symbol]
            p.qty -= qty
            if p.qty <= 1e-12:
                del self.positions[symbol]

        self.trades.append(Trade(ts, symbol, side, qty, price, notional, cost))

    # ------------------------------------------------------------------ #

    @property
    def equity_curve(self) -> np.ndarray:
        return np.array([e for _, e in self.equity_history], dtype=float)

    @property
    def returns(self) -> np.ndarray:
        eq = self.equity_curve
        if eq.size < 2:
            return np.array([])
        return np.diff(eq) / eq[:-1]

    @property
    def total_costs(self) -> float:
        return sum(t.cost for t in self.trades)

    def stats(self) -> dict:
        eq = self.equity_curve
        r = self.returns
        if eq.size < 2:
            return {"trades": len(self.trades), "final_equity": self.cash}

        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak

        gross = float(sum(t.notional for t in self.trades))
        return {
            "final_equity": float(eq[-1]),
            "total_return": float(eq[-1] / eq[0] - 1.0),
            "max_drawdown": float(dd.min()),
            "trades": len(self.trades),
            "total_costs": self.total_costs,
            "cost_pct_of_capital": self.total_costs / self.capital,
            "gross_traded": gross,
            "turnover_x": gross / self.capital if self.capital else 0.0,
            "blocked_by_hold": self.blocked_by_hold,
            "blocked_by_notional": self.blocked_by_notional,
            "periods": int(r.size),
        }
