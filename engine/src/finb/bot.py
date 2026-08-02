"""The live bot: signal, reconcile, execute.

> **What this proves and what it does not.** The strategy wired in here is
> cross-sectional momentum, which [[2026-08-01 Baseline]] tested and the gate
> **rejected** — deflated Sharpe 0.132, and 23% of zero-skill strategies did
> better. Running it live verifies the *plumbing*: that a signal becomes sized,
> risk-checked orders and that the book converges on its target. It is not a bet,
> and any P&L it produces is noise. Nothing should be promoted on the strength of
> what this loop does.

Design notes:

- **Entry times are tracked locally.** Alpaca positions do not report when they
  were opened, and the holding policy needs that. `PositionState` persists it in
  the data directory and survives restarts, because a bot that forgets when it
  entered will churn straight through its minimum hold.
- **Sells run before buys.** Proceeds fund purchases, which mirrors the ordering
  a cash account is forced into anyway.
- **Dry run is the default.** `run_bot` will not send an order unless explicitly
  told to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from finb.config import Settings
from finb.data.lake import BarLake, Timeframe
from finb.execution.alpaca_paper import AlpacaBroker
from finb.execution.base import OrderRequest
from finb.log import get_logger
from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import AssetClass, Side
from finb.sim.costs import ALPACA_CRYPTO
from finb.sim.policy import HoldingPolicy
from finb.sim.runner import build_panel, momentum_scores

log = get_logger("bot")

STABLECOINS = {"USDC", "USDT", "USDG", "DAI", "PYUSD", "BUSD", "TUSD", "USDP"}


class PositionState:
    """Remembers when each position was opened, so holding rules can be applied."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("position state was unreadable; starting fresh")

    def entry_time(self, symbol: str) -> datetime | None:
        raw = self._entries.get(AlpacaBroker.canonical(symbol))
        return datetime.fromisoformat(raw) if raw else None

    def last_entry_time(self) -> datetime | None:
        """Timestamp of the most recently opened position."""
        if not self._entries:
            return None
        times = [datetime.fromisoformat(v) for v in self._entries.values()]
        return max(times) if times else None

    def opened(self, symbol: str, when: datetime) -> None:
        self._entries.setdefault(AlpacaBroker.canonical(symbol), when.isoformat())
        self._save()

    def closed(self, symbol: str) -> None:
        self._entries.pop(AlpacaBroker.canonical(symbol), None)
        self._save()

    def reconcile(self, held: set[str]) -> None:
        """Drop entries for positions that no longer exist."""
        canonical = {AlpacaBroker.canonical(s) for s in held}
        for sym in list(self._entries):
            if sym not in canonical:
                del self._entries[sym]
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")


@dataclass
class Decision:
    symbol: str
    action: str            # buy | sell | hold | blocked
    current_value: float
    target_value: float
    reason: str = ""
    sent_value: float = 0.0
    """What was actually committed after risk sizing — not what was wanted.

    These diverge constantly: with capital locked in positions the holding
    policy will not release, a $250 target can become a $0.50 order. Reporting
    the target as though it were the order makes a book look invested when it is
    not."""


@dataclass
class BotRun:
    started_at: datetime
    dry_run: bool
    budget: float
    equity: float
    universe: int
    ranked: list[tuple[str, float]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    orders_sent: int = 0
    orders_blocked: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def target_symbols(self) -> list[str]:
        return [d.symbol for d in self.decisions if d.target_value > 0]


def load_universe_bars(s: Settings, min_bars: int = 400) -> dict:
    lake = BarLake(s.finb_data_dir)
    bars = {}
    for sym in lake.symbols(Timeframe.D1, AssetClass.CRYPTO):
        if sym.split("/")[0] in STABLECOINS:
            continue
        df = lake.read(sym, Timeframe.D1, asset=AssetClass.CRYPTO)
        if df.height >= min_bars:
            bars[sym] = df
    return bars


def run_bot(
    s: Settings,
    *,
    dry_run: bool = True,
    top_n: int = 4,
    lookback: int = 60,
    skip: int = 7,
    min_bars: int = 400,
) -> BotRun:
    """One decision cycle: score the cross-section, reconcile, execute."""
    now = datetime.now(UTC)

    risk = RiskEngine(RiskLimits(capital=s.finb_capital_usd, max_position_pct=1.0 / top_n))
    broker = AlpacaBroker(s, risk, dry_run=dry_run, allocation=s.finb_capital_usd)
    state = PositionState(s.finb_data_dir / "artifacts" / "positions.json")
    policy = HoldingPolicy.from_costs(ALPACA_CRYPTO, ALPACA_CRYPTO)

    snap = broker.account()
    budget = broker.budget(snap.equity)
    risk.update(now, budget)

    run = BotRun(
        started_at=now,
        dry_run=dry_run,
        budget=budget,
        equity=snap.equity,
        universe=0,
    )

    if not snap.is_paper:
        run.warnings.append("broker is NOT in paper mode")
        return run

    # --- signal -----------------------------------------------------------
    bars = load_universe_bars(s, min_bars=min_bars)
    run.universe = len(bars)
    if len(bars) < top_n + 1:
        run.warnings.append(f"only {len(bars)} symbols with enough history")
        return run

    # Screen for liquidity BEFORE ranking. A daily bar exists for every listed
    # pair; flow does not. Measured on this venue, CRV had not printed a trade
    # in 40 hours while its daily bars looked perfectly normal — ranking it and
    # then failing to fill silently substitutes a different strategy for the one
    # that was tested.
    from finb.data.liquidity import screen

    try:
        tradeable, rejected = screen(list(bars))
        for sym, why in rejected.items():
            run.warnings.append(f"{sym} screened out: {why}")
        keep = {AlpacaBroker.canonical(s) for s in tradeable}
        bars = {s: df for s, df in bars.items() if AlpacaBroker.canonical(s) in keep}
        run.universe = len(bars)
        effective_top_n = min(top_n, len(bars))
        if effective_top_n < 1:
            run.warnings.append(
                f"only {len(bars)} symbols are liquid enough to trade right now"
            )
            return run
    except Exception as e:  # noqa: BLE001
        run.warnings.append(f"liquidity screen unavailable ({type(e).__name__}) — not ranking")
        return run

    ts, symbols, closes = build_panel(bars)
    scores = momentum_scores(lookback=lookback, skip=skip)(closes)
    valid = np.isfinite(scores)
    if valid.sum() < effective_top_n:
        run.warnings.append("not enough valid scores to rank")
        return run

    order_idx = np.argsort(np.where(valid, scores, -np.inf))[::-1]
    run.ranked = [(symbols[j], float(scores[j])) for j in order_idx if valid[j]]
    winners = [symbols[j] for j in order_idx[:effective_top_n]]
    log.info(f"panel {closes.shape[0]}x{closes.shape[1]} to {ts[-1].date()}; top: {winners}")

    target_value = budget / effective_top_n
    held = {AlpacaBroker.canonical(p.symbol): p for p in snap.positions}
    state.reconcile(set(held))

    # --- sells: anything not in the target set ---------------------------
    wanted = {AlpacaBroker.canonical(w) for w in winners}
    for canon, pos in held.items():
        if canon in wanted:
            continue
        entry = state.entry_time(canon)
        if entry is not None:
            check = policy.check_exit(entry, now, AssetClass.CRYPTO)
            if not check.allowed:
                run.decisions.append(
                    Decision(pos.symbol, "blocked", abs(pos.market_value), 0.0, check.reason)
                )
                continue
        r = broker.submit(
            OrderRequest(pos.symbol, Side.SELL, qty=abs(pos.qty),
                         asset_class=AssetClass.CRYPTO)
        )
        run.decisions.append(
            Decision(pos.symbol, "sell" if r else "blocked", abs(pos.market_value), 0.0,
                     r.reason or ("sent" if r else "rejected"),
                     sent_value=r.submitted_notional)
        )
        if r:
            run.orders_sent += 1
            if not dry_run:
                state.closed(canon)
        else:
            run.orders_blocked += 1

    # --- buys: top up toward the target ----------------------------------
    #
    # Capital freed by the sells above is not available yet, so plan against
    # what is actually spare. A fully-invested book whose positions are all
    # inside their minimum hold cannot rebalance at all — the right response is
    # to say so, not to fire off dust orders that the risk engine shrinks to the
    # minimum notional. Observed: two $250 targets became two $1.00 orders.
    committed = sum(abs(p.market_value) for p in snap.positions)
    room = max(0.0, budget - committed)
    MEANINGFUL = 0.25   # fraction of the target worth transacting for

    for sym in winners:
        canon = AlpacaBroker.canonical(sym)
        current = abs(held[canon].market_value) if canon in held else 0.0
        gap = target_value - current

        if gap < max(1.0, target_value * 0.1):
            run.decisions.append(
                Decision(sym, "hold", current, target_value, "already at target")
            )
            continue

        fundable = min(gap, room)
        if fundable < max(1.0, target_value * MEANINGFUL):
            run.decisions.append(
                Decision(
                    sym, "blocked", current, target_value,
                    f"only ${room:,.2f} spare of ${budget:,.2f} — capital is committed "
                    "to positions still inside their minimum hold",
                )
            )
            run.orders_blocked += 1
            continue
        gap = fundable

        r = broker.submit(
            OrderRequest(sym, Side.BUY, notional=round(gap, 2),
                         asset_class=AssetClass.CRYPTO)
        )
        run.decisions.append(
            Decision(sym, "buy" if r else "blocked", current, target_value,
                     r.reason or ("sent" if r else "rejected"),
                     sent_value=r.submitted_notional)
        )
        if r:
            run.orders_sent += 1
            room -= r.submitted_notional
            if not dry_run:
                state.opened(canon, now)
        else:
            run.orders_blocked += 1

    return run
