"""Live monitor: scan every minute, trade on the strategy's own clock.

Scanning and trading are separate decisions and this module exists to keep them
that way. Watching the book minute by minute is cheap and informative. *Acting*
minute by minute is not:

| Hold | Gross edge | Cost | Net per trade |
|---|---|---|---|
| 1 minute | 0.48 bps | 57 bps | **-56.5 bps** |
| 1 hour | 3.75 bps | 57 bps | **-53.2 bps** |
| 38 days | 113 bps | 57 bps | +56.3 bps |

Expected edge grows with the square root of time; the fee does not move. At
hourly turnover a $500 book halves in roughly five sessions — arithmetic, not
bad luck.

So this loop **never places an order**. It refreshes prices, recomputes the
ranking, and reports what changed and what the holding policy would permit. When
a position's minimum hold expires it says so, and `finb run --live` is the
separate, deliberate act that trades.

One more thing it does deliberately: it logs **one** observation for the whole
watch session rather than one per scan. Sixty refreshes an hour are not sixty
hypotheses. But if you *act* on something you see here, that is a trial, and it
belongs in the ledger — see `0009`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from finb.bot import PositionState, load_universe_bars
from finb.config import Settings
from finb.execution.alpaca_paper import AlpacaBroker
from finb.log import get_logger
from finb.risk import RiskEngine, RiskLimits
from finb.sim.constraints import AssetClass
from finb.sim.costs import ALPACA_CRYPTO
from finb.sim.policy import HoldingPolicy
from finb.sim.runner import build_panel, momentum_scores

log = get_logger("watch")


@dataclass
class Scan:
    at: datetime
    equity: float
    gross_exposure: float
    positions: list[dict] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)
    ranking_changed: bool = False
    unlocked: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def unrealised(self) -> float:
        return sum(p["unrealised"] for p in self.positions)


# Longest-first so USDT/USDC are matched before USD.
_QUOTES = ("USDT", "USDC", "USDG", "USD", "BTC")


def to_data_symbol(symbol: str) -> str:
    """``AAVEUSD`` -> ``AAVE/USD``.

    Positions come back unslashed while the market-data API requires the slash;
    passing a position symbol straight through returns no price, which silently
    leaves every mark stale.
    """
    if "/" in symbol:
        return symbol
    upper = symbol.upper()
    for q in _QUOTES:
        if upper.endswith(q) and len(upper) > len(q):
            return f"{upper[: -len(q)]}/{q}"
    return symbol


def latest_prices(symbols: list[str]) -> dict[str, float]:
    """One multi-symbol request per scan rather than one per symbol."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestTradeRequest

    wanted = sorted({to_data_symbol(s) for s in symbols})
    if not wanted:
        return {}
    client = CryptoHistoricalDataClient()
    resp = client.get_crypto_latest_trade(
        CryptoLatestTradeRequest(symbol_or_symbols=wanted)
    )
    return {sym: float(t.price) for sym, t in resp.items()}


def scan_once(
    s: Settings,
    *,
    top_n: int = 4,
    lookback: int = 60,
    skip: int = 7,
    previous: list[str] | None = None,
) -> Scan:
    """One pass. Reads only — no order is ever constructed here."""
    now = datetime.now(UTC)
    risk = RiskEngine(RiskLimits(capital=s.finb_capital_usd, max_position_pct=1.0 / top_n))
    broker = AlpacaBroker(s, risk, dry_run=True, allocation=s.finb_capital_usd)
    state = PositionState(s.finb_data_dir / "artifacts" / "positions.json")
    policy = HoldingPolicy.from_costs(ALPACA_CRYPTO, ALPACA_CRYPTO)

    snap = broker.account()
    budget = broker.budget(snap.equity)
    risk.update(now, budget)

    scan = Scan(at=now, equity=budget, gross_exposure=snap.gross_exposure)

    # Alpaca re-marks positions on its own cadence, not per tick — six scans
    # across two minutes returned byte-identical P&L while live prices had
    # clearly moved. For a one-minute monitor that is stale enough to mislead,
    # so positions are re-marked here from live trades and only the entry price
    # comes from the broker.
    try:
        marks = latest_prices([p.symbol for p in snap.positions])
    except Exception as e:  # noqa: BLE001
        marks = {}
        scan.alerts.append(f"live marks unavailable ({type(e).__name__}); showing broker marks")

    def live_price(symbol: str) -> float | None:
        target = AlpacaBroker.canonical(symbol)
        return next(
            (px for sym, px in marks.items() if AlpacaBroker.canonical(sym) == target), None
        )

    live_gross = 0.0
    for p in snap.positions:
        entry = state.entry_time(p.symbol)
        remaining = ""
        unlocked = True
        if entry is not None:
            check = policy.check_exit(entry, now, AssetClass.CRYPTO)
            unlocked = check.allowed
            remaining = "" if check.allowed else f"{check.days_remaining:.1f}d"
            if check.allowed:
                scan.unlocked.append(p.symbol)

        px = live_price(p.symbol)
        if px is not None and p.avg_entry_price:
            value = abs(p.qty) * px
            unrealised = (px - p.avg_entry_price) * p.qty
            fresh = True
        else:
            value, unrealised, fresh = abs(p.market_value), p.unrealized_pl, False
        live_gross += value

        scan.positions.append(
            {
                "symbol": p.symbol,
                "value": value,
                "unrealised": unrealised,
                "pct": unrealised / value if value else 0.0,
                "locked_for": remaining,
                "unlocked": unlocked,
                "live_mark": fresh,
            }
        )

    if live_gross:
        scan.gross_exposure = live_gross

    # Live prices append a synthetic latest bar so the ranking reflects now,
    # not yesterday's close.
    bars = load_universe_bars(s, min_bars=400)
    if bars:
        ts, symbols, closes = build_panel(bars)
        try:
            live = latest_prices(symbols)
            if live:
                row = np.array([live.get(sym, closes[-1, j]) for j, sym in enumerate(symbols)])
                closes = np.vstack([closes, row])
        except Exception as e:  # noqa: BLE001
            scan.alerts.append(f"live prices unavailable ({type(e).__name__}); using last close")

        scores = momentum_scores(lookback=lookback, skip=skip)(closes)
        valid = np.isfinite(scores)
        order = np.argsort(np.where(valid, scores, -np.inf))[::-1]
        scan.ranking = [(symbols[j], float(scores[j])) for j in order if valid[j]]

        top = [sym for sym, _ in scan.ranking[:top_n]]
        if previous is not None:
            scan.ranking_changed = set(top) != set(previous)

    if risk.state.halted:
        scan.alerts.append(f"RISK HALTED — {risk.state.halt_reason}")
    dd = risk.state.drawdown
    if dd <= -0.10:
        scan.alerts.append(f"drawdown {dd:.1%}")

    return scan


def watch(
    s: Settings,
    *,
    interval: int = 60,
    duration: int = 3600,
    top_n: int = 4,
    lookback: int = 60,
    on_scan=None,
) -> list[Scan]:
    """Scan every `interval` seconds for `duration` seconds. Places no orders."""
    from finb.evaluation.ledger import SearchLedger

    ledger = SearchLedger(s.finb_data_dir / "artifacts" / "search_ledger.jsonl")
    ledger.record_observation(
        datetime.now(UTC).isoformat(timespec="seconds"),
        note=f"watch session, {duration // 60}min at {interval}s intervals",
    )

    scans: list[Scan] = []
    previous: list[str] | None = None
    deadline = time.monotonic() + duration

    while True:
        scan = scan_once(s, top_n=top_n, lookback=lookback, previous=previous)
        previous = [sym for sym, _ in scan.ranking[:top_n]]
        scans.append(scan)
        if on_scan:
            on_scan(scan, len(scans))

        if time.monotonic() >= deadline - interval:
            break
        time.sleep(interval)

    return scans
