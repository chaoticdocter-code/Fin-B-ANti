"""Alpaca adapter. Paper by default, and hard to make otherwise.

Two things this class deliberately does not do:

- **It does not trust Alpaca's P&L as a score.** Alpaca's paper engine fills
  beyond the displayed NBBO size, hands out random 10% partial fills, and models
  no queue, no impact and no latency. A strategy scored on that number is being
  scored against a simulator it can farm. Use `finb.sim` for scoring; use this
  for reaching a venue.
- **It does not size orders.** Every order passes through `RiskEngine.check`
  first, and the quantity it returns is the quantity submitted.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from finb.config import LIVE_MAGIC, Settings
from finb.execution.base import (
    AccountSnapshot,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderType,
    TimeInForce,
)
from finb.log import get_logger
from finb.risk import RiskEngine
from finb.sim.constraints import Side

log = get_logger("alpaca-exec")

PAPER_URL = "https://paper-api.alpaca.markets"


class AlpacaBroker:
    """Talks to Alpaca. Refuses to point at live without three separate keys."""

    def __init__(
        self,
        settings: Settings,
        risk: RiskEngine | None = None,
        *,
        allow_live: bool = False,
        dry_run: bool = False,
        allocation: float | None = None,
        allow_short: bool = False,
    ) -> None:
        if not (settings.alpaca_api_key_id and settings.alpaca_api_secret_key):
            raise RuntimeError("Alpaca credentials are not configured — see .env")

        wants_live = not settings.alpaca_paper
        if wants_live:
            # All three, or the object does not exist.
            if not settings.live_enabled:
                raise RuntimeError(
                    "ALPACA_PAPER=false but FINB_ALLOW_LIVE is not set to "
                    f"{LIVE_MAGIC!r}. Refusing to construct a live broker."
                )
            if not allow_live:
                raise RuntimeError(
                    "ALPACA_PAPER=false and FINB_ALLOW_LIVE is set, but this call "
                    "did not pass allow_live=True. Refusing to construct a live "
                    "broker by default."
                )
            log.warning("LIVE TRADING ENABLED — real money is at risk")

        self.settings = settings
        self.risk = risk or RiskEngine()
        self.is_paper = not wants_live
        self.dry_run = dry_run
        self._client = None

        # Off by default. Crypto cannot be shorted here at all (0 of 73 assets
        # are shortable), and an accidental short is the one position type whose
        # loss has no floor.
        self.allow_short = allow_short

        # Alpaca hands a paper account $100,000 and 4x margin on top. Sizing
        # against the broker's equity would quietly discard the entire $500
        # constraint the project is built around — a 25% position cap on
        # $95,000 is $23,750. Our budget is our own number, and the broker's
        # balance is only ever an upper bound on it.
        self.allocation = float(allocation if allocation is not None else settings.finb_capital_usd)

    @staticmethod
    def canonical(symbol: str) -> str:
        """Normalise a symbol for comparison across Alpaca's two spellings.

        Orders are placed as ``BTC/USD`` but positions come back as ``BTCUSD``.
        A naive equality check therefore never matches a crypto position, so
        `held` reads as zero and the concentration cap silently stops applying —
        observed live: two BTC orders against a $125 cap produced a $164.67
        position, because the second order could not see the first.
        """
        return symbol.replace("/", "").upper()

    def _held_value(self, snap: AccountSnapshot, symbol: str) -> float:
        target = self.canonical(symbol)
        return sum(
            abs(p.market_value) for p in snap.positions if self.canonical(p.symbol) == target
        )

    def _held_qty(self, snap: AccountSnapshot, symbol: str) -> float:
        """Signed quantity: positive long, negative short.

        The risk engine needs the sign to tell a closing sell from an opening
        short — identical orders on the wire, opposite risk.
        """
        target = self.canonical(symbol)
        return sum(p.qty for p in snap.positions if self.canonical(p.symbol) == target)

    @staticmethod
    def _gross_short(snap: AccountSnapshot) -> float:
        return sum(abs(p.market_value) for p in snap.positions if p.qty < 0)

    def budget(self, account_equity: float) -> float:
        """The equity figure all sizing is done against.

        Never more than our allocation, and never more than the account
        actually holds. Growth of the allocation is a deliberate decision, not
        something that happens because the broker was generous.
        """
        return min(account_equity, self.allocation)

    # ------------------------------------------------------------------ #

    @property
    def client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self.settings.alpaca_api_key_id,
                secret_key=self.settings.alpaca_api_secret_key,
                paper=self.is_paper,
            )
        return self._client

    def account(self) -> AccountSnapshot:
        acct = self.client.get_account()
        positions = [
            BrokerPosition(
                symbol=str(p.symbol),
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl or 0.0),
            )
            for p in self.client.get_all_positions()
        ]
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            positions=positions,
            is_paper=self.is_paper,
            taken_at=datetime.now(UTC),
        )

    # ------------------------------------------------------------------ #

    def submit(self, order: OrderRequest) -> OrderResult:
        """Risk-check, then submit. Never the other way round."""
        snap = self.account()
        budget = self.budget(snap.equity)
        self.risk.update(snap.taken_at, budget)

        price = self._reference_price(order, snap)
        if price is None or price <= 0:
            return OrderResult(False, reason=f"no reference price for {order.symbol}")

        qty = order.qty if order.qty is not None else order.notional / price
        held = self._held_value(snap, order.symbol)

        decision = self.risk.check(
            side=order.side,
            symbol=order.symbol,
            qty=qty,
            price=price,
            equity=budget,
            current_position_value=held,
            gross_exposure=snap.gross_exposure,
            current_qty=self._held_qty(snap, order.symbol),
            gross_short=self._gross_short(snap),
            allow_short=self.allow_short,
        )
        if not decision:
            log.info(f"blocked {order.side} {order.symbol}: {decision.reason}")
            return OrderResult(False, symbol=order.symbol, reason=decision.reason)

        if decision.qty < qty:
            log.info(
                f"risk reduced {order.symbol} from {qty:.6f} to {decision.qty:.6f}"
            )

        # Minimum notional is enforced by RiskEngine.check, which rejects when the
        # remaining room falls below it rather than shrinking into dust. A second
        # check here would be unreachable.
        notional = decision.qty * price

        if self.dry_run:
            return OrderResult(
                True, order_id="dry-run", symbol=order.symbol,
                submitted_qty=decision.qty, submitted_notional=notional,
                reason="dry run — nothing was sent",
            )

        submitted = self._send(order, decision.qty)
        if submitted.accepted:
            self.risk.record_fill()
            submitted = replace(submitted, submitted_notional=notional)
        return submitted

    def _reference_price(self, order: OrderRequest, snap: AccountSnapshot) -> float | None:
        if order.limit_price:
            return order.limit_price
        target = self.canonical(order.symbol)
        for p in snap.positions:
            if self.canonical(p.symbol) == target and p.qty:
                return abs(p.market_value / p.qty)
        return self._latest_price(order.symbol)

    def _latest_price(self, symbol: str) -> float | None:
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoLatestTradeRequest

            if "/" in symbol:
                c = CryptoHistoricalDataClient()
                r = c.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=[symbol])
                )
                return float(r[symbol].price)

            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            c = StockHistoricalDataClient(
                api_key=self.settings.alpaca_api_key_id,
                secret_key=self.settings.alpaca_api_secret_key,
            )
            r = c.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=[symbol], feed="iex")
            )
            return float(r[symbol].price)
        except Exception as e:  # noqa: BLE001
            log.warning(f"could not price {symbol}: {type(e).__name__}: {e}")
            return None

    def _send(self, order: OrderRequest, qty: float) -> OrderResult:
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL
        tif = {
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.IOC: AlpacaTIF.IOC,
        }[order.time_in_force]

        common = {
            "symbol": order.symbol,
            "qty": round(qty, 9),
            "side": side,
            "time_in_force": tif,
            "client_order_id": order.client_id,
        }
        try:
            if order.order_type is OrderType.LIMIT:
                req = LimitOrderRequest(**common, limit_price=order.limit_price)
            else:
                req = MarketOrderRequest(**common)
            resp = self.client.submit_order(req)
            log.info(f"submitted {order.side} {qty:.6f} {order.symbol} -> {resp.id}")
            return OrderResult(
                True, order_id=str(resp.id), symbol=order.symbol, submitted_qty=qty
            )
        except Exception as e:  # noqa: BLE001
            log.error(f"order rejected for {order.symbol}: {type(e).__name__}: {e}")
            return OrderResult(False, symbol=order.symbol, reason=f"{type(e).__name__}: {e}")

    def cancel_all(self) -> int:
        if self.dry_run:
            return 0
        resp = self.client.cancel_orders()
        return len(resp or [])

    def close_all(self) -> int:
        """Flatten everything. The manual half of the kill switch."""
        if self.dry_run:
            return 0
        self.client.close_all_positions(cancel_orders=True)
        return len(self.account().positions)
