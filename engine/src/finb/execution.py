"""Execution logic using Alpaca."""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from finb.config import get_settings
from finb.models.base import Target
import logging

logger = logging.getLogger(__name__)

class ExecutionBroker:
    """Connects to Alpaca to execute vetted target weights."""
    def __init__(self):
        self.s = get_settings()
        if not self.s.alpaca_api_key_id or not self.s.alpaca_api_secret_key:
            raise ValueError("Alpaca keys are not configured.")
            
        self.client = TradingClient(
            self.s.alpaca_api_key_id, 
            self.s.alpaca_api_secret_key, 
            paper=self.s.alpaca_paper
        )

    def get_current_positions(self) -> dict[str, float]:
        """Fetch current positions and return a dict of symbol -> qty."""
        try:
            positions = self.client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}
            
    def get_account_value(self) -> float:
        """Fetch current equity value."""
        try:
            acc = self.client.get_account()
            return float(acc.equity)
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return 0.0

    def sync_targets(
        self, 
        targets: list[Target], 
        current_prices: dict[str, float], 
        trading_capital: float = None,
        take_profit_pct: float = 0.015,
        stop_loss_pct: float = 0.0075
    ):
        """
        Compare current positions to targets and issue market orders to align them.
        `current_prices` is needed to compute shares to buy from target weights.
        `trading_capital` overrides account equity to enforce strict max-exposure limits.
        """
        self.s.assert_paper_only()
        
        current_equity = self.get_account_value()
        base_capital = trading_capital if trading_capital is not None else current_equity
        current_positions = self.get_current_positions()
        
        # In HFT, we often just want to close positions if they aren't in the active target list
        # For simplicity, if a target weight is 0.0 and we hold it, we sell to close.
        for t in targets:
            target_notional = t.weight * base_capital
            price = current_prices.get(t.symbol)
            if not price:
                logger.warning(f"No current price for {t.symbol}, cannot execute.")
                continue
                
            target_qty = target_notional / price
            current_qty = current_positions.get(t.symbol, 0.0)
            
            delta_qty = target_qty - current_qty
            
            if abs(delta_qty) * price < 1.0: # Less than $1 diff, ignore
                continue
                
            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            
            req = MarketOrderRequest(
                symbol=t.symbol,
                qty=abs(delta_qty),
                side=side,
                time_in_force=TimeInForce.GTC
            )
            logger.info(f"Submitting {side} order for {abs(delta_qty):.4f} {t.symbol}")
                
            try:
                self.client.submit_order(req)
            except Exception as e:
                logger.error(f"Order failed for {t.symbol}: {e}")
