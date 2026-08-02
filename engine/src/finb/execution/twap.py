"""Time-Weighted Average Price (TWAP) Order Slicing Executor.

Slices large rebalance orders into smaller sub-orders over a specified
duration to minimize market impact and execution slippage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from finb.execution.alpaca_paper import AlpacaBroker
from finb.execution.base import OrderRequest, OrderResult
from finb.log import get_logger
from finb.sim.constraints import AssetClass, Side

log = get_logger("twap")


@dataclass(frozen=True, slots=True)
class TWAPConfig:
    slices: int = 3
    interval_seconds: float = 2.0  # In paper mode, short pause between slices
    min_slice_notional: float = 10.0


class TWAPExecutor:
    """Slices large target notional orders into TWAP sub-orders."""

    def __init__(self, broker: AlpacaBroker, config: TWAPConfig | None = None) -> None:
        self.broker = broker
        self.config = config or TWAPConfig()

    def execute_twap(
        self,
        symbol: str,
        side: Side,
        total_notional: float,
        asset_class: AssetClass = AssetClass.CRYPTO,
    ) -> list[OrderResult]:
        """Execute total_notional in sliced sub-orders."""
        if total_notional < self.config.min_slice_notional * self.config.slices:
            # Single order if total notional is too small to slice
            res = self.broker.submit(
                OrderRequest(symbol, side, notional=round(total_notional, 2), asset_class=asset_class)
            )
            return [res] if res else []

        slice_notional = round(total_notional / self.config.slices, 2)
        results: list[OrderResult] = []

        log.info(
            f"TWAP Slicing {symbol} {side.value.upper()}: ${total_notional:.2f} into "
            f"{self.config.slices} x ${slice_notional:.2f}"
        )

        for i in range(self.config.slices):
            req = OrderRequest(symbol, side, notional=slice_notional, asset_class=asset_class)
            res = self.broker.submit(req)
            if res:
                results.append(res)
            if i < self.config.slices - 1 and self.config.interval_seconds > 0:
                time.sleep(self.config.interval_seconds)

        return results
