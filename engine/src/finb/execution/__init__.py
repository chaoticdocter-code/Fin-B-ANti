"""Order execution.

Everything here is paper by default and stays that way unless three independent
conditions are all true:

1. ``FINB_ALLOW_LIVE`` equals the exact magic string in the environment,
2. ``ALPACA_PAPER=false`` is set explicitly, and
3. the caller passes ``allow_live=True`` when constructing the broker.

Any one of them missing and the live endpoint is unreachable — not "the order is
rejected", but the object cannot be built pointing at it. Three locks rather
than one because the failure mode is unrecoverable and the cost of an extra
check is nil.

Separately, no order reaches a venue without `finb.risk.RiskEngine` approving
its size. The strategy proposes; risk disposes.
"""

from finb.execution.base import AccountSnapshot, BrokerPosition, OrderRequest, OrderResult

__all__ = ["AccountSnapshot", "BrokerPosition", "OrderRequest", "OrderResult"]
