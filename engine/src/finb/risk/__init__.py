"""Risk limits and kill switches.

This layer exists for failures that have nothing to do with strategy quality: a
loop that submits the same order a thousand times, a sign error that turns a
stop into an add, a data gap that makes every position look cheap. A good
strategy cannot protect you from those. Only a hard limit can.

Two principles:

- **Limits are enforced below the strategy, not inside it.** A strategy that
  polices its own risk is one bug away from not doing so.
- **The kill switch is one-way.** Once tripped, only position-reducing orders
  pass until a human resets it. Automatic recovery from an unexplained drawdown
  is how a small loss becomes a large one.
"""

from finb.risk.limits import RiskDecision, RiskEngine, RiskLimits, RiskState

__all__ = ["RiskDecision", "RiskEngine", "RiskLimits", "RiskState"]
