"""Turning bars into a supervised learning problem.

"Predict the market" is not a learning problem. "Given these features, will
price touch +1.5 volatility before it touches -1 volatility, within 20 bars?" is
one — it has a definite answer, a definite horizon, and a label that corresponds
to something you could actually trade.

That reframing is what `labeling.triple_barrier` does, and it is the difference
between a model that optimises an abstraction and one that optimises a trade.
"""

from finb.features.labeling import (
    TripleBarrierResult,
    ewm_volatility,
    triple_barrier,
)

__all__ = ["TripleBarrierResult", "ewm_volatility", "triple_barrier"]
