"""Deciding whether the bot has actually learned anything.

A single continuously-evolving bot is a search process. Each retrain, each
parameter change, each new feature is a draw from the space of possible
strategies — and the one you keep is, by definition, the one that looked best.
That is the same statistical situation as running a thousand bots in parallel,
spread out over time instead of across machines, which makes it much easier to
forget you are doing it.

`ledger.SearchLedger` remembers. `gate.PromotionGate` charges for it.
"""

from finb.evaluation.gate import (
    GateVerdict,
    PromotionGate,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from finb.evaluation.ledger import SearchLedger, Trial

__all__ = [
    "GateVerdict",
    "PromotionGate",
    "SearchLedger",
    "Trial",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]
