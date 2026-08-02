"""Triple-barrier labelling (López de Prado, *Advances in Financial ML*, ch. 3).

For each candidate entry, place three barriers and label by whichever is touched
first:

- an upper barrier at ``+pt × volatility`` — the profit target,
- a lower barrier at ``-sl × volatility`` — the stop,
- a vertical barrier ``max_bars`` ahead — give up and go flat.

Why this instead of "the return over the next N bars":

1. **The barriers are scaled by volatility**, so a label means the same thing in
   a calm market and a violent one. Fixed thresholds silently relabel the same
   behaviour as the regime changes.
2. **It encodes the path, not just the endpoint.** A trade that dips through
   your stop and recovers is a loss in reality and a win under a
   fixed-horizon return. Only one of those is tradeable.
3. **It reports when each label resolves.** That end time is what makes purged
   cross-validation possible — see `finb.models.cv`. Without it, overlapping
   labels leak the future into training and every score is inflated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ewm_volatility(close: np.ndarray, span: int = 100) -> np.ndarray:
    """Exponentially-weighted volatility of log returns, per bar.

    Returns an array the same length as `close`. The first `span` values are
    unreliable by construction and are backfilled from the first stable
    estimate rather than left as NaN, so callers do not silently drop the start
    of every series.
    """
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        return np.zeros_like(close)
    if np.any(close <= 0):
        raise ValueError("prices must be strictly positive to take log returns")

    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    alpha = 2.0 / (span + 1.0)

    # EW mean of squared returns, computed iteratively for numerical stability.
    var = np.empty_like(logret)
    running = logret[1] ** 2 if logret.size > 1 else 0.0
    for i, r in enumerate(logret):
        running = alpha * r**2 + (1 - alpha) * running
        var[i] = running

    vol = np.sqrt(var)
    # Guard against a dead-flat opening producing zero-width barriers.
    first_positive = vol[vol > 0]
    if first_positive.size:
        vol[vol <= 0] = first_positive[0]
    return vol


@dataclass(frozen=True, slots=True)
class TripleBarrierResult:
    """One row per event, aligned with the `event_idx` that produced it."""

    event_idx: np.ndarray
    """Bar index where the position is opened."""

    label: np.ndarray
    """+1 upper touched first, -1 lower touched first, 0 vertical barrier."""

    touch_idx: np.ndarray
    """Bar index where the label resolved. **Feed this to purged CV.**"""

    ret: np.ndarray
    """Realised return from entry to touch, before costs."""

    barrier: np.ndarray
    """Which barrier was hit: 'pt', 'sl', or 'vertical'."""

    def __len__(self) -> int:
        return self.event_idx.size

    @property
    def class_balance(self) -> dict[int, int]:
        vals, counts = np.unique(self.label, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


def triple_barrier(
    close: np.ndarray,
    *,
    event_idx: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
    pt: float = 1.0,
    sl: float = 1.0,
    max_bars: int = 20,
    side: np.ndarray | None = None,
    vol_span: int = 100,
    label_vertical_by_sign: bool = False,
) -> TripleBarrierResult:
    """Label each event by the first barrier its path touches.

    Parameters
    ----------
    close
        Price series, one value per bar.
    event_idx
        Bars at which a position would be opened. Defaults to every bar that
        has room for a full horizon.
    volatility
        Per-bar volatility used to scale the barriers. Computed with
        `ewm_volatility` if omitted.
    pt, sl
        Profit-take and stop-loss barrier widths, in multiples of volatility.
        Set either to 0 to disable that barrier.
    max_bars
        The vertical barrier, in bars.
    side
        Optional +1/-1 per event for meta-labelling: when supplied, barriers
        follow the direction of the bet, and the label becomes 1 (the bet was
        right) or 0 (it was wrong) — i.e. "should I take this signal?" rather
        than "which way will price go?".
    label_vertical_by_sign
        If True, a vertical-barrier touch is labelled by the sign of its return
        instead of 0. Off by default: a flat outcome that is not worth trading
        should not be taught to the model as a directional call.
    """
    close = np.asarray(close, dtype=float)
    n = close.size
    if n == 0:
        raise ValueError("close is empty")
    if max_bars < 1:
        raise ValueError("max_bars must be >= 1")

    # Validate the cheap arguments before computing volatility, so a bad
    # event_idx reports itself rather than surfacing as a log-domain error.
    if event_idx is None:
        event_idx = np.arange(0, max(0, n - max_bars), dtype=int)
    event_idx = np.asarray(event_idx, dtype=int)
    if event_idx.size and (event_idx.min() < 0 or event_idx.max() >= n):
        raise ValueError("event_idx out of range")

    if volatility is None:
        volatility = ewm_volatility(close, span=vol_span)
    volatility = np.asarray(volatility, dtype=float)
    if volatility.size != n:
        raise ValueError("volatility must be the same length as close")

    meta = side is not None
    if meta:
        side = np.asarray(side, dtype=float)
        if side.size != event_idx.size:
            raise ValueError("side must be the same length as event_idx")

    labels = np.zeros(event_idx.size, dtype=int)
    touch = np.zeros(event_idx.size, dtype=int)
    rets = np.zeros(event_idx.size, dtype=float)
    which = np.empty(event_idx.size, dtype=object)

    for k, i in enumerate(event_idx):
        entry = close[i]
        vol = volatility[i]
        direction = float(side[k]) if meta else 1.0

        vertical = min(i + max_bars, n - 1)
        path = close[i + 1 : vertical + 1]

        up = entry * (1.0 + pt * vol) if pt > 0 else np.inf
        dn = entry * (1.0 - sl * vol) if sl > 0 else -np.inf

        # First index where each barrier is breached; n+1 sentinel if never.
        hit_up = np.argmax(path >= up) if (pt > 0 and np.any(path >= up)) else None
        hit_dn = np.argmax(path <= dn) if (sl > 0 and np.any(path <= dn)) else None

        if hit_up is None and hit_dn is None:
            t = vertical
            raw = (close[t] / entry) - 1.0
            signed = raw * direction
            if meta:
                lab = 1 if signed > 0 else 0
            elif label_vertical_by_sign:
                lab = int(np.sign(signed)) or 0
            else:
                lab = 0
            bar = "vertical"
        else:
            first_up = hit_up if hit_up is not None else len(path) + 1
            first_dn = hit_dn if hit_dn is not None else len(path) + 1
            if first_up <= first_dn:
                t = i + 1 + int(first_up)
                bar = "pt"
                directional = 1
            else:
                t = i + 1 + int(first_dn)
                bar = "sl"
                directional = -1
            raw = (close[t] / entry) - 1.0
            signed = raw * direction
            lab = (1 if signed > 0 else 0) if meta else directional

        labels[k] = lab
        touch[k] = t
        rets[k] = raw * direction
        which[k] = bar

    return TripleBarrierResult(
        event_idx=event_idx, label=labels, touch_idx=touch, ret=rets, barrier=which
    )
