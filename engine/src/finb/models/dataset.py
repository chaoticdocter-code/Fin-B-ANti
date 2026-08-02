"""Turn a panel of bars into a supervised learning problem.

Three details here are the difference between a real experiment and a
self-flattering one:

1. **Features are computed per symbol, then stacked and sorted by time.** Any
   cross-symbol operation must happen after alignment, or a symbol's future
   leaks into another symbol's past through a shared statistic.

2. **Labels come from `triple_barrier`, so each one reports when it resolved.**
   That resolution time is carried through as `t1_idx` — the row index in the
   globally time-sorted dataset where the label became known. `PurgedKFold`
   needs exactly this, and without it the cross-validation is decorative.

3. **Rows with any null feature are dropped, never imputed.** Imputing a warmup
   value is a quiet way to invent data, and at this sample size dropping a few
   hundred rows costs less than trusting a fabricated one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from finb.features import indicators as ind
from finb.features.labeling import triple_barrier


@dataclass(frozen=True, slots=True)
class PanelDataset:
    X: np.ndarray
    y: np.ndarray
    """Binary: did price touch the upper barrier before the lower one?"""

    ret: np.ndarray
    """Realised return to the barrier touch, before costs."""

    ts: np.ndarray
    symbol: np.ndarray
    t1_idx: np.ndarray
    """Row index (in this time-sorted dataset) where each label resolved."""

    feature_names: list[str]

    def __len__(self) -> int:
        return self.y.size

    def summary(self) -> str:
        pos = float(self.y.mean()) if self.y.size else 0.0
        span = self.t1_idx - np.arange(self.y.size)
        return (
            f"{len(self):,} samples, {len(set(self.symbol.tolist()))} symbols, "
            f"{len(self.feature_names)} features, {pos:.1%} positive, "
            f"mean label span {span.mean():.0f} rows"
        )


def build_panel_dataset(
    bars: dict[str, pl.DataFrame],
    *,
    exprs: list[pl.Expr] | None = None,
    horizon: int = 38,
    pt: float = 1.5,
    sl: float = 1.0,
    vol_span: int = 60,
    warmup: int = 120,
) -> PanelDataset:
    """Build a stacked, time-sorted, purge-ready dataset.

    `horizon` should match the holding period the cost model forces — labelling
    a 38-day hold with a 5-day barrier trains the model to answer a question the
    strategy will never ask.
    """
    exprs = exprs if exprs is not None else ind.DEFAULT
    names = ind.feature_names(exprs)

    Xs, ys, rets, tss, syms, t1_tss = [], [], [], [], [], []

    for symbol in sorted(bars):
        df = bars[symbol].sort("ts")
        if df.height < warmup + horizon + 10:
            continue

        feats = ind.compute(df, exprs)
        close = df["close"].to_numpy().astype(float)
        ts = df["ts"].to_numpy()

        events = np.arange(warmup, len(close) - horizon)
        if events.size == 0:
            continue

        tb = triple_barrier(
            close,
            event_idx=events,
            pt=pt,
            sl=sl,
            max_bars=horizon,
            vol_span=vol_span,
        )

        fm = feats.to_numpy().astype(float)
        rows = fm[tb.event_idx]
        keep = np.isfinite(rows).all(axis=1)

        Xs.append(rows[keep])
        ys.append((tb.label[keep] == 1).astype(int))
        rets.append(tb.ret[keep])
        tss.append(ts[tb.event_idx][keep])
        t1_tss.append(ts[tb.touch_idx][keep])
        syms.append(np.full(keep.sum(), symbol, dtype=object))

    if not Xs:
        raise ValueError("no symbol had enough history to build samples")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    ret = np.concatenate(rets)
    ts = np.concatenate(tss)
    t1_ts = np.concatenate(t1_tss)
    symbol = np.concatenate(syms)

    # Global time sort. Purged CV assumes chronological order across the whole
    # panel, not within each symbol.
    order = np.argsort(ts, kind="stable")
    X, y, ret, ts, t1_ts, symbol = (
        X[order], y[order], ret[order], ts[order], t1_ts[order], symbol[order]
    )

    # Map each label's resolution *time* to a row index in the sorted panel.
    t1_idx = np.searchsorted(ts, t1_ts, side="right") - 1
    t1_idx = np.clip(t1_idx, np.arange(ts.size), ts.size - 1)

    return PanelDataset(
        X=X, y=y, ret=ret, ts=ts, symbol=symbol, t1_idx=t1_idx, feature_names=names
    )
