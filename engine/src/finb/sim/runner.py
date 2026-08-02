"""Drive a shadow book from a cross-sectional signal.

Deliberately the dumbest thing that could work. The point of a baseline is not
to make money — it is to prove the plumbing reproduces the known behaviour of a
documented factor. If cross-sectional momentum does not behave like
cross-sectional momentum here, nothing built on top of this can be believed, and
finding that out now costs a day instead of six months.

The execution convention is the load-bearing detail: a signal computed from
closes up to and including bar *t* is executed at the close of bar *t+1*. Acting
on the same bar you scored means trading on a price you could not have known,
which is the most common way a momentum backtest quietly becomes excellent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from finb.data.lake import Timeframe
from finb.sim.engine import ShadowBook

ScoreFn = Callable[[np.ndarray], np.ndarray]
"""(window of closes, shape (T, n_symbols)) -> score per symbol, NaN to exclude."""


def build_panel(bars: dict[str, pl.DataFrame]) -> tuple[list, list[str], np.ndarray]:
    """Align per-symbol bars into (timestamps, symbols, closes[T, N]).

    Inner join on timestamp: a symbol missing a bar removes that timestamp for
    everyone. Blunt, but it guarantees the cross-section is genuinely
    contemporaneous, and a forward-fill here would invent prices that never
    traded.
    """
    symbols = sorted(bars)
    frames = [
        bars[s].select("ts", pl.col("close").alias(s)).unique(subset=["ts"], keep="last")
        for s in symbols
    ]
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.join(f, on="ts", how="inner")
    panel = panel.sort("ts").drop_nulls()

    return panel["ts"].to_list(), symbols, panel.select(symbols).to_numpy()


def momentum_scores(lookback: int, skip: int = 0) -> ScoreFn:
    """Classic cross-sectional momentum: return over `lookback`, skipping the
    most recent `skip` bars.

    The skip is not decoration — short-horizon reversal is well documented, and
    the 12-1 convention in equities exists precisely to step over it.
    """

    def score(window: np.ndarray) -> np.ndarray:
        need = lookback + skip + 1
        if window.shape[0] < need:
            return np.full(window.shape[1], np.nan)
        end = window.shape[0] - 1 - skip
        start = end - lookback
        if start < 0:
            return np.full(window.shape[1], np.nan)
        return window[end] / window[start] - 1.0

    return score


@dataclass(frozen=True, slots=True)
class BacktestResult:
    book: ShadowBook
    timestamps: list
    symbols: list[str]
    rebalances: int
    benchmark_curve: np.ndarray
    """Equal-weight buy-and-hold over the same universe, costed once at entry.
    Beating a factor is meaningless if the factor loses to simply holding."""

    def summary(self) -> str:
        s = self.book.stats()
        bench = (
            self.benchmark_curve[-1] / self.benchmark_curve[0] - 1
            if self.benchmark_curve.size > 1
            else 0.0
        )
        return (
            f"final ${s['final_equity']:,.2f} ({s['total_return']:+.1%}), "
            f"buy-and-hold {bench:+.1%}, "
            f"maxDD {s['max_drawdown']:.1%}, {s['trades']} trades, "
            f"costs ${s['total_costs']:.2f} ({s['cost_pct_of_capital']:.1%} of capital)"
        )


def run_cross_sectional(
    bars: dict[str, pl.DataFrame],
    book: ShadowBook,
    *,
    score_fn: ScoreFn,
    top_n: int = 5,
    rebalance_every: int = 20,
    warmup: int = 60,
) -> BacktestResult:
    """Rank the cross-section every `rebalance_every` bars, hold the top `top_n`."""
    ts, symbols, closes = build_panel(bars)
    n_bars, n_sym = closes.shape
    if n_bars <= warmup + 2:
        raise ValueError(f"not enough aligned bars: {n_bars} rows after joining")

    top_n = min(top_n, n_sym)
    pending: dict[str, float] | None = None
    rebalances = 0

    for i in range(warmup, n_bars):
        prices = {s: float(closes[i, j]) for j, s in enumerate(symbols)}

        # Execute the previous bar's decision at this bar's close.
        if pending is not None:
            book.rebalance(ts[i], prices, pending)
            pending = None

        if (i - warmup) % rebalance_every == 0:
            scores = score_fn(closes[: i + 1])
            valid = np.isfinite(scores)
            if valid.sum() >= top_n:
                order = np.argsort(np.where(valid, scores, -np.inf))[::-1][:top_n]
                w = 1.0 / top_n
                pending = {symbols[j]: w for j in order}
                rebalances += 1

        book.mark(ts[i], prices)

    bench = closes[warmup:] / closes[warmup]
    return BacktestResult(
        book=book,
        timestamps=ts[warmup:],
        symbols=symbols,
        rebalances=rebalances,
        benchmark_curve=book.capital * bench.mean(axis=1),
    )


def load_panel_from_lake(lake, symbols, timeframe: Timeframe, asset) -> dict[str, pl.DataFrame]:
    out = {}
    for s in symbols:
        df = lake.read(s, timeframe, asset=asset)
        if df.height > 0:
            out[s] = df
    return out
