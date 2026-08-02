"""Partitioned Parquet bar storage with gap detection.

Layout::

    data/curated/bars/{asset_class}/{timeframe}/symbol={SYM}/{year}.parquet

Writes are idempotent: re-writing an overlapping range dedupes on timestamp and
keeps the newer row, so a backfill that overlaps existing data is safe to run
repeatedly. Every write goes to a temp file and is then atomically renamed,
because a half-written Parquet file that reads as "some data" is worse than one
that fails loudly.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path

import duckdb
import polars as pl

from finb.clock import ET, MARKET_OPEN, is_trading_day, session_close
from finb.sim.constraints import AssetClass

# Must start alphanumeric, and `..` is rejected outright below. The `symbol=`
# path prefix means traversal was never actually reachable, but a symbol that
# is not a symbol should fail loudly rather than quietly create a junk
# directory that later reads back as an empty series.
_SYMBOL_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,31}$")


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"

    @property
    def delta(self) -> timedelta:
        return {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }[self.value]


# The canonical bar schema. Sources adapt to this; nothing downstream adapts to
# a source.
BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trade_count": pl.Int64,
    "vwap": pl.Float64,
}


def _safe_symbol(symbol: str) -> str:
    """Symbols become path segments, so they are validated, not sanitised.

    Silently rewriting ``BTC/USD`` to ``BTC_USD`` would make two different
    symbols collide. Slashes are encoded reversibly instead.
    """
    if ".." in symbol or not _SYMBOL_OK.match(symbol):
        raise ValueError(f"unsupported symbol for a path segment: {symbol!r}")
    return symbol.replace("/", "-")


class BarLake:
    """Read/write OHLCV bars on the local Parquet lake."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.bars_root = self.root / "curated" / "bars"

    # ---------------------------------------------------------------- paths

    def _dir(self, asset: AssetClass, tf: Timeframe, symbol: str) -> Path:
        return self.bars_root / asset.value / tf.value / f"symbol={_safe_symbol(symbol)}"

    def _glob(self, asset: AssetClass, tf: Timeframe, symbol: str) -> str:
        return str(self._dir(asset, tf, symbol) / "*.parquet").replace("\\", "/")

    # ---------------------------------------------------------------- write

    def write(
        self,
        symbol: str,
        timeframe: Timeframe,
        df: pl.DataFrame,
        asset: AssetClass = AssetClass.CRYPTO,
    ) -> int:
        """Upsert bars. Returns the number of rows now stored for the symbol.

        `df` must carry at least ts/open/high/low/close/volume. Missing optional
        columns are filled with nulls so the on-disk schema stays stable.
        """
        if df.is_empty():
            return self.row_count(symbol, timeframe, asset)

        df = self._conform(df)
        out_dir = self._dir(asset, timeframe, symbol)
        out_dir.mkdir(parents=True, exist_ok=True)

        for (year,), part in df.group_by([pl.col("ts").dt.year().alias("y")]):
            path = out_dir / f"{year}.parquet"
            if path.exists():
                part = pl.concat([pl.read_parquet(path), part], how="vertical_relaxed")

            part = (
                part.sort("ts")
                # Keep the last row for a duplicated timestamp: a later write is
                # assumed to be a correction of an earlier one.
                .unique(subset=["ts"], keep="last", maintain_order=True)
            )

            tmp = path.with_suffix(".parquet.tmp")
            part.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)

        return self.row_count(symbol, timeframe, asset)

    def _conform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = {"ts", "open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            raise ValueError(f"bars are missing required columns: {sorted(missing)}")

        for col, dtype in BAR_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

        df = df.select(list(BAR_SCHEMA))

        ts = pl.col("ts")
        if df.schema["ts"].time_zone is None:
            # A naive timestamp is ambiguous and the single easiest way to
            # introduce a silent off-by-hours bug. Treat it as UTC and say so.
            ts = ts.dt.replace_time_zone("UTC")
        else:
            ts = ts.dt.convert_time_zone("UTC")

        return df.with_columns(ts.cast(BAR_SCHEMA["ts"]).alias("ts")).sort("ts")

    # ----------------------------------------------------------------- read

    def read(
        self,
        symbols: str | Iterable[str],
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        asset: AssetClass = AssetClass.CRYPTO,
    ) -> pl.DataFrame:
        """Read bars for one or many symbols into a single frame.

        Returns an empty, correctly-typed frame when nothing is stored, so
        callers never need to special-case the cold-start path.
        """
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        frames = []

        for sym in syms:
            if not any(self._dir(asset, timeframe, sym).glob("*.parquet")):
                continue
            q = f"SELECT * FROM read_parquet('{self._glob(asset, timeframe, sym)}')"
            clauses = []
            if start is not None:
                clauses.append(f"ts >= TIMESTAMPTZ '{start.isoformat()}'")
            if end is not None:
                clauses.append(f"ts <= TIMESTAMPTZ '{end.isoformat()}'")
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY ts"

            part = duckdb.sql(q).pl()
            if not part.is_empty():
                frames.append(part.with_columns(pl.lit(sym).alias("symbol")))

        if not frames:
            empty = pl.DataFrame(schema={**BAR_SCHEMA, "symbol": pl.String})
            return empty

        return pl.concat(frames, how="vertical_relaxed").sort(["symbol", "ts"])

    def row_count(
        self, symbol: str, timeframe: Timeframe, asset: AssetClass = AssetClass.CRYPTO
    ) -> int:
        if not any(self._dir(asset, timeframe, symbol).glob("*.parquet")):
            return 0
        q = f"SELECT count(*) FROM read_parquet('{self._glob(asset, timeframe, symbol)}')"
        return int(duckdb.sql(q).fetchone()[0])

    def coverage(
        self, symbol: str, timeframe: Timeframe, asset: AssetClass = AssetClass.CRYPTO
    ) -> tuple[datetime, datetime] | None:
        """(first, last) stored timestamp, or None if nothing is stored."""
        if not any(self._dir(asset, timeframe, symbol).glob("*.parquet")):
            return None
        q = (
            f"SELECT min(ts), max(ts) FROM "
            f"read_parquet('{self._glob(asset, timeframe, symbol)}')"
        )
        lo, hi = duckdb.sql(q).fetchone()
        return (lo, hi) if lo is not None else None

    def symbols(
        self, timeframe: Timeframe, asset: AssetClass = AssetClass.CRYPTO
    ) -> list[str]:
        base = self.bars_root / asset.value / timeframe.value
        if not base.is_dir():
            return []
        return sorted(
            d.name.removeprefix("symbol=").replace("-", "/")
            if asset is AssetClass.CRYPTO
            else d.name.removeprefix("symbol=")
            for d in base.iterdir()
            if d.is_dir() and d.name.startswith("symbol=")
        )

    # ----------------------------------------------------------------- gaps

    def missing(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        asset: AssetClass = AssetClass.CRYPTO,
    ) -> list[datetime]:
        """Timestamps that should exist in [start, end] but do not.

        This is the method that keeps the model honest. Feed its output to a
        backfill, or exclude the affected windows from training — but never
        ignore it.
        """
        expected = set(expected_timestamps(timeframe, start, end, asset))
        if not expected:
            return []
        have = self.read(symbol, timeframe, start, end, asset)
        actual = set(have["ts"].to_list()) if not have.is_empty() else set()
        return sorted(expected - actual)

    def completeness(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        asset: AssetClass = AssetClass.CRYPTO,
    ) -> float:
        """Fraction of expected bars present, in [0, 1]."""
        expected = expected_timestamps(timeframe, start, end, asset)
        if not expected:
            return 1.0
        return 1.0 - len(self.missing(symbol, timeframe, start, end, asset)) / len(expected)


def expected_timestamps(
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    asset: AssetClass = AssetClass.CRYPTO,
) -> list[datetime]:
    """The timestamp grid a complete series would have.

    Crypto is a plain regular grid — the market never closes. Equities are only
    expected during regular trading hours on actual trading days, which is why
    this consults the NYSE calendar rather than assuming Monday-to-Friday.
    """
    from datetime import UTC

    start = start.astimezone(UTC)
    end = end.astimezone(UTC)

    if asset is AssetClass.CRYPTO:
        step = timeframe.delta
        # Align to the natural grid so a start of 10:03 does not produce 10:03,
        # 10:04, ... offset from every other series.
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        n0 = -(-(start - epoch) // step)  # ceiling division
        out, t = [], epoch + n0 * step
        while t <= end:
            out.append(t)
            t += step
        return out

    if timeframe is Timeframe.D1:
        return [
            datetime.combine(d, time(0, 0), tzinfo=UTC)
            for d in _trading_days_between(start.date(), end.date())
        ]

    out = []
    step = timeframe.delta
    for d in _trading_days_between(start.date(), end.date()):
        open_et = datetime.combine(d, MARKET_OPEN, tzinfo=ET)
        close_et = datetime.combine(d, session_close(d), tzinfo=ET)
        t = open_et
        while t < close_et:
            u = t.astimezone(UTC)
            if start <= u <= end:
                out.append(u)
            t += step
    return out


def _trading_days_between(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out
