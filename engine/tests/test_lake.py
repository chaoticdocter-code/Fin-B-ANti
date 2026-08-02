"""Lake round-trips, idempotent writes, and — the important one — gap detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from finb.data.lake import BarLake, Timeframe, expected_timestamps
from finb.sim.constraints import AssetClass


def bars(start: datetime, n: int, step: timedelta, price: float = 100.0) -> pl.DataFrame:
    ts = [start + i * step for i in range(n)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": [price + i for i in range(n)],
            "high": [price + i + 1 for i in range(n)],
            "low": [price + i - 1 for i in range(n)],
            "close": [price + i + 0.5 for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


@pytest.fixture
def lake(tmp_path):
    return BarLake(tmp_path)


# --------------------------------------------------------------------------- #


def test_write_then_read_round_trips(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 24, timedelta(hours=1)))

    got = lake.read("BTC/USD", Timeframe.H1)
    assert len(got) == 24
    assert got["ts"][0] == t0
    assert got["symbol"][0] == "BTC/USD"
    # Optional columns are materialised so the schema is stable across sources.
    assert "vwap" in got.columns and got["vwap"].null_count() == 24


def test_reading_an_empty_lake_gives_a_typed_empty_frame(lake):
    got = lake.read("NOPE", Timeframe.H1)
    assert got.is_empty()
    assert "close" in got.columns  # callers can filter without special-casing


def test_overlapping_writes_are_idempotent(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 24, timedelta(hours=1)))
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 24, timedelta(hours=1)))
    assert lake.row_count("BTC/USD", Timeframe.H1) == 24

    # Overlapping-but-extending write.
    lake.write("BTC/USD", Timeframe.H1, bars(t0 + timedelta(hours=12), 24, timedelta(hours=1)))
    assert lake.row_count("BTC/USD", Timeframe.H1) == 36


def test_a_later_write_corrects_an_earlier_bar(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 3, timedelta(hours=1)))

    corrected = bars(t0, 3, timedelta(hours=1)).with_columns(pl.lit(999.0).alias("close"))
    lake.write("BTC/USD", Timeframe.H1, corrected)

    got = lake.read("BTC/USD", Timeframe.H1)
    assert len(got) == 3
    assert got["close"].to_list() == [999.0, 999.0, 999.0]


def test_writes_spanning_a_year_boundary_split_into_partitions(lake):
    t0 = datetime(2025, 12, 31, 20, tzinfo=UTC)
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 10, timedelta(hours=1)))

    files = sorted(p.name for p in (lake._dir(AssetClass.CRYPTO, Timeframe.H1, "BTC/USD")).glob("*.parquet"))
    assert files == ["2025.parquet", "2026.parquet"]
    assert lake.row_count("BTC/USD", Timeframe.H1) == 10


def test_naive_timestamps_are_treated_as_utc_not_local(lake):
    naive = bars(datetime(2026, 3, 2), 3, timedelta(hours=1)).with_columns(
        pl.col("ts").dt.replace_time_zone(None)
    )
    lake.write("X", Timeframe.H1, naive)
    assert lake.read("X", Timeframe.H1)["ts"][0] == datetime(2026, 3, 2, tzinfo=UTC)


def test_symbols_with_slashes_survive_the_round_trip(lake):
    lake.write("BTC/USD", Timeframe.H1, bars(datetime(2026, 3, 2, tzinfo=UTC), 2, timedelta(hours=1)))
    lake.write("ETH/USD", Timeframe.H1, bars(datetime(2026, 3, 2, tzinfo=UTC), 2, timedelta(hours=1)))
    assert lake.symbols(Timeframe.H1) == ["BTC/USD", "ETH/USD"]


def test_rejects_a_symbol_that_would_escape_the_lake_directory(lake):
    with pytest.raises(ValueError, match="unsupported symbol"):
        lake.write("../../etc/passwd", Timeframe.H1, bars(datetime(2026, 3, 2, tzinfo=UTC), 1, timedelta(hours=1)))


def test_missing_required_columns_is_rejected(lake):
    df = pl.DataFrame({"ts": [datetime(2026, 3, 2, tzinfo=UTC)], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        lake.write("X", Timeframe.H1, df)


# --------------------------------------------------------------------------- #
#  Gaps — a hole in the data must never look like a quiet market
# --------------------------------------------------------------------------- #


def test_detects_a_hole_in_the_middle_of_a_crypto_series(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    full = bars(t0, 24, timedelta(hours=1))
    with_hole = full.filter(~pl.col("ts").is_in(full["ts"][5:9].to_list()))
    lake.write("BTC/USD", Timeframe.H1, with_hole)

    missing = lake.missing("BTC/USD", Timeframe.H1, t0, t0 + timedelta(hours=23))
    assert len(missing) == 4
    assert missing[0] == t0 + timedelta(hours=5)
    assert lake.completeness("BTC/USD", Timeframe.H1, t0, t0 + timedelta(hours=23)) == pytest.approx(20 / 24)


def test_a_complete_series_reports_no_gaps(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 24, timedelta(hours=1)))
    assert lake.missing("BTC/USD", Timeframe.H1, t0, t0 + timedelta(hours=23)) == []
    assert lake.completeness("BTC/USD", Timeframe.H1, t0, t0 + timedelta(hours=23)) == 1.0


def test_crypto_expects_bars_around_the_clock_including_weekends():
    # Sat 2026-03-07 through Sun 2026-03-08.
    ts = expected_timestamps(
        Timeframe.H1,
        datetime(2026, 3, 7, tzinfo=UTC),
        datetime(2026, 3, 8, 23, tzinfo=UTC),
        AssetClass.CRYPTO,
    )
    assert len(ts) == 48


def test_equities_expect_bars_only_during_sessions():
    """A weekend and a holiday must not be reported as missing data."""
    # Fri 2026-01-16 through Tue 2026-01-20; the 17th-18th are a weekend and
    # the 19th is MLK Day.
    ts = expected_timestamps(
        Timeframe.H1,
        datetime(2026, 1, 16, tzinfo=UTC),
        datetime(2026, 1, 21, tzinfo=UTC),
        AssetClass.EQUITY,
    )
    days = {t.astimezone(UTC).date() for t in ts}
    assert days == {__import__("datetime").date(2026, 1, 16), __import__("datetime").date(2026, 1, 20)}
    # 09:30-16:00 stepped hourly gives 7 bars per session.
    assert len(ts) == 14


def test_equity_lake_does_not_flag_closed_days_as_gaps(lake):
    """The whole point: a complete equity series over a holiday weekend is 100%."""
    start = datetime(2026, 1, 16, tzinfo=UTC)
    end = datetime(2026, 1, 21, tzinfo=UTC)
    grid = expected_timestamps(Timeframe.H1, start, end, AssetClass.EQUITY)

    df = pl.DataFrame(
        {
            "ts": grid,
            "open": [1.0] * len(grid),
            "high": [1.0] * len(grid),
            "low": [1.0] * len(grid),
            "close": [1.0] * len(grid),
            "volume": [1.0] * len(grid),
        }
    )
    lake.write("AAPL", Timeframe.H1, df, asset=AssetClass.EQUITY)

    assert lake.missing("AAPL", Timeframe.H1, start, end, AssetClass.EQUITY) == []
    assert lake.completeness("AAPL", Timeframe.H1, start, end, AssetClass.EQUITY) == 1.0


def test_coverage_reports_the_stored_span(lake):
    t0 = datetime(2026, 3, 2, tzinfo=UTC)
    assert lake.coverage("BTC/USD", Timeframe.H1) is None
    lake.write("BTC/USD", Timeframe.H1, bars(t0, 10, timedelta(hours=1)))
    lo, hi = lake.coverage("BTC/USD", Timeframe.H1)
    assert lo == t0
    assert hi == t0 + timedelta(hours=9)
