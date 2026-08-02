"""Universe archive.

The load-bearing test is `test_asof_refuses_to_use_a_later_snapshot`: falling
forward to a newer asset list is exactly the survivorship bias this module
exists to prevent, so it must fail loudly rather than quietly return something
plausible.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from finb.data.universe import UNIVERSE_SCHEMA, UniverseArchive


def snapshot(d: date, symbols: list[str], asset_class: str = "crypto") -> pl.DataFrame:
    n = len(symbols)
    return pl.DataFrame(
        {
            "snapshot_date": [d] * n,
            "symbol": symbols,
            "name": [f"{s} coin" for s in symbols],
            "asset_class": [asset_class] * n,
            "exchange": ["CRYPTO"] * n,
            "status": ["active"] * n,
            "tradable": [True] * n,
            "marginable": [False] * n,
            "shortable": [False] * n,
            "easy_to_borrow": [False] * n,
            "fractionable": [True] * n,
            "min_order_size": [0.0001] * n,
            "min_trade_increment": [0.0001] * n,
            "price_increment": [0.01] * n,
        },
        schema=UNIVERSE_SCHEMA,
    )


@pytest.fixture
def archive(tmp_path):
    return UniverseArchive(tmp_path)


def test_write_and_load_round_trip(archive):
    archive.write(snapshot(date(2026, 8, 1), ["BTC/USD", "ETH/USD"]))
    got = archive.load(date(2026, 8, 1))
    assert sorted(got["symbol"].to_list()) == ["BTC/USD", "ETH/USD"]


def test_asof_returns_the_most_recent_prior_snapshot(archive):
    archive.write(snapshot(date(2026, 6, 1), ["BTC/USD", "LUNA/USD"]))
    archive.write(snapshot(date(2026, 7, 1), ["BTC/USD"]))

    # Mid-June must see the June list, which still contains LUNA.
    mid = archive.asof(date(2026, 6, 15))
    assert set(mid["symbol"].to_list()) == {"BTC/USD", "LUNA/USD"}


def test_asof_refuses_to_use_a_later_snapshot(archive):
    """Asking about a date before any snapshot exists must raise.

    Returning the nearest *future* list would silently answer 'what is tradeable
    now' to the question 'what was tradeable then' — the exact fabrication this
    module prevents.
    """
    archive.write(snapshot(date(2026, 8, 1), ["BTC/USD"]))
    with pytest.raises(FileNotFoundError, match="survivorship bias"):
        archive.asof(date(2026, 1, 1))


def test_delistings_are_recoverable_once_there_are_two_snapshots(archive):
    archive.write(snapshot(date(2026, 6, 1), ["BTC/USD", "ETH/USD", "DEAD/USD"]))
    archive.write(snapshot(date(2026, 7, 1), ["BTC/USD", "ETH/USD"]))

    assert archive.delistings(since=date(2026, 6, 1)) == ["DEAD/USD"]


def test_a_single_snapshot_cannot_reveal_delistings(archive):
    archive.write(snapshot(date(2026, 8, 1), ["BTC/USD"]))
    assert archive.delistings(since=date(2026, 8, 1)) == []


def test_tradable_symbols_filters_by_class_and_fractionability(archive):
    df = pl.concat(
        [
            snapshot(date(2026, 8, 1), ["BTC/USD", "ETH/USD"], "crypto"),
            snapshot(date(2026, 8, 1), ["AAPL"], "us_equity"),
        ]
    ).with_columns(
        pl.when(pl.col("symbol") == "ETH/USD")
        .then(False)
        .otherwise(pl.col("fractionable"))
        .alias("fractionable")
    )
    archive.write(df)

    assert archive.tradable_symbols(date(2026, 8, 1), "crypto") == ["BTC/USD", "ETH/USD"]
    assert archive.tradable_symbols(date(2026, 8, 1), "crypto", fractionable_only=True) == [
        "BTC/USD"
    ]
    assert archive.tradable_symbols(date(2026, 8, 1), "us_equity") == ["AAPL"]


def test_untradable_assets_are_excluded(archive):
    df = snapshot(date(2026, 8, 1), ["BTC/USD", "HALTED/USD"]).with_columns(
        pl.when(pl.col("symbol") == "HALTED/USD")
        .then(False)
        .otherwise(pl.col("tradable"))
        .alias("tradable")
    )
    archive.write(df)
    assert archive.tradable_symbols(date(2026, 8, 1)) == ["BTC/USD"]


def test_coverage_reports_staleness(archive):
    assert archive.coverage()["snapshots"] == 0
    archive.write(snapshot(date(2026, 8, 1), ["BTC/USD"]))
    cov = archive.coverage()
    assert cov["snapshots"] == 1
    assert cov["first"] == cov["last"] == date(2026, 8, 1)
    assert cov["gap_days"] >= 0


def test_refuses_to_write_an_empty_snapshot(archive):
    with pytest.raises(ValueError, match="empty universe snapshot"):
        archive.write(pl.DataFrame(schema=UNIVERSE_SCHEMA))
