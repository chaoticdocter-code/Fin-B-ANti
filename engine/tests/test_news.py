"""News feed interpretation and daily aggregation.

No network calls. The interesting assertions are about not over-claiming from a
delay measurement, and about the daily count feature not peeking at its own day.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from finb.data.sources.news import FeedDelay, daily_news_features

# --------------------------------------------------------------------------- #
#  Delay interpretation
# --------------------------------------------------------------------------- #


def test_a_sparse_sample_is_reported_as_inconclusive():
    """A quiet overnight window must not be read as evidence of a delayed feed.

    This was a real bug: the verdict used the *median* article age, so twelve
    sparse hours produced 'DELAYED by ~267 min' when the newest article was 30
    minutes old and the feed latency was entirely unmeasured.
    """
    d = FeedDelay(articles=10, median_seconds=16_049, min_seconds=1_817, max_seconds=40_000)
    assert "INCONCLUSIVE" in d.verdict
    assert "30 min" in d.verdict          # quotes the newest, not the median
    assert "267" not in d.verdict


def test_no_articles_is_inconclusive_not_a_verdict():
    assert "inconclusive" in FeedDelay(0, 0, 0, 0).verdict.lower()


def test_a_dense_fresh_sample_reads_as_real_time():
    d = FeedDelay(articles=200, median_seconds=3_000, min_seconds=45, max_seconds=20_000)
    assert "real-time" in d.verdict


def test_a_dense_stale_sample_reads_as_delayed():
    d = FeedDelay(articles=200, median_seconds=9_000, min_seconds=3_600, max_seconds=40_000)
    assert "delayed feed" in d.verdict


def test_the_ambiguous_band_refuses_to_pick_a_side():
    """15 minutes is exactly where a delay and a lull are indistinguishable."""
    d = FeedDelay(articles=100, median_seconds=5_000, min_seconds=900, max_seconds=20_000)
    v = d.verdict
    assert "upper bound" in v
    assert "quiet period" in v


# --------------------------------------------------------------------------- #
#  Daily features
# --------------------------------------------------------------------------- #


def articles(symbol: str, per_day: list[int], start=datetime(2026, 6, 1, tzinfo=UTC)):
    rows, aid = [], 0
    for day, n in enumerate(per_day):
        for i in range(n):
            aid += 1
            rows.append(
                {
                    "id": aid,
                    "created_at": start + timedelta(days=day, hours=i % 12),
                    "symbol": symbol,
                    "headline": f"headline {aid}",
                    "source": f"src{i % 3}",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "symbol": pl.String,
            "headline": pl.String,
            "source": pl.String,
        },
    )


def test_counts_and_source_diversity_per_day():
    out = daily_news_features(articles("AAPL", [3, 1, 5]), window=2)
    assert out["article_count"].to_list() == [3, 1, 5]
    assert out["source_count"].to_list() == [3, 1, 3]


def test_the_zscore_does_not_include_its_own_day():
    """A day's z-score must be computable before that day's news is complete."""
    out = daily_news_features(articles("AAPL", [2] * 40 + [50]), window=10)
    spike = out.tail(1)
    # The spike day scores high precisely because the baseline excludes it.
    assert spike["article_count"][0] == 50
    assert spike["count_z"][0] > 5


def test_symbols_are_scored_independently():
    df = pl.concat([articles("AAPL", [1] * 40), articles("TSLA", [10] * 40)])
    out = daily_news_features(df, window=5)
    counts = {
        r["symbol"]: r["article_count"]
        for r in out.group_by("symbol").agg(pl.col("article_count").max()).to_dicts()
    }
    assert counts == {"AAPL": 1, "TSLA": 10}


def test_an_article_tagged_with_two_symbols_counts_once_for_each():
    df = pl.concat([articles("AAPL", [1]), articles("MSFT", [1])])
    out = daily_news_features(df)
    assert out["article_count"].to_list() == [1, 1]


def test_empty_input_gives_a_typed_empty_frame():
    out = daily_news_features(pl.DataFrame())
    assert out.is_empty()
    assert "count_z" in out.columns
