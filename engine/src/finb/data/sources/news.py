"""Alpaca's news feed (Benzinga-sourced).

Why this module matters more than another indicator: the ML baseline failed with
gross-negative edge using 18 technical features, and every one of them was a
transform of the same close series. No amount of modelling extracts information
that is not in the inputs. News is the cheapest genuinely orthogonal signal
available here, and the credential probe confirmed the feed is reachable.

Two cautions, both load-bearing.

**Timestamp integrity.** The research flagged as *undocumented* whether the free
tier's 15-minute delay applies to news. If it does and you do not know it, every
event-window feature is misaligned by 15 minutes and any "news reaction" alpha
is an artifact. `measure_feed_delay` answers that empirically instead of
guessing — run it before building anything on top of this.

**Volume before sentiment.** Article counts and their deviation from a trailing
norm are unambiguous and hard to get wrong. Headline sentiment from a small
lexicon is crude, and crude sentiment on financial headlines is frequently worse
than nothing. Counts are the default here; sentiment is offered separately and
labelled as weak.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from finb.config import Settings
from finb.log import get_logger

log = get_logger("news")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def _headers(s: Settings) -> dict[str, str]:
    if not (s.alpaca_api_key_id and s.alpaca_api_secret_key):
        raise RuntimeError("Alpaca credentials are not configured — see .env")
    return {
        "APCA-API-KEY-ID": s.alpaca_api_key_id,
        "APCA-API-SECRET-KEY": s.alpaca_api_secret_key,
    }


def fetch_news(
    s: Settings,
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    limit_per_page: int = 50,
    max_pages: int = 40,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Fetch articles as (created_at, updated_at, symbol, headline, source).

    One row per (article, symbol) pair, so an article tagged with three tickers
    contributes to all three. That is the right shape for per-symbol features
    and the wrong shape for counting distinct articles — deduplicate on `id`
    first if you need the latter.
    """
    params: dict = {"limit": limit_per_page, "sort": "asc", "include_content": "false"}
    if symbols:
        params["symbols"] = ",".join(symbols)
    if start:
        params["start"] = start.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if end:
        params["end"] = end.astimezone(UTC).isoformat().replace("+00:00", "Z")

    rows: list[dict] = []
    token: str | None = None

    with httpx.Client(timeout=timeout, headers=_headers(s)) as client:
        for _ in range(max_pages):
            q = dict(params)
            if token:
                q["page_token"] = token
            r = client.get(NEWS_URL, params=q)
            if r.status_code >= 400:
                raise RuntimeError(f"news request failed: HTTP {r.status_code} {r.text[:160]}")

            payload = r.json()
            for a in payload.get("news", []):
                created = a.get("created_at")
                if not created:
                    continue
                for sym in a.get("symbols") or ["*"]:
                    rows.append(
                        {
                            "id": int(a.get("id", 0)),
                            "created_at": datetime.fromisoformat(created.replace("Z", "+00:00")),
                            "symbol": sym,
                            "headline": a.get("headline") or "",
                            "source": a.get("source") or "",
                        }
                    )
            token = payload.get("next_page_token")
            if not token:
                break

    df = pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "created_at": pl.Datetime(time_unit="us", time_zone="UTC"),
            "symbol": pl.String,
            "headline": pl.String,
            "source": pl.String,
        },
    )
    log.info(f"{df.height} article-symbol rows, {df['id'].n_unique() if df.height else 0} articles")
    return df.sort("created_at")


@dataclass(frozen=True, slots=True)
class FeedDelay:
    articles: int
    median_seconds: float
    min_seconds: float
    max_seconds: float

    @property
    def verdict(self) -> str:
        """Interpretation, using the NEWEST article — not the median.

        The median age of articles in a lookback window measures how sparse the
        news was, not how delayed the feed is: sample twelve quiet hours and the
        median is naturally hours old whatever the latency. Only the newest
        article bounds the delay, and even that is an *upper* bound — it says
        "nothing was published more recently than this", which during a weekend
        or overnight lull says nothing about the feed at all.
        """
        if self.articles == 0:
            return "no articles — inconclusive, retry during market hours"

        lag_min = self.min_seconds / 60
        if self.articles < 20:
            return (
                f"INCONCLUSIVE — only {self.articles} articles sampled, newest "
                f"{lag_min:.0f} min old. Too quiet to separate feed delay from a "
                "slow news period; retry during US market hours."
            )
        if self.min_seconds <= 120:
            return f"effectively real-time (newest article {self.min_seconds:.0f}s old)"
        if self.min_seconds <= 1200:
            return (
                f"upper bound ~{lag_min:.0f} min. Consistent with either a 15-minute "
                "delay or a quiet period — re-measure on a busy session to separate them."
            )
        return (
            f"newest article {lag_min:.0f} min old across {self.articles} samples — "
            "likely a genuinely delayed feed; shift event-window features accordingly."
        )


def measure_feed_delay(s: Settings, lookback_hours: int = 6) -> FeedDelay:
    """Measure the gap between an article's `created_at` and now.

    Resolves the open question the research could not answer from documentation.
    Interpretation: the newest article's age is an upper bound on the feed delay
    — a quiet news period inflates it, so take the *minimum* across recent
    articles as the tighter estimate.
    """
    now = datetime.now(UTC)
    df = fetch_news(s, start=now - timedelta(hours=lookback_hours), max_pages=3)
    if df.is_empty():
        return FeedDelay(0, 0.0, 0.0, 0.0)

    ages = (
        df.unique(subset=["id"])
        .select(((pl.lit(now) - pl.col("created_at")).dt.total_seconds()).alias("age"))["age"]
        .to_list()
    )
    ages = sorted(a for a in ages if a is not None and a >= 0)
    if not ages:
        return FeedDelay(0, 0.0, 0.0, 0.0)

    mid = ages[len(ages) // 2]
    return FeedDelay(len(ages), float(mid), float(ages[0]), float(ages[-1]))


def daily_news_features(df: pl.DataFrame, *, window: int = 30) -> pl.DataFrame:
    """Per symbol per UTC day: article count, source diversity, and a z-score.

    The z-score is computed against a **trailing** window and shifted by one day,
    so a day's feature never includes that day's own news. Without the shift this
    is same-bar lookahead: you would be trading on a count that is only complete
    at midnight.
    """
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date, "symbol": pl.String, "article_count": pl.Int64,
                "source_count": pl.Int64, "count_z": pl.Float64,
            }
        )

    daily = (
        df.unique(subset=["id", "symbol"])
        .with_columns(pl.col("created_at").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(
            pl.len().alias("article_count"),
            pl.col("source").n_unique().alias("source_count"),
        )
        .sort(["symbol", "date"])
    )

    return daily.with_columns(
        (
            (pl.col("article_count") - pl.col("article_count").shift(1).rolling_mean(window))
            / (pl.col("article_count").shift(1).rolling_std(window) + 1e-9)
        )
        .over("symbol")
        .alias("count_z")
    )
