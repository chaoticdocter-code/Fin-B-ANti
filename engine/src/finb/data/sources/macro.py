"""FRED macro series, fetched point-in-time.

The subtlety that makes this module worth writing carefully: **most FRED series
are revised.** GDP, non-farm payrolls, and CPI are all restated, sometimes
substantially, months after their first publication. Pulling today's values and
joining them to a 2023 backtest tells the model what the number was eventually
revised *to* — which nobody knew at the time.

That is severe lookahead, and it is silent. The revised series usually looks
*more* predictive, because revisions incorporate information that arrived later.

FRED solves this if you ask correctly. `output_type=4` returns the **initial
release only** — the number as first published, which is the number a trader
actually saw. That is the default here, and overriding it requires saying so.

Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import httpx
import polars as pl

from finb.config import Settings
from finb.log import get_logger

log = get_logger("fred")

BASE = "https://api.stlouisfed.org/fred/series/observations"

INITIAL_RELEASE_ONLY = 4
"""FRED output_type: the value as first published, before any revision."""


@dataclass(frozen=True, slots=True)
class MacroSeries:
    series_id: str
    name: str
    revised: bool
    """Whether the series is subject to revision. Revised series MUST be pulled
    initial-release-only or they leak the future."""

    note: str = ""


# Series chosen for orthogonality to price. The ML baseline failed because every
# feature was a transform of the same close series; these are not.
SERIES: dict[str, MacroSeries] = {
    "VIXCLS": MacroSeries("VIXCLS", "CBOE volatility index", False, "risk appetite"),
    "DGS10": MacroSeries("DGS10", "10-year Treasury yield", False, "discount rate"),
    "T10Y2Y": MacroSeries("T10Y2Y", "10y-2y spread", False, "curve shape / recession signal"),
    "DTWEXBGS": MacroSeries("DTWEXBGS", "Broad dollar index", False, "dollar strength"),
    "DFF": MacroSeries("DFF", "Effective fed funds rate", False, "policy stance"),
    "BAMLH0A0HYM2": MacroSeries(
        "BAMLH0A0HYM2", "High-yield credit spread", False, "credit stress"
    ),
    "CPIAUCSL": MacroSeries("CPIAUCSL", "CPI", True, "revised — initial release only"),
    "UNRATE": MacroSeries("UNRATE", "Unemployment rate", True, "revised"),
    "PAYEMS": MacroSeries("PAYEMS", "Non-farm payrolls", True, "revised, sometimes heavily"),
}


def fetch_series(
    settings: Settings,
    series_id: str,
    start: date | datetime,
    end: date | datetime | None = None,
    *,
    initial_release_only: bool = True,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Fetch one FRED series as (date, value).

    `initial_release_only` defaults to True. Turn it off only when you have a
    specific reason and have thought about revision lookahead — for a genuinely
    unrevised series such as VIX or a Treasury yield it makes no difference.
    """
    if not settings.fred_api_key:
        raise RuntimeError("FRED_API_KEY is not configured — see .env")

    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "observation_start": (start.date() if isinstance(start, datetime) else start).isoformat(),
    }
    if end is not None:
        params["observation_end"] = (
            end.date() if isinstance(end, datetime) else end
        ).isoformat()
    if initial_release_only:
        # output_type=4 searches across vintages, so it needs an explicit
        # real-time window covering all of them. Without these FRED defaults
        # realtime_start to today, finds a single vintage, and returns 400.
        params["output_type"] = INITIAL_RELEASE_ONLY
        params["realtime_start"] = "1776-07-04"   # FRED's documented earliest
        params["realtime_end"] = "9999-12-31"     # FRED's documented latest

    r = httpx.get(BASE, params=params, timeout=timeout)
    if r.status_code >= 400:
        # Never surface the response URL — FRED carries the API key in the
        # query string, so the default httpx error message leaks it into logs
        # and tracebacks.
        raise RuntimeError(
            f"FRED request for {series_id} failed with HTTP {r.status_code}: "
            f"{r.json().get('error_message', r.text[:200]) if r.text else ''}"
        )
    obs = r.json().get("observations", [])

    rows = []
    for o in obs:
        raw = o.get("value")
        if raw in (".", "", None):
            continue          # FRED marks missing observations with a dot
        try:
            rows.append({"date": date.fromisoformat(o["date"]), "value": float(raw)})
        except (ValueError, KeyError):
            continue

    df = pl.DataFrame(rows, schema={"date": pl.Date, "value": pl.Float64}).sort("date")
    log.info(
        f"{series_id}: {df.height} observations"
        f"{' (initial release only)' if initial_release_only else ''}"
    )
    return df


def fetch_panel(
    settings: Settings,
    start: date | datetime,
    end: date | datetime | None = None,
    series_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Fetch several series and align them onto one daily date index.

    Values are **forward-filled**, which is causal: a monthly figure published in
    January remains the latest known value through February. Back-filling here
    would be lookahead, which is why `fill_null(strategy="forward")` is the only
    direction used.
    """
    ids = series_ids or list(SERIES)
    out: pl.DataFrame | None = None

    for sid in ids:
        meta = SERIES.get(sid)
        df = fetch_series(
            settings, sid, start, end,
            initial_release_only=meta.revised if meta else True,
        ).rename({"value": sid})
        out = df if out is None else out.join(df, on="date", how="full", coalesce=True)

    if out is None:
        return pl.DataFrame(schema={"date": pl.Date})

    return out.sort("date").with_columns(
        [pl.col(c).fill_null(strategy="forward") for c in out.columns if c != "date"]
    )


def macro_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Turn levels into stationary, trailing-window features.

    Raw levels are non-stationary and a tree model will happily split on "the
    year is 2021" via the level of the fed funds rate. Changes and z-scores
    against a trailing window are what carry information.
    """
    exprs: list[pl.Expr] = [pl.col("date")]
    for col in panel.columns:
        if col == "date":
            continue
        c = pl.col(col)
        exprs += [
            (c - c.shift(21)).alias(f"{col}_chg21"),
            ((c - c.rolling_mean(window_size=252)) / c.rolling_std(window_size=252)).alias(
                f"{col}_z252"
            ),
        ]
    return panel.select(exprs)
