"""US market calendar and session arithmetic.

Written rather than pulled from a library because the dependency-light options
are stale and the heavy ones want pandas at import time. The rules below are
small, stable, and testable, and every downstream constraint — PDT windows, T+1
settlement, the daily session — depends on getting them exactly right.

Crypto has no calendar. `is_trading_day` and friends are equities-only; crypto
paths should not consult them at all.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
PREMARKET_OPEN = time(4, 0)
AFTERHOURS_CLOSE = time(20, 0)


def easter(year: int) -> date:
    """Gregorian Easter Sunday, via the anonymous algorithm. Gives Good Friday.

    Verified against 2024-03-31, 2025-04-20, 2026-04-05 in the tests — this is
    fiddly enough to be worth pinning to known dates rather than trusting.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of a month. Negative n counts from the end."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset + 7 * (-n - 1))


def _observed(d: date) -> date | None:
    """Shift a fixed-date holiday to the day the market actually closes.

    Saturday -> preceding Friday, Sunday -> following Monday. The exception is
    New Year's Day on a Saturday: the NYSE does not close the preceding Friday,
    because that Friday belongs to the prior year.
    """
    if d.weekday() == 5:  # Saturday
        if d.month == 1 and d.day == 1:
            return None
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures for a calendar year."""
    out: set[date] = set()

    for fixed in (
        date(year, 1, 1),    # New Year's Day
        date(year, 6, 19),   # Juneteenth (NYSE holiday since 2022)
        date(year, 7, 4),    # Independence Day
        date(year, 12, 25),  # Christmas
    ):
        if fixed.month == 6 and year < 2022:
            continue
        if (obs := _observed(fixed)) is not None:
            out.add(obs)

    out.add(_nth_weekday(year, 1, 0, 3))    # MLK Jr Day — 3rd Monday of January
    out.add(_nth_weekday(year, 2, 0, 3))    # Washington's Birthday — 3rd Monday Feb
    out.add(easter(year) - timedelta(days=2))  # Good Friday
    out.add(_nth_weekday(year, 5, 0, -1))   # Memorial Day — last Monday of May
    out.add(_nth_weekday(year, 9, 0, 1))    # Labor Day — 1st Monday of September
    out.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving — 4th Thursday November

    return frozenset(out)


@lru_cache(maxsize=64)
def nyse_early_closes(year: int) -> frozenset[date]:
    """Days the market closes at 13:00 ET."""
    out: set[date] = set()

    # Day after Thanksgiving.
    out.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))

    # Christmas Eve, when it is a weekday and not itself the observed holiday.
    ce = date(year, 12, 24)
    if ce.weekday() < 5 and ce not in nyse_holidays(year):
        out.add(ce)

    # July 3rd, when Independence Day falls midweek.
    j3 = date(year, 7, 3)
    if j3.weekday() < 5 and j3 not in nyse_holidays(year):
        out.add(j3)

    return frozenset(out)


def is_trading_day(d: date) -> bool:
    """True if US equity markets have a regular session that day."""
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date, n: int = 1) -> date:
    for _ in range(n):
        d += timedelta(days=1)
        while not is_trading_day(d):
            d += timedelta(days=1)
    return d


def previous_trading_day(d: date, n: int = 1) -> date:
    for _ in range(n):
        d -= timedelta(days=1)
        while not is_trading_day(d):
            d -= timedelta(days=1)
    return d


def trading_days(start: date, end: date) -> list[date]:
    """Trading days in [start, end], inclusive."""
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def settlement_date(trade_date: date, days: int = 1) -> date:
    """T+1 settlement for US equities (SEC rule effective 28 May 2024).

    Crypto settles instantly and must not use this.
    """
    return next_trading_day(trade_date, days)


def session_close(d: date) -> time:
    return EARLY_CLOSE if d in nyse_early_closes(d.year) else MARKET_CLOSE


def is_market_open(ts: datetime, *, extended: bool = False) -> bool:
    """Whether US equities are tradeable at `ts` (tz-aware, any zone)."""
    et = ts.astimezone(ET)
    if not is_trading_day(et.date()):
        return False
    t = et.time()
    if extended:
        return PREMARKET_OPEN <= t < AFTERHOURS_CLOSE
    return MARKET_OPEN <= t < session_close(et.date())
