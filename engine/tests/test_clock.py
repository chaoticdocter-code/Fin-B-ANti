"""Calendar rules checked against known-correct NYSE dates.

Every downstream constraint reads this module, so an off-by-one here becomes a
silently wrong backtest rather than a crash.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from finb.clock import (
    ET,
    easter,
    is_market_open,
    is_trading_day,
    next_trading_day,
    nyse_early_closes,
    nyse_holidays,
    previous_trading_day,
    session_close,
    settlement_date,
    trading_days,
)


@pytest.mark.parametrize(
    "year,expected",
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))],
)
def test_easter(year, expected):
    assert easter(year) == expected


def test_2025_holidays_match_the_published_nyse_calendar():
    assert nyse_holidays(2025) == frozenset(
        {
            date(2025, 1, 1),    # New Year's Day
            date(2025, 1, 20),   # MLK Jr
            date(2025, 2, 17),   # Washington's Birthday
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 26),   # Memorial Day
            date(2025, 6, 19),   # Juneteenth
            date(2025, 7, 4),    # Independence Day
            date(2025, 9, 1),    # Labor Day
            date(2025, 11, 27),  # Thanksgiving
            date(2025, 12, 25),  # Christmas
        }
    )


def test_2026_holidays():
    h = nyse_holidays(2026)
    assert date(2026, 4, 3) in h            # Good Friday
    assert date(2026, 7, 3) in h            # Jul 4 falls Saturday -> observed Friday
    assert date(2026, 7, 4) not in h
    assert date(2026, 11, 26) in h          # Thanksgiving
    assert date(2026, 1, 19) in h           # MLK


def test_weekend_holidays_shift_the_way_the_nyse_actually_shifts():
    # Christmas 2021 was a Saturday -> market closed Friday the 24th.
    assert date(2021, 12, 24) in nyse_holidays(2021)
    # Independence Day 2021 was a Sunday -> closed Monday the 5th.
    assert date(2021, 7, 5) in nyse_holidays(2021)
    # Christmas 2022 was a Sunday -> closed Monday the 26th.
    assert date(2022, 12, 26) in nyse_holidays(2022)


def test_new_years_on_a_saturday_does_not_close_the_prior_friday():
    """The NYSE exception: 1 Jan 2022 was a Saturday and 31 Dec 2021 stayed open."""
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)
    assert is_trading_day(date(2021, 12, 31))


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 18) not in nyse_holidays(2021)
    assert date(2021, 6, 21) not in nyse_holidays(2021)
    assert date(2022, 6, 20) in nyse_holidays(2022)  # 19th was a Sunday


def test_trading_day_navigation():
    # Friday 2026-01-16, then MLK Monday 2026-01-19 is closed.
    assert is_trading_day(date(2026, 1, 16))
    assert not is_trading_day(date(2026, 1, 17))  # Saturday
    assert not is_trading_day(date(2026, 1, 19))  # MLK
    assert next_trading_day(date(2026, 1, 16)) == date(2026, 1, 20)
    assert previous_trading_day(date(2026, 1, 20)) == date(2026, 1, 16)
    assert next_trading_day(date(2026, 1, 16), 2) == date(2026, 1, 21)


def test_settlement_skips_the_holiday_weekend():
    """A Friday trade before a long weekend settles Tuesday, not Saturday."""
    assert settlement_date(date(2026, 1, 16)) == date(2026, 1, 20)
    assert settlement_date(date(2026, 3, 10)) == date(2026, 3, 11)


def test_trading_days_range_excludes_closures():
    days = trading_days(date(2026, 6, 29), date(2026, 7, 6))
    assert date(2026, 7, 3) not in days   # observed Independence Day
    assert date(2026, 7, 4) not in days   # Saturday
    assert date(2026, 7, 6) in days       # Monday
    assert len(days) == 5


def test_early_closes():
    assert date(2024, 11, 29) in nyse_early_closes(2024)  # day after Thanksgiving
    assert date(2024, 12, 24) in nyse_early_closes(2024)  # Christmas Eve, a Tuesday
    assert date(2024, 7, 3) in nyse_early_closes(2024)    # Jul 4 was a Thursday

    assert session_close(date(2024, 11, 29)).hour == 13
    assert session_close(date(2024, 11, 26)).hour == 16


def test_is_market_open_respects_session_and_timezone():
    assert is_market_open(datetime(2026, 3, 10, 10, 0, tzinfo=ET))
    assert not is_market_open(datetime(2026, 3, 10, 9, 0, tzinfo=ET))
    assert not is_market_open(datetime(2026, 3, 10, 16, 0, tzinfo=ET))
    assert is_market_open(datetime(2026, 3, 10, 9, 0, tzinfo=ET), extended=True)

    # Same instant expressed in UTC must give the same answer.
    utc = datetime(2026, 3, 10, 14, 30, tzinfo=ZoneInfo("UTC"))  # 10:30 ET
    assert is_market_open(utc)

    assert not is_market_open(datetime(2026, 1, 19, 11, 0, tzinfo=ET))  # MLK


def test_early_close_shortens_the_session():
    assert is_market_open(datetime(2024, 11, 29, 12, 0, tzinfo=ET))
    assert not is_market_open(datetime(2024, 11, 29, 14, 0, tzinfo=ET))
