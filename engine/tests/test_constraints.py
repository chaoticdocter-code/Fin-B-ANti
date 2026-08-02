"""Account mechanics.

Note what these tests do *not* assert any more: FINRA's pattern day trader rule
was repealed effective 2026-06-04, so there is no day-trade cap to test. The
`LEGACY_PDT` policy is retained solely so historical backtests can reproduce the
constraints that actually applied before that date.

The settlement tests are unchanged, because the amendment did not touch cash
accounts — and at $500 settlement is the constraint that actually binds.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from finb.clock import ET
from finb.sim.constraints import (
    ALPACA,
    LEGACY_PDT,
    AccountType,
    AssetClass,
    BrokerPolicy,
    CashSettlement,
    DayTradeMonitor,
    Fill,
    Side,
    round_trip_capacity,
)


def fill(sym, side, d, qty=10, price=50.0, asset=AssetClass.EQUITY):
    return Fill(sym, side, qty, price, datetime(d.year, d.month, d.day, 10, 30, tzinfo=ET), asset)


def round_trip(mon, sym, d, **kw):
    mon.record(fill(sym, Side.BUY, d, **kw))
    mon.record(fill(sym, Side.SELL, d, **kw))


# --------------------------------------------------------------------------- #
#  Post-repeal behaviour
# --------------------------------------------------------------------------- #


def test_no_day_trade_cap_applies_by_default():
    """Twenty day trades in a week at $500. Formerly an instant PDT flag; now
    simply not a regulatory event."""
    mon = DayTradeMonitor()
    days = [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30)]
    for d in days:
        for i in range(5):
            round_trip(mon, f"S{i}", d)

    s = mon.status(date(2026, 7, 30), equity=500, account=AccountType.MARGIN)
    assert s.day_trades_in_window == 20
    assert not s.restricted
    assert s.remaining_day_trades is None
    assert "repealed" in s.reason


def test_a_small_margin_account_is_told_it_effectively_trades_as_cash():
    mon = DayTradeMonitor()
    s = mon.status(date(2026, 7, 30), equity=500, account=AccountType.MARGIN)
    assert "2,000" in s.reason and "cash account" in s.reason

    funded = mon.status(date(2026, 7, 30), equity=5_000, account=AccountType.MARGIN)
    assert "2,000" not in funded.reason


def test_cash_account_is_reminded_that_settlement_still_binds():
    mon = DayTradeMonitor()
    s = mon.status(date(2026, 7, 30), equity=500, account=AccountType.CASH)
    assert not s.restricted
    assert "T+1" in s.reason


def test_a_broker_may_still_impose_its_own_cap():
    """Brokers can be stricter than FINRA, especially during the phase-in to
    2027-10-20. The limit is modelled as broker policy, not regulation."""
    mon = DayTradeMonitor()
    for i, d in enumerate([date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]):
        round_trip(mon, f"S{i}", d)

    assert not mon.status(date(2026, 7, 29), 500, AccountType.MARGIN, ALPACA).restricted

    strict = mon.status(date(2026, 7, 29), 500, AccountType.MARGIN, LEGACY_PDT)
    assert strict.restricted
    assert strict.remaining_day_trades == 0
    assert "caps day trades at 3" in strict.reason


def test_custom_broker_policy():
    mon = DayTradeMonitor()
    for i, d in enumerate([date(2026, 7, 29), date(2026, 7, 30)]):
        round_trip(mon, f"S{i}", d)

    lenient = BrokerPolicy(name="lenient", max_day_trades_per_5d=10)
    s = mon.status(date(2026, 7, 30), 500, AccountType.MARGIN, lenient)
    assert not s.restricted
    assert s.remaining_day_trades == 8


# --------------------------------------------------------------------------- #
#  Day-trade counting (still useful: turnover drives cost)
# --------------------------------------------------------------------------- #


def test_the_window_rolls_so_old_day_trades_drop_out():
    mon = DayTradeMonitor()
    for i, d in enumerate([date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 5)]):
        round_trip(mon, f"S{i}", d)

    assert mon.status(date(2026, 3, 5), 500, AccountType.MARGIN).day_trades_in_window == 4
    assert mon.status(date(2026, 3, 12), 500, AccountType.MARGIN).day_trades_in_window == 0


def test_window_counts_business_days_skipping_a_holiday():
    """Trading days: Wed 14, Thu 15, Fri 16, [weekend], [MLK Mon 19], Tue 20, Wed 21, Thu 22.
    The window ending Tue 20 is [20, 16, 15, 14, 13], so a trade on the 14th only
    falls out once the window ends on Thu 22 — eight calendar days later."""
    mon = DayTradeMonitor()
    round_trip(mon, "X", date(2026, 1, 14))

    assert mon.status(date(2026, 1, 20), 500, AccountType.MARGIN).day_trades_in_window == 1
    assert mon.status(date(2026, 1, 21), 500, AccountType.MARGIN).day_trades_in_window == 1
    assert mon.status(date(2026, 1, 22), 500, AccountType.MARGIN).day_trades_in_window == 0


def test_multiple_fills_in_one_symbol_are_one_day_trade():
    mon = DayTradeMonitor()
    d = date(2026, 3, 10)
    for _ in range(3):
        mon.record(fill("AAPL", Side.BUY, d))
    mon.record(fill("AAPL", Side.SELL, d, qty=30))

    assert mon.status(d, 500, AccountType.MARGIN).day_trades_in_window == 1


def test_buying_without_selling_is_not_a_day_trade():
    mon = DayTradeMonitor()
    d = date(2026, 3, 10)
    mon.record(fill("AAPL", Side.BUY, d))
    assert mon.status(d, 500, AccountType.MARGIN).day_trades_in_window == 0


def test_crypto_is_never_counted():
    mon = DayTradeMonitor()
    d = date(2026, 3, 10)
    for _ in range(50):
        round_trip(mon, "BTC/USD", d, asset=AssetClass.CRYPTO)

    assert mon.status(d, 500, AccountType.MARGIN, LEGACY_PDT).day_trades_in_window == 0


def test_would_be_day_trade_predicts_before_the_order_goes_out():
    mon = DayTradeMonitor()
    d = date(2026, 3, 10)
    assert not mon.would_be_day_trade("AAPL", Side.SELL, d)
    mon.record(fill("AAPL", Side.BUY, d))
    assert mon.would_be_day_trade("AAPL", Side.SELL, d)
    assert not mon.would_be_day_trade("MSFT", Side.SELL, d)


# --------------------------------------------------------------------------- #
#  T+1 settlement — unchanged by the Rule 4210 amendment
# --------------------------------------------------------------------------- #


def test_proceeds_settle_the_next_trading_day():
    acct = CashSettlement(settled_cash=500.0)
    acct.buy("AAPL", 10, 50.0, date(2026, 3, 10))
    assert acct.settled_cash == 0.0

    acct.sell("AAPL", 10, 51.0, date(2026, 3, 10))
    assert acct.unsettled_cash == 510.0
    assert acct.settled_cash == 0.0

    acct.advance_to(date(2026, 3, 11))
    assert acct.settled_cash == 510.0
    assert acct.unsettled_cash == 0.0


def test_selling_before_the_funding_proceeds_settle_is_a_good_faith_violation():
    acct = CashSettlement(settled_cash=500.0)

    acct.buy("AAA", 10, 50.0, date(2026, 3, 10))
    acct.sell("AAA", 10, 50.0, date(2026, 3, 10))       # proceeds settle 3-11

    acct.buy("BBB", 10, 50.0, date(2026, 3, 10))        # funded by unsettled cash
    assert acct.sell("BBB", 10, 51.0, date(2026, 3, 10)) is True
    assert "good-faith violation" in acct.violations[0]


def test_waiting_for_settlement_avoids_the_violation():
    acct = CashSettlement(settled_cash=500.0)
    acct.buy("AAA", 10, 50.0, date(2026, 3, 10))
    acct.sell("AAA", 10, 50.0, date(2026, 3, 10))

    acct.advance_to(date(2026, 3, 11))
    acct.buy("BBB", 10, 50.0, date(2026, 3, 11))
    assert acct.sell("BBB", 10, 51.0, date(2026, 3, 11)) is False
    assert acct.violations == []


def test_three_violations_restricts_the_account():
    acct = CashSettlement(settled_cash=500.0)
    for i in range(3):
        d = date(2026, 3, 10)
        acct.buy(f"A{i}", 10, 50.0, d)
        acct.sell(f"A{i}", 10, 50.0, d)
        acct.buy(f"B{i}", 10, 50.0, d)
        acct.sell(f"B{i}", 10, 50.0, d)
    assert acct.restricted


def test_cannot_spend_more_than_buying_power():
    acct = CashSettlement(settled_cash=500.0)
    with pytest.raises(ValueError, match="insufficient buying power"):
        acct.buy("AAPL", 100, 50.0, date(2026, 3, 10))


# --------------------------------------------------------------------------- #


def test_capacity_summary_says_cost_is_the_limit_not_regulation():
    crypto = round_trip_capacity(AccountType.MARGIN, AssetClass.CRYPTO)
    assert "Cost is the only limit" in crypto

    cash = round_trip_capacity(AccountType.CASH, AssetClass.EQUITY)
    assert "no day-trade limit" in cash and "T+1" in cash

    legacy = round_trip_capacity(AccountType.MARGIN, AssetClass.EQUITY, LEGACY_PDT)
    assert "3 day trades" in legacy
