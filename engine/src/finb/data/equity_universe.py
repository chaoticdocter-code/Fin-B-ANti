"""Selecting a tradeable equity universe.

Three filters, and the third is the one people forget at this account size.

**Shortable and easy-to-borrow.** 5,151 of 14,162 listed names are shortable.
"Shortable" means the broker permits it; "easy to borrow" means there is stock
available without a locate fee. Requiring both avoids discovering at the point
of order that the borrow has gone.

**Liquid.** Ranked on dollar volume — but note this is *IEX* dollar volume, about
2.4% of consolidated. It is a biased sample and a poor estimate of true volume,
yet a decent *ranking* signal: a name IEX barely prints is not a name that trades
heavily anywhere. Use it to order candidates, never as a level.

**Affordable as a short.** The constraint that bites at $500. Alpaca supports
fractional shares, but **not fractional short sales** — a short must be at least
one whole share. With a 25% position cap on a $500 book, that is $125, so any
stock above ~$125 cannot be shorted at all here. NVDA at $180 is simply not
shortable by this account, regardless of signal.

Longs do not have that problem: fractional buying means any price works.
The practical consequence is an **asymmetric universe** — everything is longable,
only cheap names are shortable — and a long/short strategy that ignores this
will keep generating short signals it cannot execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl

from finb.config import Settings
from finb.data.universe import UniverseArchive
from finb.log import get_logger

log = get_logger("equity-universe")

BATCH = 200


@dataclass(frozen=True, slots=True)
class EquityCandidate:
    symbol: str
    price: float
    dollar_volume: float
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool

    def can_short(self, max_position_usd: float) -> bool:
        """One whole share must fit inside the position cap."""
        return (
            self.shortable
            and self.easy_to_borrow
            and self.price > 0
            and self.price <= max_position_usd
        )


def _latest_bars(s: Settings, symbols: list[str], lookback_days: int = 10) -> pl.DataFrame:
    """Most recent daily bar per symbol, batched."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=s.alpaca_api_key_id, secret_key=s.alpaca_api_secret_key
    )
    end = datetime.now(UTC) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)

    rows = []
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i : i + BATCH]
        try:
            resp = client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed="iex",
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"batch {i // BATCH}: {type(e).__name__}")
            continue

        for sym, bars in resp.data.items():
            if not bars:
                continue
            recent = bars[-5:]
            px = float(recent[-1].close)
            dv = sum(float(b.close) * float(b.volume) for b in recent) / len(recent)
            rows.append({"symbol": sym, "price": px, "dollar_volume": dv})

        log.info(f"priced {i + len(chunk)}/{len(symbols)}")

    return pl.DataFrame(
        rows, schema={"symbol": pl.String, "price": pl.Float64, "dollar_volume": pl.Float64}
    )


def build_equity_universe(
    s: Settings,
    *,
    top_n: int = 150,
    max_candidates: int = 1500,
    require_shortable: bool = True,
) -> list[EquityCandidate]:
    """Screen the archive, price the survivors, and return the most liquid.

    `max_candidates` bounds the pricing work; the archive is filtered on flags
    first, which is free, and only that subset is priced.
    """
    archive = UniverseArchive(s.finb_data_dir)
    snap = archive.asof(datetime.now(UTC).date())

    eq = snap.filter(
        (pl.col("asset_class") == "us_equity")
        & pl.col("tradable")
        & pl.col("fractionable")   # needed for notional longs at $500
    )
    if require_shortable:
        eq = eq.filter(pl.col("shortable") & pl.col("easy_to_borrow"))

    symbols = eq["symbol"].to_list()[:max_candidates]
    log.info(f"{len(symbols)} candidates pass the flag screen")

    priced = _latest_bars(s, symbols)
    if priced.is_empty():
        return []

    flags = {
        r["symbol"]: r
        for r in eq.select(
            "symbol", "shortable", "easy_to_borrow", "fractionable"
        ).to_dicts()
    }

    ranked = priced.sort("dollar_volume", descending=True).head(top_n)
    out = []
    for r in ranked.to_dicts():
        f = flags.get(r["symbol"])
        if not f:
            continue
        out.append(
            EquityCandidate(
                symbol=r["symbol"],
                price=r["price"],
                dollar_volume=r["dollar_volume"],
                shortable=bool(f["shortable"]),
                easy_to_borrow=bool(f["easy_to_borrow"]),
                fractionable=bool(f["fractionable"]),
            )
        )
    return out


def summarise(candidates: list[EquityCandidate], max_position_usd: float) -> dict:
    shortable = [c for c in candidates if c.can_short(max_position_usd)]
    too_expensive = [
        c for c in candidates
        if c.shortable and c.easy_to_borrow and c.price > max_position_usd
    ]
    return {
        "candidates": len(candidates),
        "longable": len(candidates),
        "shortable": len(shortable),
        "blocked_by_share_price": len(too_expensive),
        "max_position_usd": max_position_usd,
        "median_price": (
            sorted(c.price for c in candidates)[len(candidates) // 2] if candidates else 0.0
        ),
    }
