"""Building and persisting the equity trading universe, then filling its history.

The trap this module exists to avoid: re-screening for "the 92 most liquid
shortable names" on every run means the universe silently becomes *today's*
survivors. Backtest that and you have selected, for every past date, the names
that went on to remain liquid and solvent — the same fabrication the crypto
[[Universe archive]] was built to prevent, one level up.

So the selected universe is **snapshotted with a date and reused**. It is
re-screened deliberately, on a schedule, and every version is kept. A backtest
covering March uses the universe as it was chosen in March, including the names
that later became untradeable.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from finb.config import Settings
from finb.data.equity_universe import EquityCandidate, build_equity_universe
from finb.data.lake import BarLake, Timeframe
from finb.log import get_logger
from finb.sim.constraints import AssetClass

log = get_logger("equity-ingest")

BATCH = 100


class EquityUniverseStore:
    """Dated snapshots of the *selected* universe, not just the listed one."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "curated" / "equity-universe"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, d: date) -> Path:
        return self.root / f"{d.isoformat()}.json"

    def save(self, candidates: list[EquityCandidate], d: date | None = None) -> Path:
        d = d or datetime.now(UTC).date()
        path = self._path(d)
        path.write_text(
            json.dumps([asdict(c) for c in candidates], indent=2), encoding="utf-8"
        )
        return path

    def dates(self) -> list[date]:
        out = []
        for p in self.root.glob("*.json"):
            try:
                out.append(date.fromisoformat(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def load(self, d: date) -> list[EquityCandidate]:
        raw = json.loads(self._path(d).read_text(encoding="utf-8"))
        return [EquityCandidate(**r) for r in raw]

    def asof(self, d: date) -> list[EquityCandidate]:
        """The universe as selected on or before `d`. Never falls forward."""
        prior = [x for x in self.dates() if x <= d]
        if not prior:
            raise FileNotFoundError(
                f"no equity universe selected on or before {d}. Refusing to use a "
                "later selection — that is survivorship bias."
            )
        return self.load(max(prior))

    def latest(self) -> list[EquityCandidate]:
        ds = self.dates()
        if not ds:
            raise FileNotFoundError("no equity universe has been selected yet")
        return self.load(ds[-1])


def select_universe(
    s: Settings, *, top_n: int = 120, max_candidates: int = 1500
) -> list[EquityCandidate]:
    """Screen and persist today's universe."""
    store = EquityUniverseStore(s.finb_data_dir)
    candidates = build_equity_universe(
        s, top_n=top_n, max_candidates=max_candidates, require_shortable=True
    )
    if candidates:
        path = store.save(candidates)
        log.info(f"selected {len(candidates)} names -> {path.name}")
    return candidates


def ingest_bars(
    s: Settings,
    symbols: list[str],
    *,
    start: datetime | None = None,
    timeframe: Timeframe = Timeframe.D1,
) -> dict:
    """Backfill daily bars for the universe into the lake.

    `adjustment="raw"` deliberately: split- and dividend-adjusted history bakes
    *future* corporate actions into past prices, so an executor trained on
    adjusted bars has been shown the future. Adjust for features, execute on raw.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    lake = BarLake(s.finb_data_dir)
    client = StockHistoricalDataClient(
        api_key=s.alpaca_api_key_id, secret_key=s.alpaca_api_secret_key
    )
    start = start or datetime(2021, 1, 1, tzinfo=UTC)
    end = datetime.now(UTC) - timedelta(minutes=20)   # free tier: SIP needs 15min

    stored, total_bars, failed = 0, 0, []
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
                    adjustment="raw",
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"batch {i // BATCH} failed: {type(e).__name__}")
            failed.extend(chunk)
            continue

        for sym, bars in resp.data.items():
            if not bars:
                continue
            df = pl.DataFrame(
                {
                    "ts": [b.timestamp for b in bars],
                    "open": [float(b.open) for b in bars],
                    "high": [float(b.high) for b in bars],
                    "low": [float(b.low) for b in bars],
                    "close": [float(b.close) for b in bars],
                    "volume": [float(b.volume) for b in bars],
                    "trade_count": [
                        int(b.trade_count) if b.trade_count is not None else None for b in bars
                    ],
                    "vwap": [float(b.vwap) if b.vwap is not None else None for b in bars],
                }
            )
            lake.write(sym, timeframe, df, asset=AssetClass.EQUITY)
            stored += 1
            total_bars += df.height

        log.info(f"ingested {min(i + BATCH, len(symbols))}/{len(symbols)} symbols")

    return {"symbols": stored, "bars": total_bars, "failed": failed}


def coverage_report(s: Settings, symbols: list[str]) -> pl.DataFrame:
    """Per-symbol bar count and span, so gaps are visible before modelling."""
    lake = BarLake(s.finb_data_dir)
    rows = []
    for sym in symbols:
        n = lake.row_count(sym, Timeframe.D1, AssetClass.EQUITY)
        cov = lake.coverage(sym, Timeframe.D1, AssetClass.EQUITY)
        rows.append(
            {
                "symbol": sym,
                "bars": n,
                "first": cov[0].date() if cov else None,
                "last": cov[1].date() if cov else None,
            }
        )
    return pl.DataFrame(rows).sort("bars")
