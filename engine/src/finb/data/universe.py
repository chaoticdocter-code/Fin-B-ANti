"""Point-in-time snapshots of the tradeable universe.

The single most time-sensitive component in the project, and one of the
smallest. Every day it does not run is a day of survivorship bias that can never
be reconstructed.

The failure it prevents: you build a crypto momentum backtest over 2024 using
the pairs Alpaca lists *today*. Every pair that was delisted, collapsed, or
depegged in between has silently vanished from the universe. Your backtest only
ever considers survivors, and it will look excellent — not because the strategy
works, but because you removed every way to lose. In crypto this is the most
extreme survivorship environment in finance. Backfilling today's survivors into
last year is not a bias; it is a fabrication.

There is no way to fix this retroactively. Alpaca does not publish historical
asset lists. The only cure is to start writing them down, which is why this
module exists before the model does.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from finb.config import Settings
from finb.log import get_logger

log = get_logger("universe")

UNIVERSE_SCHEMA: dict[str, pl.DataType] = {
    "snapshot_date": pl.Date,
    "symbol": pl.String,
    "name": pl.String,
    "asset_class": pl.String,
    "exchange": pl.String,
    "status": pl.String,
    "tradable": pl.Boolean,
    "marginable": pl.Boolean,
    "shortable": pl.Boolean,
    "easy_to_borrow": pl.Boolean,
    "fractionable": pl.Boolean,
    "min_order_size": pl.Float64,
    "min_trade_increment": pl.Float64,
    "price_increment": pl.Float64,
}


def _as_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch_alpaca_universe(settings: Settings) -> pl.DataFrame:
    """Pull the current tradeable asset list from Alpaca (equities + crypto).

    Read-only. Lists instruments; places nothing.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass as AlpacaAssetClass
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    if not (settings.alpaca_api_key_id and settings.alpaca_api_secret_key):
        raise RuntimeError("Alpaca credentials are not configured — see .env")

    client = TradingClient(
        api_key=settings.alpaca_api_key_id,
        secret_key=settings.alpaca_api_secret_key,
        paper=settings.alpaca_paper,
    )

    today = datetime.now(UTC).date()
    rows: list[dict] = []

    for cls in (AlpacaAssetClass.US_EQUITY, AlpacaAssetClass.CRYPTO):
        assets = client.get_all_assets(
            GetAssetsRequest(asset_class=cls, status=AssetStatus.ACTIVE)
        )
        log.info(f"{cls.value}: {len(assets)} active assets")
        for a in assets:
            rows.append(
                {
                    "snapshot_date": today,
                    "symbol": str(a.symbol),
                    "name": str(a.name or ""),
                    "asset_class": str(getattr(a.asset_class, "value", a.asset_class)),
                    "exchange": str(getattr(a.exchange, "value", a.exchange) or ""),
                    "status": str(getattr(a.status, "value", a.status)),
                    "tradable": bool(a.tradable),
                    "marginable": bool(a.marginable),
                    "shortable": bool(a.shortable),
                    "easy_to_borrow": bool(a.easy_to_borrow),
                    "fractionable": bool(a.fractionable),
                    "min_order_size": _as_float(getattr(a, "min_order_size", None)),
                    "min_trade_increment": _as_float(getattr(a, "min_trade_increment", None)),
                    "price_increment": _as_float(getattr(a, "price_increment", None)),
                }
            )

    return pl.DataFrame(rows, schema=UNIVERSE_SCHEMA)


class UniverseArchive:
    """Dated snapshots on disk: ``data/curated/universe/YYYY-MM-DD.parquet``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "curated" / "universe"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, d: date) -> Path:
        return self.root / f"{d.isoformat()}.parquet"

    def write(self, df: pl.DataFrame) -> Path:
        if df.is_empty():
            raise ValueError("refusing to write an empty universe snapshot")
        d = df["snapshot_date"][0]
        path = self._path(d)
        df.write_parquet(path, compression="zstd")
        return path

    def dates(self) -> list[date]:
        out = []
        for p in self.root.glob("*.parquet"):
            try:
                out.append(date.fromisoformat(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def load(self, d: date) -> pl.DataFrame:
        path = self._path(d)
        if not path.exists():
            raise FileNotFoundError(f"no universe snapshot for {d}")
        return pl.read_parquet(path)

    def asof(self, d: date) -> pl.DataFrame:
        """The most recent snapshot at or before `d`.

        Raises rather than falling forward. Using a *later* snapshot to decide
        what was tradeable on an earlier date is precisely the lookahead this
        module exists to prevent, so there is no lenient mode.
        """
        candidates = [x for x in self.dates() if x <= d]
        if not candidates:
            raise FileNotFoundError(
                f"no universe snapshot on or before {d}; the earliest is "
                f"{self.dates()[0] if self.dates() else 'none'}. Refusing to use a "
                "later snapshot — that would be survivorship bias."
            )
        return self.load(max(candidates))

    def tradable_symbols(
        self, d: date, asset_class: str = "crypto", *, fractionable_only: bool = False
    ) -> list[str]:
        df = self.asof(d).filter(
            (pl.col("asset_class") == asset_class) & pl.col("tradable")
        )
        if fractionable_only:
            df = df.filter(pl.col("fractionable"))
        return sorted(df["symbol"].to_list())

    def coverage(self) -> dict:
        ds = self.dates()
        if not ds:
            return {"snapshots": 0, "first": None, "last": None, "gap_days": None}
        return {
            "snapshots": len(ds),
            "first": ds[0],
            "last": ds[-1],
            "gap_days": (datetime.now(UTC).date() - ds[-1]).days,
        }

    def delistings(self, since: date) -> list[str]:
        """Symbols present in the `since` snapshot but absent from the latest.

        The whole payoff of keeping these files. After a few months this is a
        real, local record of what disappeared — the survivorship correction no
        vendor will sell you.
        """
        ds = self.dates()
        if len(ds) < 2:
            return []
        old = set(self.asof(since)["symbol"].to_list())
        new = set(self.load(ds[-1])["symbol"].to_list())
        return sorted(old - new)
