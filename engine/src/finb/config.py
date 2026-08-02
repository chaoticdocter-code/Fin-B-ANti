"""Configuration and credential detection.

Design notes
------------
The `.env` file lives at the **vault root** (``D:\\Fin B\\.env``), not next to the
code, because the vault is the project. `_find_vault_root` walks upward looking
for the ``.obsidian`` marker so this keeps working if the engine is moved or
installed as a wheel.

Nothing here raises when a credential is missing. The farm is designed to run on
whatever subset of providers happens to be configured — a missing Polygon key
disables the Polygon source and nothing else. Use `finb doctor` to see what was
detected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The literal string that must appear in FINB_ALLOW_LIVE for real-money order
# placement to even be considered. Chosen to be impossible to set by accident.
LIVE_MAGIC = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"


def _find_vault_root() -> Path:
    """Locate the Obsidian vault root (the project root).

    Order: explicit env override -> walk up from this file looking for
    ``.obsidian`` -> fall back to three levels up from the package.
    """
    if override := os.environ.get("FINB_VAULT_DIR"):
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".obsidian").is_dir():
            return parent

    # engine/src/finb/config.py -> engine/src -> engine -> <vault>
    return here.parents[3]


VAULT_ROOT = _find_vault_root()


class Settings(BaseSettings):
    """Runtime settings, populated from ``<vault>/.env`` and the environment."""

    model_config = SettingsConfigDict(
        env_file=(VAULT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- safety -----------------------------------------------------------
    finb_allow_live: str = Field(default="no")
    finb_capital_usd: float = Field(default=500.0)

    # ---- paths ------------------------------------------------------------
    finb_vault_dir: Path = Field(default=VAULT_ROOT)
    finb_data_dir: Path = Field(default=VAULT_ROOT / "data")
    finb_log_dir: Path = Field(default=VAULT_ROOT / "logs")

    # ---- session ----------------------------------------------------------
    # Anchored to Eastern because that is what the market runs on; the operator's
    # local zone is only used for display and scheduling. The window ends 30
    # minutes after the close so the session's last act runs on final bars.
    finb_timezone: str = "America/Los_Angeles"
    finb_session_start_et: str = "11:30"
    finb_session_end_et: str = "16:30"

    # ---- broker: alpaca ---------------------------------------------------
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    alpaca_paper: bool = True

    # ---- crypto exchanges -------------------------------------------------
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    binance_testnet: bool = True
    binance_us: bool = False

    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None
    bybit_testnet: bool = True

    okx_api_key: str | None = None
    okx_api_secret: str | None = None
    okx_passphrase: str | None = None
    okx_demo: bool = True

    kraken_api_key: str | None = None
    kraken_api_secret: str | None = None

    coinbase_api_key: str | None = None
    coinbase_api_secret: str | None = None

    # ---- market data ------------------------------------------------------
    polygon_api_key: str | None = None
    finnhub_api_key: str | None = None
    twelvedata_api_key: str | None = None
    tiingo_api_key: str | None = None
    alphavantage_api_key: str | None = None
    eodhd_api_key: str | None = None
    fred_api_key: str | None = None

    # ---- news -------------------------------------------------------------
    newsapi_key: str | None = None
    marketaux_api_key: str | None = None

    # ------------------------------------------------------------------ #

    def session_window_local(self) -> tuple[str, str]:
        """The ET session window rendered in the operator's own timezone."""
        from datetime import datetime, time
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        local = ZoneInfo(self.finb_timezone)
        # An arbitrary weekday in standard time; DST offsets track together for
        # US zones, so the local window is stable year-round.
        day = datetime(2026, 3, 10, tzinfo=et).date()

        out = []
        for hhmm in (self.finb_session_start_et, self.finb_session_end_et):
            h, m = (int(x) for x in hhmm.split(":"))
            out.append(
                datetime.combine(day, time(h, m), tzinfo=et)
                .astimezone(local)
                .strftime("%H:%M")
            )
        return out[0], out[1]

    @property
    def live_enabled(self) -> bool:
        """True only if the operator set the exact magic string.

        Note this is necessary but *not* sufficient — `finb.risk` additionally
        requires a strategy to have cleared the promotion gate.
        """
        return self.finb_allow_live.strip() == LIVE_MAGIC

    def assert_paper_only(self) -> None:
        """Guard for any code path that could reach a real order book."""
        if self.live_enabled:
            raise RuntimeError(
                "Live trading is enabled but this code path is paper-only. "
                "Refusing to continue."
            )


@dataclass(frozen=True, slots=True)
class Provider:
    """One external service and whether we have what we need to use it."""

    name: str
    kind: str  # broker | exchange | data | news | macro
    configured: bool
    needs_key: bool
    note: str


def detect_providers(s: Settings) -> list[Provider]:
    """Report which providers are usable given the current credentials.

    `needs_key=False` entries work with no credential at all — these are the
    backbone of a zero-budget build and are always available.
    """

    def both(a: str | None, b: str | None) -> bool:
        return bool(a and b)

    return [
        # --- keyless: always available -------------------------------------
        Provider("binance-public-data", "data", True, False,
                 "Bulk historical klines/aggTrades from data.binance.vision. No key, no limit."),
        Provider("ccxt-public", "data", True, False,
                 "Public OHLCV/orderbook across 100+ exchanges. No key needed."),
        Provider("yahoo-finance", "data", True, False,
                 "Daily + limited intraday equities history via yfinance. Unofficial; can break."),
        Provider("sec-edgar", "data", True, False,
                 "Filings + fundamentals. Free, requires a descriptive User-Agent header."),
        Provider("gdelt", "news", True, False,
                 "Global news event stream, 15-min cadence. Entirely free."),

        # --- keyed ---------------------------------------------------------
        Provider("alpaca", "broker", both(s.alpaca_api_key_id, s.alpaca_api_secret_key), True,
                 "US equities + crypto paper trading, plus Benzinga news feed."),
        Provider("binance", "exchange",
                 both(s.binance_api_key, s.binance_api_secret), True,
                 "Testnet order placement." + (" Binance.US endpoint." if s.binance_us else "")),
        Provider("bybit", "exchange", both(s.bybit_api_key, s.bybit_api_secret), True,
                 "Testnet order placement."),
        Provider("okx", "exchange",
                 both(s.okx_api_key, s.okx_api_secret) and bool(s.okx_passphrase), True,
                 "Demo trading. Requires key + secret + passphrase."),
        Provider("kraken", "exchange", both(s.kraken_api_key, s.kraken_api_secret), True, ""),
        Provider("coinbase", "exchange", both(s.coinbase_api_key, s.coinbase_api_secret), True, ""),

        Provider("polygon", "data", bool(s.polygon_api_key), True, ""),
        Provider("finnhub", "data", bool(s.finnhub_api_key), True, ""),
        Provider("twelvedata", "data", bool(s.twelvedata_api_key), True, ""),
        Provider("tiingo", "data", bool(s.tiingo_api_key), True, ""),
        Provider("alphavantage", "data", bool(s.alphavantage_api_key), True, ""),
        Provider("eodhd", "data", bool(s.eodhd_api_key), True, ""),
        Provider("fred", "macro", bool(s.fred_api_key), True,
                 "Macro series. Free key, instant signup — worth getting if missing."),

        Provider("newsapi", "news", bool(s.newsapi_key), True, ""),
        Provider("marketaux", "news", bool(s.marketaux_api_key), True, ""),
    ]


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
