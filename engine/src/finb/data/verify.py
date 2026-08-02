"""Live credential probes.

`finb doctor` answers "is a key present". This answers "does it work", which is
a different question and the one that matters — a typo'd key, a revoked key, and
a key for the wrong environment all look identical to a presence check.

Every probe is **read-only**: account reads, balance reads, and single-symbol
quotes. Nothing here places, cancels, or modifies anything.

Failures are classified rather than lumped together, because the fixes differ:
a bad key needs regenerating, a rate limit needs waiting, and an unreachable
host needs neither.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx

from finb.config import Settings

Status = Literal["ok", "auth_failed", "rate_limited", "unreachable", "no_key", "error"]

TIMEOUT = 15.0


@dataclass(frozen=True, slots=True)
class ProbeResult:
    provider: str
    kind: str
    status: Status
    detail: str = ""
    latency_ms: float | None = None
    needs_key: bool = True

    @property
    def symbol(self) -> str:
        return {
            "ok": "OK",
            "auth_failed": "BAD KEY",
            "rate_limited": "RATE LIMIT",
            "unreachable": "UNREACHABLE",
            "no_key": "not configured",
            "error": "ERROR",
        }[self.status]


def _classify(exc: Exception) -> tuple[Status, str]:
    """Map an exception to a status. Never include the request URL — several
    providers carry the API key in the query string."""
    name = type(exc).__name__
    text = str(exc)

    # Strip anything that looks like a credential before it reaches a log.
    for marker in ("api_key=", "apikey=", "token="):
        if marker in text.lower():
            text = text.split(marker)[0] + f"{marker}<redacted>"

    lowered = text.lower()
    if any(s in lowered for s in ("401", "403", "unauthorized", "forbidden", "invalid api",
                                  "authentication", "permission denied", "invalid key")):
        return "auth_failed", f"{name}: {text[:160]}"
    if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
        return "rate_limited", f"{name}: {text[:160]}"
    if any(s in lowered for s in ("timeout", "timed out", "connection", "dns", "network",
                                  "unreachable", "ssl")):
        return "unreachable", f"{name}: {text[:160]}"
    return "error", f"{name}: {text[:200]}"


def _timed(fn) -> tuple[Status, str, float]:
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        return "ok", detail, (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        status, detail = _classify(e)
        return status, detail, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- #
#  Keyed providers
# --------------------------------------------------------------------------- #


def probe_alpaca_trading(s: Settings) -> ProbeResult:
    if not (s.alpaca_api_key_id and s.alpaca_api_secret_key):
        return ProbeResult("alpaca (trading)", "broker", "no_key")

    def run() -> str:
        from alpaca.trading.client import TradingClient

        c = TradingClient(
            api_key=s.alpaca_api_key_id,
            secret_key=s.alpaca_api_secret_key,
            paper=s.alpaca_paper,
        )
        a = c.get_account()
        env = "paper" if s.alpaca_paper else "LIVE"
        return f"{env} account {str(a.account_number)[-4:].rjust(8, '*')}, equity ${float(a.equity):,.0f}"

    st, detail, ms = _timed(run)
    return ProbeResult("alpaca (trading)", "broker", st, detail, ms)


def probe_alpaca_equity_data(s: Settings) -> ProbeResult:
    if not (s.alpaca_api_key_id and s.alpaca_api_secret_key):
        return ProbeResult("alpaca (equity data)", "data", "no_key")

    def run() -> str:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        c = StockHistoricalDataClient(
            api_key=s.alpaca_api_key_id, secret_key=s.alpaca_api_secret_key
        )
        end = datetime.now(UTC) - timedelta(minutes=20)
        r = c.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Day,
                start=end - timedelta(days=10),
                end=end,
                feed="iex",
            )
        )
        n = len(r.data.get("SPY", []))
        return f"IEX feed, {n} recent SPY daily bars"

    st, detail, ms = _timed(run)
    return ProbeResult("alpaca (equity data)", "data", st, detail, ms)


def probe_alpaca_news(s: Settings) -> ProbeResult:
    if not (s.alpaca_api_key_id and s.alpaca_api_secret_key):
        return ProbeResult("alpaca (news)", "news", "no_key")

    def run() -> str:
        r = httpx.get(
            "https://data.alpaca.markets/v1beta1/news",
            params={"limit": 1},
            headers={
                "APCA-API-KEY-ID": s.alpaca_api_key_id,
                "APCA-API-SECRET-KEY": s.alpaca_api_secret_key,
            },
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        n = len(r.json().get("news", []))
        return f"Benzinga feed reachable ({n} article returned)"

    st, detail, ms = _timed(run)
    return ProbeResult("alpaca (news)", "news", st, detail, ms)


def probe_fred(s: Settings) -> ProbeResult:
    if not s.fred_api_key:
        return ProbeResult("fred", "macro", "no_key")

    def run() -> str:
        r = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "VIXCLS",
                "api_key": s.fred_api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            },
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            # Never surface the URL — FRED puts the key in the query string.
            msg = r.json().get("error_message", "") if r.text else ""
            raise RuntimeError(f"HTTP {r.status_code}: {msg[:120]}")
        obs = r.json().get("observations", [])
        return f"latest VIX observation {obs[0]['date']}" if obs else "reachable"

    st, detail, ms = _timed(run)
    return ProbeResult("fred", "macro", st, detail, ms)


def probe_finnhub(s: Settings) -> ProbeResult:
    if not s.finnhub_api_key:
        return ProbeResult("finnhub", "data", "no_key")

    def run() -> str:
        r = httpx.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL"},
            headers={"X-Finnhub-Token": s.finnhub_api_key},
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        j = r.json()
        if not j or j.get("c") in (None, 0):
            raise RuntimeError("empty quote — key may lack entitlement")
        return f"AAPL quote ${j['c']:.2f}"

    st, detail, ms = _timed(run)
    return ProbeResult("finnhub", "data", st, detail, ms)


_CCXT_EXCHANGES = {
    "kraken": ("kraken_api_key", "kraken_api_secret", None, {}),
    "coinbase": ("coinbase_api_key", "coinbase_api_secret", None, {}),
    "binance": ("binance_api_key", "binance_api_secret", None, {}),
    "bybit": ("bybit_api_key", "bybit_api_secret", None, {}),
    "okx": ("okx_api_key", "okx_api_secret", "okx_passphrase", {}),
}


def probe_exchange(s: Settings, name: str) -> ProbeResult:
    """Authenticate against an exchange with a private balance read.

    `fetch_balance` is the standard way to prove a key works: it is read-only,
    requires authentication, and needs no trading permission.
    """
    key_attr, secret_attr, pass_attr, opts = _CCXT_EXCHANGES[name]
    key = getattr(s, key_attr, None)
    secret = getattr(s, secret_attr, None)
    passphrase = getattr(s, pass_attr, None) if pass_attr else None

    if not (key and secret) or (pass_attr and not passphrase):
        return ProbeResult(name, "exchange", "no_key")

    def run() -> str:
        import ccxt

        cfg = {"apiKey": key, "secret": secret, "enableRateLimit": True, "timeout": 15000, **opts}
        if passphrase:
            cfg["password"] = passphrase

        ex = getattr(ccxt, name)(cfg)
        if name == "binance" and getattr(s, "binance_testnet", False):
            ex.set_sandbox_mode(True)
        if name == "bybit" and getattr(s, "bybit_testnet", False):
            ex.set_sandbox_mode(True)

        bal = ex.fetch_balance()
        held = sorted(
            (a for a, v in (bal.get("total") or {}).items() if v),
            key=lambda a: -(bal["total"][a]),
        )[:3]
        return f"authenticated; balances held: {', '.join(held) if held else 'none'}"

    st, detail, ms = _timed(run)
    return ProbeResult(name, "exchange", st, detail, ms)


# --------------------------------------------------------------------------- #
#  Keyless sources — verify reachability, not credentials
# --------------------------------------------------------------------------- #


def probe_alpaca_crypto_data(_s: Settings) -> ProbeResult:
    def run() -> str:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        c = CryptoHistoricalDataClient()
        r = c.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=["BTC/USD"],
                timeframe=TimeFrame.Day,
                start=datetime.now(UTC) - timedelta(days=5),
            )
        )
        bars = r.data.get("BTC/USD", [])
        return f"BTC/USD last close ${float(bars[-1].close):,.0f}" if bars else "reachable"

    st, detail, ms = _timed(run)
    return ProbeResult("alpaca (crypto data)", "data", st, detail, ms, needs_key=False)


def probe_binance_bulk(_s: Settings) -> ProbeResult:
    def run() -> str:
        r = httpx.head(
            "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/"
            "BTCUSDT-1d-2026-07-01.zip",
            timeout=TIMEOUT,
            follow_redirects=True,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        size = int(r.headers.get("content-length", 0))
        return f"bulk archive reachable ({size:,} byte sample)"

    st, detail, ms = _timed(run)
    return ProbeResult("binance bulk data", "data", st, detail, ms, needs_key=False)


def probe_ccxt_public(_s: Settings) -> ProbeResult:
    def run() -> str:
        import ccxt

        t = ccxt.kraken({"enableRateLimit": True, "timeout": 15000}).fetch_ticker("BTC/USD")
        return f"public ticker BTC/USD ${float(t['last']):,.0f}"

    st, detail, ms = _timed(run)
    return ProbeResult("ccxt public", "data", st, detail, ms, needs_key=False)


def probe_sec_edgar(_s: Settings) -> ProbeResult:
    def run() -> str:
        r = httpx.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={"action": "getcompany", "CIK": "0000320193", "type": "10-K", "count": "1"},
            headers={"User-Agent": "finb-research contact@example.com"},
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        return "filings reachable (descriptive User-Agent required)"

    st, detail, ms = _timed(run)
    return ProbeResult("sec edgar", "data", st, detail, ms, needs_key=False)


def probe_gdelt(_s: Settings) -> ProbeResult:
    def run() -> str:
        # GDELT throttles anonymous clients hard. A descriptive User-Agent and
        # one retry turn a spurious 429 into a clean read — verified: without
        # the header the first call returned 429, with it the first call
        # returned articles.
        last = ""
        for attempt in range(2):
            r = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": "bitcoin", "mode": "artlist", "maxrecords": 1,
                        "format": "json"},
                headers={"User-Agent": "finb-research/0.1"},
                timeout=25.0,
            )
            if r.status_code == 200:
                return f"news API reachable ({len(r.json().get('articles', []))} article)"
            last = f"HTTP {r.status_code}"
            if attempt == 0:
                time.sleep(5)
        raise RuntimeError(last)

    st, detail, ms = _timed(run)
    return ProbeResult("gdelt", "news", st, detail, ms, needs_key=False)


def probe_yahoo(_s: Settings) -> ProbeResult:
    def run() -> str:
        import yfinance as yf

        h = yf.Ticker("SPY").history(period="5d")
        if h.empty:
            raise RuntimeError("empty response — yfinance may be blocked or broken")
        return f"SPY last close ${float(h['Close'].iloc[-1]):,.2f}"

    st, detail, ms = _timed(run)
    return ProbeResult("yahoo (yfinance)", "data", st, detail, ms, needs_key=False)


# --------------------------------------------------------------------------- #


def verify_all(s: Settings, *, include_keyless: bool = True) -> list[ProbeResult]:
    """Probe every provider. Read-only throughout."""
    results = [
        probe_alpaca_trading(s),
        probe_alpaca_equity_data(s),
        probe_alpaca_news(s),
        probe_fred(s),
        probe_finnhub(s),
    ]
    results += [probe_exchange(s, name) for name in _CCXT_EXCHANGES]

    if include_keyless:
        results += [
            probe_alpaca_crypto_data(s),
            probe_binance_bulk(s),
            probe_ccxt_public(s),
            probe_sec_edgar(s),
            probe_gdelt(s),
            probe_yahoo(s),
        ]
    return results
