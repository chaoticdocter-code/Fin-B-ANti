"""Crypto Sentiment & Funding Rate Data Interface.

Fetches keyless funding rate statistics and market sentiment proxies using
public endpoints (CCXT / Binance Public API / Alternative.me Crypto Fear & Greed).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from finb.log import get_logger

log = get_logger("sentiment")

FEAR_GREED_URL = "https://api.alternative.me/fng/"


@dataclass(frozen=True, slots=True)
class CryptoSentiment:
    timestamp: datetime
    fear_greed_index: int  # 0 to 100 (0=Extreme Fear, 100=Extreme Greed)
    classification: str     # e.g., "Extreme Fear", "Greed"

    @property
    def is_extreme_fear(self) -> bool:
        return self.fear_greed_index <= 25

    @property
    def is_extreme_greed(self) -> bool:
        return self.fear_greed_index >= 75


def fetch_crypto_sentiment() -> CryptoSentiment | None:
    """Fetch current Crypto Fear & Greed index from Alternative.me (free, keyless)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(FEAR_GREED_URL)
            if resp.status_code == 200:
                data = resp.json()
                item = data.get("data", [])[0]
                val = int(item["value"])
                clas = str(item["value_classification"])
                ts = datetime.fromtimestamp(int(item["timestamp"]), tz=UTC)
                log.info(f"Crypto Sentiment: {val} ({clas})")
                return CryptoSentiment(timestamp=ts, fear_greed_index=val, classification=clas)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Failed to fetch crypto sentiment: {type(e).__name__} ({e})")
    return None
