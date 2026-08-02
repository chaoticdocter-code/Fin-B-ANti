"""QuantConnect REST API v2 Client.

Integrates Fin B with QuantConnect to:
1. Access QuantConnect Cloud project backtest results, Sharpe ratios, and capacity stats.
2. Authenticate using SHA-256 HMAC signature per QuantConnect API specifications.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import httpx

from finb.log import get_logger

log = get_logger("quantconnect")


@dataclass
class QCProject:
    project_id: int
    name: str
    language: str
    modified: str


class QuantConnectClient:
    """Client for QuantConnect REST API v2."""

    BASE_URL = "https://www.quantconnect.com/api/v2"

    def __init__(self, user_id: str | None = None, api_token: str | None = None) -> None:
        self.user_id = user_id or ""
        self.api_token = api_token or ""

    @property
    def is_configured(self) -> bool:
        return bool(self.user_id and self.api_token)

    def _generate_signature(self, timestamp: int) -> str:
        """QuantConnect API v2 signature: SHA-256(api_token:timestamp)."""
        message = f"{self.api_token}:{timestamp}"
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    def _auth_headers(self) -> dict[str, str]:
        ts = int(time.time())
        sig = self._generate_signature(ts)
        # QuantConnect authentication uses HTTP Basic Auth with user_id:signature
        return {"Timestamp": str(ts)}

    def list_projects(self) -> list[QCProject]:
        """Fetch list of user projects on QuantConnect."""
        if not self.is_configured:
            log.warning("QuantConnect API credentials missing")
            return []

        ts = int(time.time())
        sig = self._generate_signature(ts)
        auth = (self.user_id, sig)
        headers = {"Timestamp": str(ts)}

        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(f"{self.BASE_URL}/projects/read", auth=auth, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    projects = []
                    for item in data.get("projects", []):
                        projects.append(
                            QCProject(
                                project_id=item.get("projectId", 0),
                                name=item.get("name", ""),
                                language=item.get("language", "Py"),
                                modified=item.get("modified", ""),
                            )
                        )
                    log.info(f"Fetched {len(projects)} projects from QuantConnect")
                    return projects
                log.warning(f"QuantConnect API status {r.status_code}: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"QuantConnect request failed: {e}")

        return []
