"""Autonomous Bot Daemon & Resilient Execution Loop.

Schedules and executes periodic trading decision cycles with exponential
backoff retry handling, health logging, and portfolio guard enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from finb.bot import run_bot
from finb.config import Settings, get_settings
from finb.log import get_logger
from finb.risk.equity_curve_guard import EquityCurveGuard

log = get_logger("daemon")


@dataclass
class DaemonHealth:
    cycles_completed: int = 0
    errors_encountered: int = 0
    last_cycle_time: datetime | None = None
    last_error: str = ""


class BotDaemon:
    """Autonomous trading daemon for periodic rebalance cycles."""

    def __init__(
        self,
        settings: Settings | None = None,
        cycle_interval_seconds: float = 3600.0,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.cycle_interval_seconds = cycle_interval_seconds
        self.max_retries = max_retries
        self.guard = EquityCurveGuard(initial_capital=self.settings.finb_capital_usd)
        self.health = DaemonHealth()

    def run_single_cycle(self, live: bool = True, top_n: int = 2) -> bool:
        """Run a single execution cycle with retry backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(f"Starting daemon cycle (attempt {attempt}/{self.max_retries})...")
                run = run_bot(self.settings, live=live, top_n=top_n)

                self.health.cycles_completed += 1
                self.health.last_cycle_time = datetime.now(UTC)

                log.info(
                    f"Daemon cycle complete: {run.orders_sent} orders sent, "
                    f"{run.orders_blocked} orders blocked."
                )
                return True
            except Exception as e:  # noqa: BLE001
                self.health.errors_encountered += 1
                self.health.last_error = f"{type(e).__name__}: {e}"
                log.warning(f"Daemon cycle attempt {attempt} failed: {self.health.last_error}")
                if attempt < self.max_retries:
                    backoff = 2 ** attempt
                    time.sleep(backoff)

        return False
