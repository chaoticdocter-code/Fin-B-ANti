"""Logging setup.

Two sinks: a colourised console sink for the supervised session, and a rotating
JSONL file sink so a later run (or an agent) can parse exactly what happened.
Trading systems are forensic by nature — the file sink is not optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False


def setup_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure loguru. Idempotent — safe to call from any entry point."""
    global _configured
    if _configured:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<dim>{time:HH:mm:ss}</dim> "
            "<level>{level: <7}</level> "
            "<cyan>{extra[ctx]}</cyan> "
            "<level>{message}</level>"
        ),
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "finb-{time:YYYY-MM-DD}.jsonl",
            level="DEBUG",
            serialize=True,
            rotation="00:00",
            retention="90 days",
            enqueue=True,
        )

    logger.configure(extra={"ctx": "-"})
    _configured = True


def get_logger(ctx: str):
    """A logger bound to a short context tag, e.g. ``get_logger("collector")``."""
    return logger.bind(ctx=ctx)
