"""The daily session.

One command that does the things which must happen every day and cannot be
recovered if skipped, then writes what it found into the vault.

Order matters here. The universe snapshot runs **first**, before anything can
fail, because it is the only step whose omission is permanent — a missed day of
asset listings cannot be reconstructed from any source at any price.

The session also opens by recording an observation in the
[[Search Ledger]], per `0009`. Looking at results is a trial; the ledger is
charged for it whether or not anything is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from finb.config import Settings
from finb.data.lake import BarLake, Timeframe
from finb.data.universe import UniverseArchive
from finb.evaluation.ledger import SearchLedger
from finb.log import get_logger
from finb.sim.constraints import AssetClass
from finb.vault.notes import write_note

log = get_logger("session")

STABLECOINS = {"USDC", "USDT", "USDG", "DAI", "PYUSD", "BUSD", "TUSD", "USDP"}


@dataclass
class SessionReport:
    started_at: datetime
    steps: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    universe: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    account: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    risk: str = ""

    def ok(self, step: str, detail: str) -> None:
        self.steps.append((step, detail))
        log.info(f"{step}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning(message)


def run_session(
    settings: Settings,
    *,
    refresh_data: bool = True,
    max_symbols: int = 40,
) -> SessionReport:
    """Run the daily maintenance session. Reads and records; places no orders."""
    report = SessionReport(started_at=datetime.now(UTC))
    lake = BarLake(settings.finb_data_dir)
    archive = UniverseArchive(settings.finb_data_dir)
    ledger = SearchLedger(settings.finb_data_dir / "artifacts" / "search_ledger.jsonl")

    # --- 0. charge for the session itself --------------------------------
    day = report.started_at.date().isoformat()
    ledger.record_observation(day, note="daily session")
    report.ledger = {
        "trials": ledger.n_trials,
        "by_kind": ledger.count_by_kind(),
        "summary": ledger.summary(),
    }

    # --- 1. universe snapshot: irrecoverable if skipped ------------------
    try:
        from finb.data.universe import fetch_alpaca_universe

        if archive.dates() and archive.dates()[-1] == report.started_at.date():
            report.ok("universe", "already captured today")
        else:
            df = fetch_alpaca_universe(settings)
            archive.write(df)
            report.ok("universe", f"{df.height} assets captured")
    except Exception as e:  # noqa: BLE001
        report.warn(f"universe snapshot FAILED ({type(e).__name__}) — this day is unrecoverable")

    cov = archive.coverage()
    report.universe = dict(cov)
    if cov["snapshots"] and cov["gap_days"] and cov["gap_days"] > 3:
        report.warn(f"universe archive is {cov['gap_days']} days stale")

    # --- 2. tracked symbols ----------------------------------------------
    try:
        tradable = archive.tradable_symbols(report.started_at.date(), "crypto")
        symbols = [
            s for s in tradable
            if s.endswith("/USD") and s.split("/")[0] not in STABLECOINS
        ][:max_symbols]
    except Exception:  # noqa: BLE001
        symbols = lake.symbols(Timeframe.D1, AssetClass.CRYPTO)[:max_symbols]
    report.ok("universe", f"{len(symbols)} tracked crypto symbols")

    # --- 3. incremental data refresh --------------------------------------
    fetched = updated = 0
    if refresh_data:
        from finb.data.sources.alpaca import fetch_crypto_bars

        for sym in symbols:
            try:
                coverage = lake.coverage(sym, Timeframe.D1, AssetClass.CRYPTO)
                start = (
                    coverage[1] - timedelta(days=3)   # small overlap; writes dedupe
                    if coverage
                    else datetime(2021, 1, 1, tzinfo=UTC)
                )
                if coverage and (report.started_at - coverage[1]) < timedelta(hours=20):
                    continue
                df = fetch_crypto_bars(sym, Timeframe.D1, start)
                if df.height:
                    lake.write(sym, Timeframe.D1, df, asset=AssetClass.CRYPTO)
                    updated += 1
                fetched += df.height
            except Exception as e:  # noqa: BLE001
                report.warn(f"refresh failed for {sym}: {type(e).__name__}")
        report.ok("data", f"{updated} symbols updated, {fetched} bars fetched")

    # --- 4. data health ---------------------------------------------------
    stored = lake.symbols(Timeframe.D1, AssetClass.CRYPTO)
    end = report.started_at
    start = end - timedelta(days=120)
    health = []
    for sym in stored:
        c = lake.coverage(sym, Timeframe.D1, AssetClass.CRYPTO)
        if not c:
            continue
        completeness = lake.completeness(sym, Timeframe.D1, start, end, AssetClass.CRYPTO)
        stale_days = (end - c[1]).days
        health.append({"symbol": sym, "completeness": completeness, "stale_days": stale_days})
        if completeness < 0.95:
            report.warn(f"{sym} is {completeness:.0%} complete over the last 120 days")
        if stale_days > 3:
            report.warn(f"{sym} last bar is {stale_days} days old")

    report.data = {
        "symbols": len(stored),
        "health": sorted(health, key=lambda h: h["completeness"]),
        "mean_completeness": (
            sum(h["completeness"] for h in health) / len(health) if health else 0.0
        ),
    }
    report.ok("data", f"{len(stored)} symbols stored, "
                      f"{report.data['mean_completeness']:.1%} mean completeness")

    # --- 5. account + risk ------------------------------------------------
    try:
        from finb.execution.alpaca_paper import AlpacaBroker
        from finb.risk import RiskEngine, RiskLimits

        broker = AlpacaBroker(
            settings,
            RiskEngine(RiskLimits(capital=settings.finb_capital_usd)),
            dry_run=True,
        )
        snap = broker.account()
        budget = broker.budget(snap.equity)
        broker.risk.update(snap.taken_at, budget)

        report.account = {
            "is_paper": snap.is_paper,
            "broker_equity": snap.equity,
            "our_budget": budget,
            "positions": len(snap.positions),
            "gross_exposure": snap.gross_exposure,
        }
        report.risk = broker.risk.status()
        report.ok("account", f"paper={snap.is_paper}, budget ${budget:,.2f}, "
                             f"{len(snap.positions)} positions")
    except Exception as e:  # noqa: BLE001
        report.warn(f"account read failed: {type(e).__name__}: {e}")

    return report


def write_session_note(settings: Settings, report: SessionReport) -> str:
    """Publish the session into the vault, preserving anything hand-written."""
    day = report.started_at.date().isoformat()
    path = settings.finb_vault_dir / "40-Sessions" / f"{day} Session.md"

    worst = report.data.get("health", [])[:5]
    health_rows = "\n".join(
        f"| {h['symbol']} | {h['completeness']:.1%} | {h['stale_days']}d |" for h in worst
    ) or "| — | — | — |"

    acct = report.account
    warn_block = (
        "\n".join(f"- ⚠️ {w}" for w in report.warnings)
        if report.warnings
        else "None."
    )

    managed = f"""## Status

| | |
|---|---|
| Mode | {"PAPER" if acct.get("is_paper", True) else "**LIVE**"} |
| Our budget | ${acct.get("our_budget", 0):,.2f} |
| Broker equity | ${acct.get("broker_equity", 0):,.2f} |
| Open positions | {acct.get("positions", 0)} |
| Risk | {report.risk or "n/a"} |

## Search ledger

**{report.ledger.get("trials", 0)} trials** — {report.ledger.get("by_kind", {})}

Every session counts as one observation. The luck hurdle rises with it, which is
the price of looking.

## Data

{report.data.get("symbols", 0)} symbols stored, mean completeness
{report.data.get("mean_completeness", 0):.1%} over the last 120 days.

| Symbol | Complete | Stale |
|---|---|---|
{health_rows}

## Universe archive

{report.universe.get("snapshots", 0)} snapshots, {report.universe.get("first")} → {report.universe.get("last")}

## Warnings

{warn_block}

## Steps

{chr(10).join(f"- **{s}** — {d}" for s, d in report.steps)}
"""

    write_note(
        path,
        frontmatter={
            "type": "session",
            "date": day,
            "trials": report.ledger.get("trials", 0),
            "budget_usd": round(acct.get("our_budget", 0), 2),
            "positions": acct.get("positions", 0),
            "data_completeness": round(report.data.get("mean_completeness", 0), 4),
            "warnings": len(report.warnings),
        },
        managed=managed,
        initial_body=f"# {day} — Session\n\n## My notes\n\n",
    )
    return str(path)
