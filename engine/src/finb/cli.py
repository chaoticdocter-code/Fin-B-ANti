"""Command-line entry point: ``uv run finb ...``"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finb.config import LIVE_MAGIC, detect_providers, get_settings
from finb.log import setup_logging

app = typer.Typer(
    name="finb",
    help="Fin B — a farm of competing, continuously-evolved trading strategies. Paper only.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def doctor() -> None:
    """Check the install: paths, credentials, and which providers are usable."""
    s = get_settings()
    setup_logging(s.finb_log_dir)

    # --- safety posture, reported first because it matters most -------------
    if s.live_enabled:
        console.print(
            Panel(
                "[bold red]LIVE TRADING IS ENABLED.[/bold red]\n"
                "FINB_ALLOW_LIVE is set to the magic string. Real money is at risk.\n"
                "Set FINB_ALLOW_LIVE=no in your .env to disable.",
                border_style="red",
                title="DANGER",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]PAPER MODE[/bold green] — no code path can place a real order.\n"
                f"Simulated capital: [bold]${s.finb_capital_usd:,.2f}[/bold]\n"
                f"(Live would require FINB_ALLOW_LIVE={LIVE_MAGIC})",
                border_style="green",
                title="Safety",
            )
        )

    # --- paths --------------------------------------------------------------
    paths = Table(title="Paths", show_header=True, header_style="bold")
    paths.add_column("What")
    paths.add_column("Where")
    paths.add_column("Exists")
    for label, p in [
        ("vault", s.finb_vault_dir),
        ("data", s.finb_data_dir),
        ("logs", s.finb_log_dir),
        (".env", s.finb_vault_dir / ".env"),
    ]:
        ok = "[green]yes[/green]" if p.exists() else "[yellow]no[/yellow]"
        paths.add_row(label, str(p), ok)
    console.print(paths)

    # --- providers ----------------------------------------------------------
    provs = detect_providers(s)
    tbl = Table(title="Providers", show_header=True, header_style="bold")
    tbl.add_column("Provider")
    tbl.add_column("Kind")
    tbl.add_column("Status")
    tbl.add_column("Note", overflow="fold")

    for p in sorted(provs, key=lambda x: (not x.configured, x.kind, x.name)):
        if p.configured and not p.needs_key:
            status = "[green]ready (keyless)[/green]"
        elif p.configured:
            status = "[green]ready[/green]"
        else:
            status = "[dim]no key[/dim]"
        tbl.add_row(p.name, p.kind, status, p.note)
    console.print(tbl)

    ready = [p for p in provs if p.configured]
    console.print(
        f"\n[bold]{len(ready)}[/bold] of {len(provs)} providers usable "
        f"({sum(1 for p in ready if not p.needs_key)} need no credentials at all)."
    )

    if not (s.finb_vault_dir / ".env").exists():
        console.print(
            "\n[yellow]No .env found.[/yellow] Copy [bold].env.example[/bold] to "
            "[bold].env[/bold] and fill in the services you have."
        )


@app.command()
def run(
    live: bool = False,
    top_n: int = 4,
    lookback: int = 60,
) -> None:
    """Run one bot cycle. Dry run unless --live is passed (paper account either way)."""
    from finb.bot import run_bot

    s = get_settings()
    setup_logging(s.finb_log_dir)

    console.print(
        Panel(
            "The strategy wired in here is cross-sectional momentum, which the gate\n"
            "[bold]rejected[/bold] (DSR 0.132; 23% of zero-skill strategies beat it).\n"
            "This verifies the plumbing, not an edge. Any P&L is noise.",
            title="What this run means",
            border_style="yellow",
        )
    )

    r = run_bot(s, dry_run=not live, top_n=top_n, lookback=lookback)

    mode = "[bold red]SENDING ORDERS[/bold red]" if live else "[green]DRY RUN[/green]"
    console.print(
        f"\n{mode}  ·  budget [bold]${r.budget:,.2f}[/bold] of ${r.equity:,.2f}  ·  "
        f"{r.universe} symbols  ·  target ${r.budget / top_n:,.2f} each\n"
    )

    if r.ranked:
        rank = Table(title=f"Momentum ranking ({lookback}d, skip 7)", header_style="bold")
        rank.add_column("#", justify="right")
        rank.add_column("Symbol")
        rank.add_column("Score", justify="right")
        rank.add_column("", justify="left")
        for i, (sym, score) in enumerate(r.ranked[:10], 1):
            rank.add_row(
                str(i), sym, f"{score:+.1%}",
                "[green]target[/green]" if i <= top_n else "",
            )
        console.print(rank)

    if r.decisions:
        tbl = Table(title="Decisions", header_style="bold")
        tbl.add_column("Symbol")
        tbl.add_column("Action")
        tbl.add_column("Now", justify="right")
        tbl.add_column("Target", justify="right")
        tbl.add_column("Sent", justify="right")
        tbl.add_column("Detail", overflow="fold")
        colour = {"buy": "green", "sell": "cyan", "hold": "dim", "blocked": "yellow"}
        for d in r.decisions:
            c = colour.get(d.action, "white")
            tbl.add_row(
                d.symbol, f"[{c}]{d.action}[/{c}]",
                f"${d.current_value:,.2f}", f"${d.target_value:,.2f}",
                f"${d.sent_value:,.2f}" if d.sent_value else "—",
                d.reason,
            )
        console.print(tbl)

    console.print(
        f"\norders sent [bold]{r.orders_sent}[/bold] · blocked {r.orders_blocked}"
    )
    for w in r.warnings:
        console.print(f"[yellow]  • {w}[/yellow]")
    if not live:
        console.print("\n[dim]Nothing was sent. Re-run with --live to place paper orders.[/dim]")


@app.command()
def watch(
    interval: int = 60,
    minutes: int = 60,
    top_n: int = 4,
    lookback: int = 60,
) -> None:
    """Monitor the book every `interval` seconds. Never places an order."""
    from finb.watch import watch as run_watch

    s = get_settings()
    setup_logging(s.finb_log_dir)

    console.print(
        Panel(
            f"Scanning every [bold]{interval}s[/bold] for [bold]{minutes} min[/bold].\n"
            "[bold]No orders are placed here.[/bold] Trading hourly would cost 53bps a "
            "round trip against 3.75bps of expected edge —\n"
            "a $500 book halves in about five sessions. Use [bold]finb run --live[/bold] "
            "when the holding policy releases capital.",
            title="Watch",
            border_style="cyan",
        )
    )

    def render(scan, n: int) -> None:
        stamp = scan.at.strftime("%H:%M:%S")
        pnl = scan.unrealised
        colour = "green" if pnl >= 0 else "red"
        head = (
            f"[dim]{stamp}[/dim]  scan {n:>3}  "
            f"exposure ${scan.gross_exposure:>7,.2f}  "
            f"unrealised [{colour}]${pnl:+.2f}[/{colour}]"
        )
        if scan.ranking_changed:
            head += "  [yellow]RANKING CHANGED[/yellow]"
        console.print(head)

        for p in scan.positions:
            mark = "[green]unlocked[/green]" if p["unlocked"] else f"locked {p['locked_for']}"
            c = "green" if p["unrealised"] >= 0 else "red"
            stale = "" if p.get("live_mark") else " [yellow](broker mark)[/yellow]"
            console.print(
                f"    {p['symbol']:<9} ${p['value']:>7,.2f}  "
                f"[{c}]{p['pct']:+6.2%}[/{c}]  {mark}{stale}"
            )
        if scan.ranking:
            top = "  ".join(f"{s}({v:+.1%})" for s, v in scan.ranking[:top_n])
            console.print(f"    [dim]top: {top}[/dim]")
        for a in scan.alerts:
            console.print(f"    [yellow]! {a}[/yellow]")

    scans = run_watch(
        s, interval=interval, duration=minutes * 60,
        top_n=top_n, lookback=lookback, on_scan=render,
    )

    changes = sum(1 for x in scans if x.ranking_changed)
    unlocked = sorted({sym for x in scans for sym in x.unlocked})
    console.print(
        f"\n[bold]{len(scans)}[/bold] scans · ranking changed {changes}x · "
        f"final unrealised ${scans[-1].unrealised:+.2f}" if scans else "no scans"
    )
    if unlocked:
        console.print(
            f"[green]Holding period expired for: {', '.join(unlocked)}[/green] — "
            "`finb run --live` can now rebalance these."
        )
    else:
        console.print("[dim]No position has cleared its minimum hold yet.[/dim]")


@app.command()
def verify(keyless: bool = True) -> None:
    """Actually call each provider to see which credentials work. Read-only."""
    from finb.data.verify import verify_all

    s = get_settings()
    setup_logging(s.finb_log_dir)
    console.print("[bold]Probing providers...[/bold] (read-only calls)\n")

    results = verify_all(s, include_keyless=keyless)

    colour = {
        "ok": "green",
        "auth_failed": "bold red",
        "rate_limited": "yellow",
        "unreachable": "yellow",
        "no_key": "dim",
        "error": "red",
    }

    tbl = Table(header_style="bold")
    tbl.add_column("Provider")
    tbl.add_column("Kind")
    tbl.add_column("Status")
    tbl.add_column("ms", justify="right")
    tbl.add_column("Detail", overflow="fold")

    order = {"ok": 0, "auth_failed": 1, "error": 2, "rate_limited": 3, "unreachable": 4, "no_key": 5}
    for r in sorted(results, key=lambda x: (order[x.status], x.provider)):
        tbl.add_row(
            r.provider,
            r.kind,
            f"[{colour[r.status]}]{r.symbol}[/{colour[r.status]}]",
            f"{r.latency_ms:.0f}" if r.latency_ms else "",
            r.detail,
        )
    console.print(tbl)

    working = [r for r in results if r.status == "ok"]
    broken = [r for r in results if r.status in ("auth_failed", "error")]
    absent = [r for r in results if r.status == "no_key"]

    console.print(
        f"\n[bold green]{len(working)} working[/bold green] · "
        f"[bold red]{len(broken)} broken[/bold red] · "
        f"[dim]{len(absent)} not configured[/dim]"
    )
    for r in broken:
        console.print(f"  [red]{r.provider}[/red] — {r.detail}")


@app.command()
def session(refresh: bool = True) -> None:
    """Run the daily session: snapshot, refresh, health-check, and log it."""
    from finb.session import run_session, write_session_note

    s = get_settings()
    setup_logging(s.finb_log_dir)

    console.print("[bold]Running daily session...[/bold]\n")
    report = run_session(s, refresh_data=refresh)

    tbl = Table(title="Session", header_style="bold")
    tbl.add_column("Step")
    tbl.add_column("Result", overflow="fold")
    for step, detail in report.steps:
        tbl.add_row(step, detail)
    console.print(tbl)

    acct = report.account
    console.print(
        Panel(
            f"budget [bold]${acct.get('our_budget', 0):,.2f}[/bold] of "
            f"${acct.get('broker_equity', 0):,.2f} broker equity   ·   "
            f"{acct.get('positions', 0)} positions\n"
            f"ledger: [bold]{report.ledger.get('trials', 0)}[/bold] trials "
            f"{report.ledger.get('by_kind', {})}\n"
            f"risk: {report.risk or 'n/a'}",
            title="PAPER" if acct.get("is_paper", True) else "LIVE",
            border_style="green" if acct.get("is_paper", True) else "red",
        )
    )

    if report.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for w in report.warnings:
            console.print(f"  • {w}")

    path = write_session_note(s, report)
    console.print(f"\nwrote [bold]{path}[/bold]")


@app.command()
def account() -> None:
    """Read the Alpaca paper account. Places nothing."""
    from finb.execution.alpaca_paper import AlpacaBroker
    from finb.risk import RiskEngine, RiskLimits

    s = get_settings()
    setup_logging(s.finb_log_dir)

    broker = AlpacaBroker(s, RiskEngine(RiskLimits(capital=s.finb_capital_usd)), dry_run=True)
    snap = broker.account()

    mode = "[green]PAPER[/green]" if snap.is_paper else "[bold red]LIVE[/bold red]"
    console.print(
        Panel(
            f"{mode}   equity [bold]${snap.equity:,.2f}[/bold]   "
            f"cash ${snap.cash:,.2f}   buying power ${snap.buying_power:,.2f}\n"
            f"Our allocation is [bold]${s.finb_capital_usd:,.2f}[/bold] — the broker's "
            f"balance is not the budget.",
            title="Alpaca account",
            border_style="green" if snap.is_paper else "red",
        )
    )

    if snap.positions:
        tbl = Table(title="Positions", header_style="bold")
        for col in ("Symbol", "Qty", "Value", "Entry", "Unrealised"):
            tbl.add_column(col, justify="right" if col != "Symbol" else "left")
        for p in snap.positions:
            tbl.add_row(
                p.symbol, f"{p.qty:.6f}", f"${p.market_value:,.2f}",
                f"${p.avg_entry_price:,.2f}", f"${p.unrealized_pl:+,.2f}",
            )
        console.print(tbl)
        console.print(f"gross exposure ${snap.gross_exposure:,.2f}")
    else:
        console.print("no open positions")

    console.print(f"\nrisk: {broker.risk.status()}")


@app.command()
def universe() -> None:
    """Snapshot today's tradeable universe. Run this daily — it cannot be backfilled."""
    from finb.data.universe import UniverseArchive, fetch_alpaca_universe

    s = get_settings()
    setup_logging(s.finb_log_dir)
    archive = UniverseArchive(s.finb_data_dir)

    df = fetch_alpaca_universe(s)
    path = archive.write(df)

    by_class = df.group_by("asset_class").len().sort("asset_class")
    tbl = Table(title="Universe snapshot", header_style="bold")
    tbl.add_column("Asset class")
    tbl.add_column("Active", justify="right")
    tbl.add_column("Tradable", justify="right")
    tbl.add_column("Fractionable", justify="right")
    for row in by_class.iter_rows(named=True):
        cls = row["asset_class"]
        sub = df.filter(pl_col_eq("asset_class", cls))
        tbl.add_row(
            cls,
            f"{row['len']:,}",
            f"{int(sub['tradable'].sum()):,}",
            f"{int(sub['fractionable'].sum()):,}",
        )
    console.print(tbl)
    console.print(f"wrote [bold]{path.name}[/bold]")

    cov = archive.coverage()
    console.print(
        f"  archive: {cov['snapshots']} snapshot(s), {cov['first']} to {cov['last']}"
    )
    if cov["snapshots"] == 1:
        console.print(
            "\n[yellow]This is the first snapshot.[/yellow] Survivorship correction "
            "only becomes possible once there are several — run this every session."
        )


def pl_col_eq(col: str, val: str):
    import polars as pl

    return pl.col(col) == val


@app.command()
def costs(notional: float = 500.0) -> None:
    """Show the breakeven hurdle per round trip at each venue."""
    from finb.sim.costs import VENUES, breakeven_table, edge_required

    tbl = Table(title=f"Round-trip cost on ${notional:,.0f}", header_style="bold")
    tbl.add_column("Venue")
    tbl.add_column("Taker", justify="right")
    tbl.add_column("Maker", justify="right")
    tbl.add_column("Move needed to break even", justify="right")

    for name, taker, maker in breakeven_table(notional):
        tbl.add_row(name, f"{taker:.1f} bps", f"{maker:.1f} bps", f"{taker / 100:.3f}%")
    console.print(tbl)

    console.print("\n[bold]Per-trade edge needed for +20%/yr[/bold] (net of costs)")
    hurdle = Table(header_style="bold")
    hurdle.add_column("Venue")
    for n in (1, 4, 20):
        hurdle.add_column(f"{n}/day", justify="right")
    for name, m in VENUES.items():
        c = m.round_trip_bps(notional)
        hurdle.add_row(name, *[f"{edge_required(c, n):.1f} bps" for n in (1, 4, 20)])
    console.print(hurdle)
    console.print(
        "\n[dim]Compare against the size of the move you expect to predict. "
        "If the hurdle is larger, no model quality rescues it.[/dim]"
    )

    # --- feasibility map: how long must we hold to cover cost? --------------
    from finb.sim.constraints import AssetClass
    from finb.sim.costs import feasibility

    vol = {AssetClass.CRYPTO: 0.035, AssetClass.EQUITY: 0.018}
    fmap = Table(
        title="\nMinimum holding period to cover costs 2x (IC=0.03)",
        header_style="bold",
    )
    fmap.add_column("Venue")
    fmap.add_column("Breakeven", justify="right")
    fmap.add_column("2x cover", justify="right")
    fmap.add_column("Trades/yr", justify="right")
    fmap.add_column("Yrs to 100", justify="right")
    fmap.add_column("Viable?", justify="center")

    for m in VENUES.values():
        f = feasibility(m, vol[m.asset], notional=notional)
        mark = "[green]yes[/green]" if f["viable"] else "[red]no[/red]"
        fmap.add_row(
            f["venue"],
            f"{f['breakeven_hold_days']:.1f}d",
            f"{f['min_hold_days']:.1f}d",
            f"{f['trades_per_year']:.0f}",
            f"{f['years_to_100_trades']:.1f}",
            mark,
        )
    console.print(fmap)
    console.print(
        "[dim]Edge grows with sqrt(time) while cost is fixed per round trip, so "
        "holding period is the only lever that improves the ratio.\n"
        "'Yrs to 100' is for ONE symbol. Trading N symbols in parallel divides it "
        "by N — which is why breadth, not frequency, is how a slow strategy earns "
        "a sample size.[/dim]"
    )


@app.command(name="map")
def vault_map() -> None:
    """Regenerate the Obsidian system map canvas from the code's real state."""
    from finb.vault.scaffold import COLUMNS, build_system_map

    s = get_settings()
    out = build_system_map(s.finb_vault_dir)

    comps = [c for _, col in COLUMNS for c in col]
    built = sum(c.built for c in comps)
    console.print(f"wrote [bold]{out.relative_to(s.finb_vault_dir)}[/bold]")
    console.print(f"  {built}/{len(comps)} components built")
    for c in comps:
        if not c.built:
            console.print(f"    [yellow]◻︎[/yellow] {c.title} [dim]{c.module}[/dim]")


@app.command()
def version() -> None:
    """Print the engine version."""
    from finb import __version__

    console.print(f"finb {__version__}")


if __name__ == "__main__":
    app()
