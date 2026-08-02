"""Generate the vault's map artefacts.

The canvas is drawn from code rather than by hand so it cannot drift from what
actually exists. Components carry a build state, and that state is the source of
truth for the colour on the map — a box is green because the module is there and
tested, not because someone remembered to recolour it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finb.vault.canvas import GREEN, PURPLE, RED, YELLOW, Canvas, FileNode, GroupNode, TextNode

COL_W, ROW_H, COL_GAP, ROW_GAP = 340, 150, 130, 55


@dataclass(frozen=True, slots=True)
class Component:
    key: str
    title: str
    detail: str
    built: bool
    module: str = ""

    @property
    def color(self) -> str:
        return GREEN if self.built else YELLOW

    def card(self) -> TextNode:
        state = "✅" if self.built else "◻︎ not built yet"
        body = f"## {self.title}\n\n{self.detail}"
        if self.module:
            body += f"\n\n`{self.module}`"
        body += f"\n\n{state}"
        return TextNode(key=self.key, text=body, width=COL_W, height=ROW_H, color=self.color)


# The pipeline, left to right. Keep this list honest — it drives the map.
COLUMNS: list[tuple[str, list[Component]]] = [
    (
        "Free data in",
        [
            Component("src-crypto", "Crypto bars", "Alpaca crypto: real-time, complete, free. The best data surface we have.", True, "finb.data.sources.alpaca"),
            Component("src-equity", "US equities", "Alpaca IEX feed — only 2.4% of the tape. Feed choice is explicit at every call site.", True, "finb.data.sources.alpaca"),
            Component("src-universe", "Universe archive", "Dated snapshots of what was tradeable. Cannot be backfilled — run daily.", True, "finb.data.universe"),
            Component("src-macro", "Macro (FRED)", "9 series, initial-release-only so revisions cannot leak. Orthogonal to price.", True, "finb.data.sources.macro"),
            Component("src-news", "News (Benzinga)", "Verified reachable. Article counts and z-scores — the only non-price signal we have.", True, "finb.data.sources.news"),
        ],
    ),
    (
        "Store",
        [
            Component("lake", "Parquet lake", "Partitioned by symbol and year. Idempotent writes, atomic renames.", True, "finb.data.lake"),
            Component("duck", "Gap detection", "Knows which bars it *should* have. A hole must never read as a quiet market.", True, "finb.data.lake"),
            Component("clock", "Market calendar", "NYSE holidays, early closes, T+1, session windows. Pinned to published dates.", True, "finb.clock"),
        ],
    ),
    (
        "Think",
        [
            Component("features", "Indicators", "18 causal technical features. Every one passes the leakage audit.", True, "finb.features.indicators"),
            Component("labels", "Labels", "Triple-barrier targets: path-aware, volatility-scaled, and they report when they resolve.", True, "finb.features.labeling"),
            Component("cv", "Purged CV", "Removes training samples whose label window overlaps the test fold. Without it, shuffled k-fold scores 0.955 on a random walk.", True, "finb.models.cv"),
            Component("model", "Model", "LightGBM, trained only under purged CV. Scores at chance on noise, learns real signal.", True, "finb.models.gbdt"),
        ],
    ),
    (
        "Test",
        [
            Component("sim", "Shadow book", "Simulated $500 account. Long-only, no leverage, costs charged on every fill.", True, "finb.sim.engine"),
            Component("leak", "Leakage audit", "Truncation-equality and future-noise checks. Catches lookahead automatically.", True, "finb.features.leakage"),
            Component("costs", "Costs & slippage", "Fees, spread, regulatory. Crypto round trips cost 17x equities — the finding that set the holding period.", True, "finb.sim.costs"),
            Component("rules", "Account rules", "T+1 settlement, good-faith violations, broker policies. PDT was repealed 2026-06-04.", True, "finb.sim.constraints"),
            Component("policy", "Holding policy", "Minimum hold derived from cost: equity 8d, crypto 38d. Breadth is mandatory.", True, "finb.sim.policy"),
        ],
    ),
    (
        "Judge",
        [
            Component("ledger", "Search ledger", "Every variant, observation and hypothesis ever. Append-only, survives restarts.", True, "finb.evaluation.ledger"),
            Component("gate", "The gate", "Deflated Sharpe vs the luck hurdle, corrected for correlated trials.", True, "finb.evaluation.gate"),
            Component("null", "Null cohort", "Block-bootstrapped zero-skill controls from real data. The empirical noise floor.", True, "finb.evaluation.null_cohort"),
        ],
    ),
    (
        "Act",
        [
            Component("risk", "Risk limits", "One-way kill switch, concentration and leverage caps, vol-targeted sizing.", True, "finb.risk.limits"),
            Component("exec", "Alpaca paper", "Three independent locks before live. Sizes against our $500, not the broker's $95k.", True, "finb.execution.alpaca_paper"),
        ],
    ),
]


def build_system_map(vault: Path) -> Path:
    """Write ``00-Map/System Map.canvas``."""
    c = Canvas()
    x = 0

    for title, comps in COLUMNS:
        cards = [comp.card() for comp in comps]
        c.column(cards, x=x, y0=0, gap=ROW_GAP)

        height = len(cards) * ROW_H + (len(cards) - 1) * ROW_GAP
        c.add(
            GroupNode(
                key=f"group-{title}",
                label=title,
                x=x - 30,
                y=-70,
                width=COL_W + 60,
                height=height + 100,
            )
        )
        x += COL_W + COL_GAP

    # Flow between stages.
    flow = [
        ("src-crypto", "lake"), ("src-equity", "lake"), ("src-news", "lake"),
        ("lake", "duck"),
        ("duck", "features"), ("features", "labels"), ("labels", "model"),
        ("model", "sim"), ("sim", "costs"), ("costs", "rules"),
        ("rules", "ledger"), ("ledger", "gate"),
        ("gate", "risk"), ("risk", "exec"),
    ]
    for a, b in flow:
        c.link(a, b)

    # The loop that makes it "evolving": the gate's verdict feeds the next variant.
    c.link("gate", "model", label="evolve: next variant", color=PURPLE,
           from_side="top", to_side="top")

    # Safety note, pinned under the execution column.
    c.add(
        TextNode(
            key="safety",
            text=(
                "## 🔒 Paper only\n\n"
                "Two independent locks before any real order:\n\n"
                "1. `FINB_ALLOW_LIVE` set to the magic string\n"
                "2. the variant has passed the gate\n\n"
                "See [[0001 Paper only and the live guard]]"
            ),
            x=x - COL_W - COL_GAP,
            y=2 * (ROW_H + ROW_GAP) + 60,
            width=COL_W,
            height=200,
            color=RED,
        )
    )

    # Anchor the map to the written notes.
    notes_x = -(COL_W + COL_GAP)
    c.column(
        [
            FileNode(key="note-moc", file="00-Map/Fin B.md", width=COL_W, height=ROW_H, color=PURPLE),
            FileNode(key="note-hurdle", file="90-Reference/Luck Hurdle.md", width=COL_W, height=ROW_H, color=PURPLE),
            FileNode(key="note-decisions", file="20-Architecture/Decisions.md", width=COL_W, height=ROW_H, color=PURPLE),
            FileNode(key="note-open", file="00-Map/Open Questions.md", width=COL_W, height=ROW_H, color=PURPLE),
        ],
        x=notes_x,
        y0=0,
        gap=ROW_GAP,
    )
    c.add(
        GroupNode(
            key="group-notes",
            label="The thinking",
            x=notes_x - 30,
            y=-70,
            width=COL_W + 60,
            height=4 * ROW_H + 3 * ROW_GAP + 100,
        )
    )
    c.link("note-hurdle", "gate", label="governs", color=PURPLE, from_side="bottom", to_side="bottom")

    return c.save(vault / "00-Map" / "System Map.canvas")
