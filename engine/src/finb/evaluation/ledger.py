"""A permanent record of every strategy variant the bot has ever tried.

This exists because of one specific failure mode. The bot evolves across many
sessions. In session 40 a variant posts a strong month and it is tempting to
call it skill. But if sessions 1-39 quietly tried 800 variants, that strong month
is the best of 800 draws, and the honest hurdle is far higher than it looks.

Nobody remembers 800 variants. So the ledger remembers, it only ever grows, and
it survives restarts. Deleting it does not reset the statistics — it just
destroys the evidence of how hard you searched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Trial:
    """One entry in the search history.

    Three kinds, and all three count toward the trial total:

    - ``variant``     — a strategy configuration that was evaluated.
    - ``observation`` — a session in which a human or agent looked at live P&L.
      Looking is a test. You cannot un-see an equity curve, and every decision
      afterwards is conditioned on it.
    - ``hypothesis``  — a proposed idea ("try a volatility filter"), whether or
      not it was ever coded. Proposing is where the selection happens; the code
      is just the execution of a choice already made.

    Counting only ``variant`` rows is the standard way to undercount the trial
    total by one to two orders of magnitude and make every deflated Sharpe in
    the project decorative.
    """

    variant_id: str
    created_at: str
    sharpe: float
    """Per-period (NOT annualised) Sharpe, so it composes with the gate.
    Zero and meaningless for observation and hypothesis rows."""

    n_observations: int
    genome_hash: str
    note: str = ""
    kept: bool = False
    kind: str = "variant"


class SearchLedger:
    """Append-only JSONL log of evaluated variants.

    JSONL rather than a single JSON document so a crash mid-write costs one line
    instead of the whole history.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._trials: list[Trial] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._trials.append(Trial(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                # A torn final line from an interrupted write. Skip it rather
                # than lose the entire search history.
                continue

    # ------------------------------------------------------------------ #

    def record(
        self,
        variant_id: str,
        *,
        returns: np.ndarray | None = None,
        sharpe: float | None = None,
        n_observations: int | None = None,
        genome_hash: str = "",
        note: str = "",
        kept: bool = False,
    ) -> Trial:
        """Log a variant. Pass either `returns` or (`sharpe`, `n_observations`)."""
        from finb.evaluation.gate import sharpe_ratio

        if returns is not None:
            r = np.asarray(returns, dtype=float)
            r = r[np.isfinite(r)]
            sharpe = sharpe_ratio(r)
            n_observations = int(r.size)
        if sharpe is None or n_observations is None:
            raise ValueError("provide either returns=, or both sharpe= and n_observations=")

        return self._append(
            Trial(
                variant_id=variant_id,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                sharpe=float(sharpe),
                n_observations=int(n_observations),
                genome_hash=genome_hash,
                note=note,
                kept=kept,
                kind="variant",
            )
        )

    def record_observation(self, session_id: str, note: str = "") -> Trial:
        """Log that live P&L was looked at during a session.

        Call this every time results are viewed. It is deliberately cheap to
        call and deliberately impossible to avoid paying for: each observation
        raises the luck hurdle a little, because each one is a decision point
        that could have gone another way.
        """
        return self._append(
            Trial(
                variant_id=f"session:{session_id}",
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                sharpe=0.0,
                n_observations=0,
                genome_hash="",
                note=note,
                kind="observation",
            )
        )

    def record_hypothesis(self, description: str, proposed_by: str = "agent") -> Trial:
        """Log a proposed idea, whether or not it was ever implemented."""
        return self._append(
            Trial(
                variant_id=f"hypothesis:{proposed_by}",
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                sharpe=0.0,
                n_observations=0,
                genome_hash="",
                note=description,
                kind="hypothesis",
            )
        )

    def _append(self, t: Trial) -> Trial:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(asdict(t)) + "\n")
        self._trials.append(t)
        return t

    # ------------------------------------------------------------------ #

    @property
    def trials(self) -> list[Trial]:
        return list(self._trials)

    @property
    def variants(self) -> list[Trial]:
        """Only the rows that carry a measurable Sharpe."""
        return [t for t in self._trials if t.kind == "variant"]

    @property
    def n_trials(self) -> int:
        """Every trial of any kind — variants, observations, and hypotheses.

        This is what the gate must be charged for. It is intentionally larger
        than the number of strategies you think you tried.
        """
        return len(self._trials)

    def count_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self._trials:
            out[t.kind] = out.get(t.kind, 0) + 1
        return out

    def sr_variance(self, min_trials: int = 10) -> float:
        """Dispersion of Sharpe across the search — the scale of the luck hurdle.

        Measured over variants only, since observations and hypotheses carry no
        Sharpe. Falls back to the theoretical null variance (1/T) when too few
        variants exist to estimate it, because an empirical variance from three
        samples would understate the hurdle and wave through noise.
        """
        variants = self.variants
        if len(variants) < min_trials:
            n = max((t.n_observations for t in variants), default=0)
            return 1.0 / n if n > 1 else 0.0
        return float(np.var([t.sharpe for t in variants], ddof=1))

    def best(self) -> Trial | None:
        return max(self.variants, key=lambda t: t.sharpe, default=None)

    def summary(self) -> str:
        if not self._trials:
            return "nothing recorded yet"
        counts = self.count_by_kind()
        parts = ", ".join(f"{v} {k}{'s' if v != 1 else ''}" for k, v in sorted(counts.items()))
        head = f"{self.n_trials} trials since {self._trials[0].created_at[:10]} ({parts})"
        b = self.best()
        if b is None:
            return head
        return f"{head}; best per-period Sharpe {b.sharpe:.4f} ({b.variant_id})"
