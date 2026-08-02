"""Session note publishing.

The session runner itself is I/O orchestration against live services, so what is
tested here is the part with a correctness contract: the note it writes must
never destroy anything the operator typed into it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from finb.config import Settings
from finb.session import SessionReport, write_session_note
from finb.vault.notes import read_frontmatter


def report(**kw) -> SessionReport:
    r = SessionReport(started_at=datetime(2026, 8, 2, 15, 30, tzinfo=UTC))
    r.ok("universe", "14,235 assets captured")
    r.ok("data", "36 symbols stored")
    r.ledger = {"trials": 21, "by_kind": {"variant": 18, "observation": 3}}
    r.account = {
        "is_paper": True, "broker_equity": 95_468.36, "our_budget": 500.0,
        "positions": 0, "gross_exposure": 0.0,
    }
    r.risk = "ok — equity $500.00, drawdown 0.0%"
    r.data = {
        "symbols": 36,
        "mean_completeness": 0.986,
        "health": [{"symbol": "USDG/USD", "completeness": 0.55, "stale_days": 0}],
    }
    r.universe = {"snapshots": 2, "first": "2026-08-01", "last": "2026-08-02"}
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def settings(tmp_path) -> Settings:
    (tmp_path / "40-Sessions").mkdir(parents=True, exist_ok=True)
    return Settings(finb_vault_dir=tmp_path, finb_data_dir=tmp_path / "data")


def test_session_note_carries_the_numbers_in_frontmatter(tmp_path):
    s = settings(tmp_path)
    path = write_session_note(s, report())

    fm = read_frontmatter(__import__("pathlib").Path(path))
    assert fm["type"] == "session"
    assert fm["date"] == "2026-08-02"
    assert fm["trials"] == 21
    assert fm["budget_usd"] == 500.0
    assert fm["data_completeness"] == 0.986


def test_the_budget_not_the_broker_balance_is_what_reads_as_ours(tmp_path):
    s = settings(tmp_path)
    text = __import__("pathlib").Path(write_session_note(s, report())).read_text(encoding="utf-8")
    assert "$500.00" in text
    assert "$95,468.36" in text          # shown for context...
    assert "Our budget | $500.00" in text  # ...but clearly not the budget


def test_warnings_are_surfaced(tmp_path):
    s = settings(tmp_path)
    r = report()
    r.warn("USDG/USD is 55% complete over the last 120 days")
    text = __import__("pathlib").Path(write_session_note(s, r)).read_text(encoding="utf-8")
    assert "55%" in text


def test_no_warnings_reads_cleanly(tmp_path):
    s = settings(tmp_path)
    text = __import__("pathlib").Path(write_session_note(s, report())).read_text(encoding="utf-8")
    assert "None." in text


def test_rerunning_a_session_preserves_operator_notes(tmp_path):
    """The whole point of the managed-region pattern."""
    from pathlib import Path

    s = settings(tmp_path)
    path = Path(write_session_note(s, report()))

    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "## My notes\n", "## My notes\n\nUSDG gap looks like a listing date, not a bug.\n"
    )
    path.write_text(text, encoding="utf-8")

    # Session runs again later the same day with new numbers.
    r2 = report()
    r2.ledger = {"trials": 25, "by_kind": {"variant": 21, "observation": 4}}
    write_session_note(s, r2)

    out = path.read_text(encoding="utf-8")
    assert "USDG gap looks like a listing date, not a bug." in out   # survived
    assert "25 trials" in out                                        # updated
    assert read_frontmatter(path)["trials"] == 25
