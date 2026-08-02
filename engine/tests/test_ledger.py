"""The ledger's only real requirement: it must not forget, and it must survive
being interrupted mid-write."""

from __future__ import annotations

import numpy as np

from finb.evaluation import PromotionGate, SearchLedger


def test_counts_every_variant_across_restarts(tmp_path):
    p = tmp_path / "ledger.jsonl"
    rng = np.random.default_rng(1)

    led = SearchLedger(p)
    for i in range(25):
        led.record(f"v{i}", returns=rng.normal(0, 0.01, 200))
    assert led.n_trials == 25

    # A new session opens the same file and inherits the full search history.
    reopened = SearchLedger(p)
    assert reopened.n_trials == 25
    reopened.record("v25", returns=rng.normal(0, 0.01, 200))
    assert SearchLedger(p).n_trials == 26


def test_torn_final_line_costs_one_trial_not_the_history(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = SearchLedger(p)
    for i in range(5):
        led.record(f"v{i}", sharpe=0.01 * i, n_observations=100)

    # Simulate a crash partway through appending a record.
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"variant_id": "v5", "created_at": "2026-07-30T')

    assert SearchLedger(p).n_trials == 5


def test_sr_variance_falls_back_to_the_null_when_evidence_is_thin(tmp_path):
    led = SearchLedger(tmp_path / "l.jsonl")
    # Three near-identical variants would imply almost zero dispersion, which
    # would wrongly collapse the luck hurdle to nothing.
    for i in range(3):
        led.record(f"v{i}", sharpe=0.05 + i * 1e-6, n_observations=250)

    assert led.sr_variance() == 1 / 250


def test_ledger_drives_the_gate_without_the_caller_tracking_trials(tmp_path):
    rng = np.random.default_rng(20260730)
    led = SearchLedger(tmp_path / "l.jsonl")
    gate = PromotionGate()

    # 400 variants of pure noise; keep the best-looking one.
    best_returns, best_sr = None, -np.inf
    for i in range(400):
        r = rng.normal(0.0, 0.01, size=252)
        t = led.record(f"v{i}", returns=r)
        if t.sharpe > best_sr:
            best_sr, best_returns = t.sharpe, r

    verdict = gate.evaluate_ledger(best_returns, led)

    assert led.n_trials == 400
    assert not verdict.passed
    assert verdict.sharpe_annual > 1.5      # looks great
    assert verdict.dsr < 0.95               # isn't
