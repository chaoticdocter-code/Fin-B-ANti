"""Automated lookahead detection for feature functions.

[[Leakage]] showed what a leak is worth: shuffled k-fold scored 0.955 on a
random walk. That was a *cross-validation* leak. This module catches the other
kind — a feature that reads the future directly.

These bugs are quiet. A stray `.rolling(...).mean()` without a shift, a
`.fillna(method="bfill")`, a normalisation using the full series' mean, a
resample that labels a bar with its closing timestamp — none of them raise.
They just make the backtest better.

The core idea is a property that any causal feature must satisfy:

    computing features on the first t bars, and taking the last row,
    must give exactly the same answer as computing on all bars and
    taking row t-1.

If those differ, the value at time t-1 depended on data after t-1. There is no
way for that to be innocent. Three checks are implemented:

- **truncation equality** — the property above. The strongest and cheapest test.
- **future noise** — corrupt the bars *after* time t and confirm nothing at or
  before t moves.
- **warmup sanity** — a feature with a lookback must be null at the start;
  a fully-populated column often means someone back-filled.

Wire `assert_causal` into CI for every feature function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    check: str
    column: str
    row: int
    expected: float | None
    actual: float | None
    detail: str

    def __str__(self) -> str:
        return (
            f"[{self.check}] column '{self.column}' row {self.row}: "
            f"expected {self.expected!r}, got {self.actual!r} — {self.detail}"
        )


@dataclass(slots=True)
class LeakageReport:
    checks_run: int = 0
    findings: list[LeakageFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def leaking_columns(self) -> list[str]:
        return sorted({f.column for f in self.findings})

    def __str__(self) -> str:
        if self.passed:
            return f"causal: {self.checks_run} checks passed, no lookahead detected"
        head = (
            f"LOOKAHEAD DETECTED in {len(self.leaking_columns)} column(s): "
            f"{', '.join(self.leaking_columns)}"
        )
        return head + "\n  " + "\n  ".join(str(f) for f in self.findings[:12])


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in df.schema.items() if dt.is_numeric()]


def _value(df: pl.DataFrame, col: str, row: int) -> float | None:
    v = df[col][row]
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else v


def _differs(a: float | None, b: float | None, tol: float) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) > tol * scale


def _probe_rows(n: int, warmup: int, count: int) -> list[int]:
    """Evenly spaced rows in the post-warmup region, always including the last."""
    lo = max(warmup + 1, 2)
    if n <= lo:
        return []
    rows = sorted({int(x) for x in np.linspace(lo, n - 1, min(count, n - lo))})
    return [r for r in rows if lo <= r < n]


def check_truncation_equality(
    fn: FeatureFn,
    bars: pl.DataFrame,
    *,
    warmup: int = 0,
    probes: int = 24,
    tol: float = 1e-9,
) -> tuple[int, list[LeakageFinding]]:
    """Features on `bars[:t]` must match features on all bars, at row t-1."""
    full = fn(bars)
    cols = _numeric_columns(full)
    findings: list[LeakageFinding] = []
    checks = 0

    for t in _probe_rows(bars.height, warmup, probes):
        partial = fn(bars.head(t))
        if partial.height != t:
            findings.append(
                LeakageFinding(
                    "truncation", "<shape>", t, t, partial.height,
                    "feature function changed row count on truncated input",
                )
            )
            continue

        for col in cols:
            if col not in partial.columns:
                continue
            checks += 1
            a = _value(partial, col, t - 1)
            b = _value(full, col, t - 1)
            if _differs(a, b, tol):
                findings.append(
                    LeakageFinding(
                        "truncation", col, t - 1, b, a,
                        "value at this row changed when later bars were removed, "
                        "so it depended on them",
                    )
                )
    return checks, findings


def check_future_noise(
    fn: FeatureFn,
    bars: pl.DataFrame,
    *,
    warmup: int = 0,
    probes: int = 6,
    tol: float = 1e-9,
    shock: float = 3.0,
) -> tuple[int, list[LeakageFinding]]:
    """Corrupting bars after time t must not move any feature at or before t."""
    full = fn(bars)
    cols = _numeric_columns(full)
    findings: list[LeakageFinding] = []
    checks = 0
    price_cols = [c for c in ("open", "high", "low", "close") if c in bars.columns]

    for t in _probe_rows(bars.height, warmup, probes):
        idx = pl.int_range(pl.len()).alias("__i")
        corrupted = bars.with_columns(
            [
                pl.when(idx >= t).then(pl.col(c) * shock).otherwise(pl.col(c)).alias(c)
                for c in price_cols
            ]
        )
        out = fn(corrupted)
        if out.height != full.height:
            continue

        for col in cols:
            if col not in out.columns:
                continue
            checks += 1
            # Compare the ENTIRE prefix, not a sample of it. Leaks are often
            # sparse — a back-fill only corrupts the rows that happened to be
            # null — and evenly-spaced probes will walk straight past them.
            a = full[col].head(t).to_numpy().astype(float)
            b = out[col].head(t).to_numpy().astype(float)

            a_null = ~np.isfinite(a)
            b_null = ~np.isfinite(b)
            null_mismatch = a_null != b_null
            both_present = ~a_null & ~b_null

            scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
            value_mismatch = np.zeros_like(null_mismatch)
            value_mismatch[both_present] = (
                np.abs(a[both_present] - b[both_present]) > tol * scale[both_present]
            )

            bad = np.flatnonzero(null_mismatch | value_mismatch)
            if bad.size:
                row = int(bad[0])
                findings.append(
                    LeakageFinding(
                        "future-noise", col, row,
                        _value(full, col, row), _value(out, col, row),
                        f"changed when bars from index {t} onward were corrupted "
                        f"({bad.size} row(s) affected)",
                    )
                )
    return checks, findings


def check_warmup(
    fn: FeatureFn, bars: pl.DataFrame, *, expect_warmup: int
) -> tuple[int, list[LeakageFinding]]:
    """A feature with an N-bar lookback cannot be defined before bar N.

    A fully-populated column usually means a back-fill, which is lookahead
    wearing a convenience costume.
    """
    if expect_warmup <= 0:
        return 0, []
    out = fn(bars)
    findings, checks = [], 0
    for col in _numeric_columns(out):
        checks += 1
        if out[col].null_count() == 0:
            findings.append(
                LeakageFinding(
                    "warmup", col, 0, None, _value(out, col, 0),
                    f"no nulls despite a {expect_warmup}-bar lookback — "
                    "was this back-filled?",
                )
            )
    return checks, findings


def audit_features(
    fn: FeatureFn,
    bars: pl.DataFrame,
    *,
    warmup: int = 0,
    tol: float = 1e-9,
    expect_warmup: int | None = None,
) -> LeakageReport:
    """Run every check. `warmup` skips early rows that are legitimately null."""
    report = LeakageReport()

    for checker in (
        lambda: check_truncation_equality(fn, bars, warmup=warmup, tol=tol),
        lambda: check_future_noise(fn, bars, warmup=warmup, tol=tol),
    ):
        n, found = checker()
        report.checks_run += n
        report.findings.extend(found)

    if expect_warmup:
        n, found = check_warmup(fn, bars, expect_warmup=expect_warmup)
        report.checks_run += n
        report.findings.extend(found)

    return report


def assert_causal(fn: FeatureFn, bars: pl.DataFrame, **kwargs) -> None:
    """Raise if `fn` looks ahead. Use this in tests for every feature function."""
    report = audit_features(fn, bars, **kwargs)
    if not report.passed:
        raise AssertionError(str(report))
