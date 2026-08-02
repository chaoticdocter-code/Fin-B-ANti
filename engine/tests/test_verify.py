"""Credential probe classification.

No network calls here. What is tested is the part with a correctness contract:
failures must be classified into the right bucket (because the fixes differ),
and credentials must never survive into a message.
"""

from __future__ import annotations

import httpx
import pytest

from finb.config import Settings
from finb.data.verify import ProbeResult, _classify, probe_exchange, probe_fred

# --------------------------------------------------------------------------- #
#  Classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("HTTP 401: unauthorized"), "auth_failed"),
        (RuntimeError("403 Forbidden"), "auth_failed"),
        (RuntimeError("Invalid API key provided"), "auth_failed"),
        (RuntimeError("authentication failed"), "auth_failed"),
        (RuntimeError("HTTP 429 Too Many Requests"), "rate_limited"),
        (RuntimeError("rate limit exceeded"), "rate_limited"),
        (httpx.ConnectTimeout("connection timed out"), "unreachable"),
        (RuntimeError("Temporary failure in DNS resolution"), "unreachable"),
        (RuntimeError("something else entirely"), "error"),
    ],
)
def test_failures_land_in_the_right_bucket(exc, expected):
    status, _ = _classify(exc)
    assert status == expected


def test_a_leaked_key_is_redacted_from_the_message():
    """FRED and several others put the key in the query string, so an
    unmodified exception message publishes it into logs and tracebacks."""
    exc = RuntimeError(
        "Client error for url 'https://api.stlouisfed.org/fred/x?api_key=SECRET123&f=json'"
    )
    _, detail = _classify(exc)
    assert "SECRET123" not in detail
    assert "<redacted>" in detail


@pytest.mark.parametrize("marker", ["api_key=", "apikey=", "token="])
def test_every_credential_marker_is_redacted(marker):
    _, detail = _classify(RuntimeError(f"failed: https://x.test/?{marker}abcdef123456"))
    assert "abcdef123456" not in detail


# --------------------------------------------------------------------------- #
#  Missing credentials short-circuit without a network call
# --------------------------------------------------------------------------- #


def test_absent_key_reports_no_key_and_does_not_call_out(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not make a request without a key")

    monkeypatch.setattr(httpx, "get", explode)
    r = probe_fred(Settings(fred_api_key=None))
    assert r.status == "no_key"
    assert r.latency_ms is None


def test_exchange_without_both_halves_of_the_credential_is_skipped():
    s = Settings(kraken_api_key="present", kraken_api_secret=None)
    assert probe_exchange(s, "kraken").status == "no_key"


def test_okx_requires_a_passphrase_as_well():
    s = Settings(okx_api_key="k", okx_api_secret="s", okx_passphrase=None)
    assert probe_exchange(s, "okx").status == "no_key"

    s2 = Settings(okx_api_key="k", okx_api_secret="s", okx_passphrase="p")
    assert probe_exchange(s2, "okx").status != "no_key"


# --------------------------------------------------------------------------- #


def test_status_labels_are_human_readable():
    for status, label in [
        ("ok", "OK"),
        ("auth_failed", "BAD KEY"),
        ("rate_limited", "RATE LIMIT"),
        ("no_key", "not configured"),
    ]:
        assert ProbeResult("x", "y", status).symbol == label
