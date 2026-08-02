"""Unit tests for Free-Tier QuantConnect Integration & LEAN Exporter."""

from __future__ import annotations

from finb.data.sources.quantconnect import QuantConnectClient
from finb.sim.qc_adaptor import export_quick_move_to_lean


def test_quantconnect_signature_generation():
    client = QuantConnectClient(user_id="12345", api_token="test_token_secret")
    sig = client._generate_signature(1700000000)
    assert len(sig) == 64  # SHA-256 hex hash length
    assert client.is_configured is True


def test_quantconnect_free_unconfigured_fallback():
    client = QuantConnectClient()
    assert client.is_configured is False
    assert client.list_projects() == []


def test_export_quick_move_to_lean_open_source():
    code = export_quick_move_to_lean(["BTCUSD", "ETHUSD"], take_profit_pct=0.02, stop_loss_pct=0.012)
    assert "class FinBQuickMoveAlgorithm(QCAlgorithm):" in code
    assert "self.SetCash(500)" in code
    assert '"BTCUSD", "ETHUSD"' in code
