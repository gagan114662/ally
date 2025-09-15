#!/usr/bin/env python3
"""
QuantConnect Paper trading gating tests - ensure proper live mode enforcement
"""

import os
import pytest

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_qc_paper_live_blocked_when_ally_live_0(monkeypatch):
    """live=True must hard-fail if ALLY_LIVE != 1."""
    monkeypatch.setenv("ALLY_LIVE", "0")
    monkeypatch.delenv("QC_USER_ID", raising=False)
    monkeypatch.delenv("QC_API_TOKEN", raising=False)
    
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    
    adapter = QCPaperAdapter()
    
    # Should fail with gating error
    with pytest.raises(Exception) as exc_info:
        adapter.start_session(
            project_slug="test-project",
            symbols=["AAPL"],
            live=True
        )
    
    error_msg = str(exc_info.value).lower()
    assert "ally_live" in error_msg and ("live" in error_msg or "network" in error_msg), f"Expected gating error, got: {exc_info.value}"


def test_qc_paper_live_blocked_when_missing_credentials(monkeypatch):
    """Even if ALLY_LIVE=1, missing QC credentials must hard-fail."""
    monkeypatch.setenv("ALLY_LIVE", "1")
    monkeypatch.delenv("QC_USER_ID", raising=False)
    monkeypatch.delenv("QC_API_TOKEN", raising=False)
    
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    
    adapter = QCPaperAdapter()
    
    # Should fail with credentials error
    with pytest.raises(Exception) as exc_info:
        adapter.start_session(
            project_slug="test-project",
            symbols=["AAPL"],
            live=True
        )
    
    error_msg = str(exc_info.value).lower()
    assert any(term in error_msg for term in ["qc_user_id", "qc_api_token", "credential"]), f"Expected credentials error, got: {exc_info.value}"


def test_qc_paper_live_blocked_with_placeholder_credentials(monkeypatch):
    """Placeholder credentials must be rejected explicitly."""
    monkeypatch.setenv("ALLY_LIVE", "1")
    
    placeholder_pairs = [
        ("your_qc_user_id_here", "your_qc_api_token_here"),
        ("", ""),
        ("demo", "demo"),
        ("123", "changeme"),
    ]
    
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    
    for user_id, token in placeholder_pairs:
        monkeypatch.setenv("QC_USER_ID", user_id)
        monkeypatch.setenv("QC_API_TOKEN", token)
        
        adapter = QCPaperAdapter()
        
        with pytest.raises(Exception) as exc_info:
            adapter.start_session(
                project_slug="test-project",
                symbols=["AAPL"],
                live=True
            )
        
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ["invalid", "placeholder", "credential"]), \
            f"Expected placeholder rejection for '{user_id}'/'{token}', got: {exc_info.value}"


def test_qc_paper_offline_does_not_require_credentials(monkeypatch):
    """live=False path must work offline and not require any credentials."""
    monkeypatch.setenv("ALLY_LIVE", "0")
    monkeypatch.delenv("QC_USER_ID", raising=False)
    monkeypatch.delenv("QC_API_TOKEN", raising=False)
    
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    
    adapter = QCPaperAdapter()
    
    # Should succeed in offline mode
    session = adapter.start_session(
        project_slug="test-project",
        symbols=["AAPL"],
        live=False  # offline mock path
    )
    
    assert session is not None, "Offline mode should work without credentials"
    assert session.backend == "qc_paper", "Backend should be qc_paper"
    assert session.status == "active", "Session should be active"
    assert "offline_mock" in session.metadata.get("mode", ""), "Should be offline mock mode"


def test_qc_paper_place_order_offline():
    """Test placing orders in offline mode"""
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    from ally.schemas.broker import OrderSide, OrderType
    
    adapter = QCPaperAdapter()
    
    # Place mock order (offline)
    order = adapter.place_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        live=False
    )
    
    assert order is not None, "Should return mock order"
    assert order.symbol == "AAPL", "Symbol should match"
    assert order.side == OrderSide.BUY, "Side should match"
    assert order.qty == 100, "Quantity should match"
    assert order.provider == "qc_paper", "Provider should be qc_paper"
    assert "offline_mock" in order.metadata.get("mode", ""), "Should be offline mock"


def test_qc_paper_get_account_offline():
    """Test getting account info in offline mode"""
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    
    adapter = QCPaperAdapter()
    
    # Get mock account (offline)
    account = adapter.get_account(live=False)
    
    assert account is not None, "Should return mock account"
    assert account.provider == "qc_paper", "Provider should be qc_paper"
    assert account.cash > 0, "Should have positive cash balance"
    assert "offline_mock" in account.metadata.get("mode", ""), "Should be offline mock"


def test_qc_paper_direct_adapter_gating():
    """Test gating at the adapter level directly."""
    from ally.adapters.broker.qc_paper_adapter import QCPaperAdapter
    from ally.utils.gating import LiveModeError
    
    # Test ALLY_LIVE=0 blocks live mode
    with monkeypatch.context() as m:
        m.setenv("ALLY_LIVE", "0")
        adapter = QCPaperAdapter(qc_user_id="test_user", qc_api_token="test_token")
        
        with pytest.raises(LiveModeError) as exc_info:
            adapter.start_session(
                project_slug="test-project",
                symbols=["AAPL"],
                live=True
            )
        
        assert "ALLY_LIVE=0" in str(exc_info.value)
        assert "QuantConnect" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])