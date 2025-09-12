import json
import pytest
from datetime import datetime
from ally.tools import TOOL_REGISTRY
from ally.schemas.tcost import (
    Fill, OrderSide, FillQuality, TransactionCostConfig, 
    MarketMicrostructure, TransactionCostAnalysis
)
from ally.utils.tcost import (
    calculate_market_impact, calculate_spread_cost, 
    calculate_slippage, analyze_transaction_costs
)

pytestmark = pytest.mark.m11


def load_fixture(filename: str):
    """Load test fixture data."""
    fixture_path = f"data/fixtures/tcost/{filename}"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_transaction_cost_schemas():
    """Test transaction cost schema models."""
    # Test Fill model
    fill = Fill(
        fill_id="test_001",
        parent_order_id="order_001",
        symbol="AAPL", 
        side=OrderSide.BUY,
        fill_price=100.0,
        fill_size=1000,
        fill_time=datetime.now(),
        quality=FillQuality.AGGRESSIVE,
        commission=5.0,
        bid_at_fill=99.95,
        ask_at_fill=100.05,
        volume_before=50000
    )
    
    assert fill.notional == 100000.0
    assert fill.mid_at_fill == 100.0
    
    # Test MarketMicrostructure
    market = MarketMicrostructure(
        symbol="AAPL",
        timestamp=datetime.now(),
        bid_price=99.95,
        ask_price=100.05,
        bid_size=1000,
        ask_size=800,
        last_price=100.0,
        volume_1m=1500,
        volatility_1h=0.02
    )
    
    assert abs(market.spread_bps - 10.0) < 0.1
    assert market.mid_price == 100.0


def test_market_impact_calculation():
    """Test market impact calculation."""
    fill = Fill(
        fill_id="test", parent_order_id="", symbol="AAPL",
        side=OrderSide.BUY, fill_price=100.0, fill_size=1000,
        fill_time=datetime.now(), quality=FillQuality.AGGRESSIVE,
        commission=0, bid_at_fill=99.95, ask_at_fill=100.05,
        volume_before=50000
    )
    
    config = TransactionCostConfig(
        commission_bps=1.0, market_impact_alpha=0.6, 
        market_impact_beta=0.4
    )
    
    market = MarketMicrostructure(
        symbol="AAPL", timestamp=datetime.now(),
        bid_price=99.95, ask_price=100.05, bid_size=1000, 
        ask_size=800, last_price=100.0, volume_1m=1500,
        volatility_1h=0.02
    )
    
    impact = calculate_market_impact(fill, config, market)
    assert isinstance(impact, float)
    assert 0 <= impact <= 1000  # Reasonable bounds


def test_spread_cost_calculation():
    """Test spread cost calculation.""" 
    fill = Fill(
        fill_id="test", parent_order_id="", symbol="AAPL",
        side=OrderSide.BUY, fill_price=100.0, fill_size=1000,
        fill_time=datetime.now(), quality=FillQuality.AGGRESSIVE,
        commission=0, bid_at_fill=99.95, ask_at_fill=100.05,
        volume_before=50000
    )
    
    config = TransactionCostConfig(commission_bps=1.0, spread_capture_rate=0.5)
    
    spread_cost = calculate_spread_cost(fill, config)
    assert isinstance(spread_cost, float)
    assert spread_cost > 0  # Aggressive fills should have positive spread cost


def test_slippage_calculation():
    """Test slippage calculation."""
    fill = Fill(
        fill_id="test", parent_order_id="", symbol="AAPL",
        side=OrderSide.BUY, fill_price=100.5, fill_size=1000,
        fill_time=datetime.now(), quality=FillQuality.AGGRESSIVE,
        commission=0, bid_at_fill=99.95, ask_at_fill=100.05,
        volume_before=50000
    )
    
    benchmark_price = 100.0
    slippage = calculate_slippage(fill, benchmark_price)
    
    # Buy at 100.5 vs benchmark 100.0 = 50bps slippage
    assert abs(slippage - 50.0) < 0.1


def test_tcost_analyze_tool():
    """Test tcost.analyze tool with fixtures."""
    fills_data = load_fixture("sample_fills.json")
    config_data = load_fixture("config_sample.json") 
    market_data = load_fixture("market_data.json")
    
    result = TOOL_REGISTRY["tcost.analyze"](
        fills_data=fills_data,
        config_data=config_data,
        market_data=market_data
    )
    
    assert result.ok
    
    # Verify result structure
    data = result.data
    assert "analysis" in data
    assert "summary" in data
    
    # Verify summary metrics
    summary = data["summary"]
    assert "total_cost_bps" in summary
    assert "commission_bps" in summary
    assert "spread_bps" in summary
    assert "impact_bps" in summary
    assert "slippage_bps" in summary
    assert "fill_count" in summary
    assert "aggressive_ratio" in summary
    assert "implementation_shortfall_bps" in summary
    
    # Verify values are reasonable
    assert summary["fill_count"] == 3
    assert summary["total_cost_bps"] >= 0
    assert 0 <= summary["aggressive_ratio"] <= 1


def test_tcost_simulate_fills_tool():
    """Test tcost.simulate_fills tool."""
    market_conditions = {
        "price": 150.0,
        "volatility": 0.02,
        "spread_bps": 8,
        "avg_volume": 100000
    }
    
    result = TOOL_REGISTRY["tcost.simulate_fills"](
        symbol="AAPL",
        target_quantity=2000,
        order_side="buy",
        market_conditions=market_conditions,
        execution_strategy="twap", 
        time_horizon_minutes=30,
        seed=42
    )
    
    assert result.ok
    
    data = result.data
    assert "fills" in data
    assert "summary" in data
    assert "fills_fingerprint" in data
    
    # Verify summary
    summary = data["summary"] 
    assert summary["total_quantity"] == 2000
    assert summary["strategy"] == "twap"
    assert summary["total_fills"] > 0
    
    # Verify fills structure
    fills = data["fills"]
    assert len(fills) > 0
    
    for fill in fills:
        assert "fill_id" in fill
        assert "symbol" in fill
        assert "fill_price" in fill
        assert "fill_size" in fill
        assert fill["symbol"] == "AAPL"
        assert fill["fill_price"] > 0
        assert fill["fill_size"] > 0


def test_tcost_benchmark_config_tool():
    """Test tcost.benchmark_config tool."""
    result = TOOL_REGISTRY["tcost.benchmark_config"](
        market_regime="normal",
        asset_class="equity"
    )
    
    assert result.ok
    
    data = result.data
    assert "config" in data
    assert "regime" in data
    assert "asset_class" in data 
    assert "config_hash" in data
    
    config = data["config"]
    assert "commission_bps" in config
    assert "spread_capture_rate" in config
    assert "market_impact_alpha" in config
    assert "market_impact_beta" in config
    
    # Test different regimes
    volatile_result = TOOL_REGISTRY["tcost.benchmark_config"](
        market_regime="volatile",
        asset_class="equity"
    )
    
    assert volatile_result.ok
    volatile_config = volatile_result.data["config"]
    
    # Volatile regime should have higher impact parameters
    assert volatile_config["market_impact_alpha"] > config["market_impact_alpha"]


def test_tcost_deterministic_behavior():
    """Test that transaction cost tools produce deterministic results."""
    market_conditions = {
        "price": 100.0,
        "volatility": 0.015,
        "spread_bps": 10,
        "avg_volume": 80000
    }
    
    # Run simulation twice with same seed
    result1 = TOOL_REGISTRY["tcost.simulate_fills"](
        symbol="TEST",
        target_quantity=1000, 
        order_side="buy",
        market_conditions=market_conditions,
        seed=1337
    )
    
    result2 = TOOL_REGISTRY["tcost.simulate_fills"](
        symbol="TEST",
        target_quantity=1000,
        order_side="buy", 
        market_conditions=market_conditions,
        seed=1337
    )
    
    assert result1.ok and result2.ok
    
    # Fingerprints should be identical
    assert result1.data["fills_fingerprint"] == result2.data["fills_fingerprint"]
    assert result1.data["summary"]["total_quantity"] == result2.data["summary"]["total_quantity"]
    
    # Individual fill prices should be identical
    fills1 = result1.data["fills"]
    fills2 = result2.data["fills"]
    
    assert len(fills1) == len(fills2)
    
    for f1, f2 in zip(fills1, fills2):
        assert f1["fill_price"] == f2["fill_price"]
        assert f1["fill_size"] == f2["fill_size"]


def test_full_tcost_analysis_workflow():
    """Test complete transaction cost analysis workflow."""
    # 1. Generate benchmark config
    config_result = TOOL_REGISTRY["tcost.benchmark_config"](
        market_regime="normal",
        asset_class="equity"
    )
    assert config_result.ok
    
    # 2. Simulate fills  
    market_conditions = {
        "price": 150.0,
        "volatility": 0.02,
        "spread_bps": 12,
        "avg_volume": 120000
    }
    
    fills_result = TOOL_REGISTRY["tcost.simulate_fills"](
        symbol="AAPL",
        target_quantity=1500,
        order_side="sell",
        market_conditions=market_conditions,
        seed=42
    )
    assert fills_result.ok
    
    # 3. Create market data (simplified)
    market_data = [
        {
            "symbol": "AAPL",
            "timestamp": "2024-01-15T10:00:00Z",
            "bid_price": 149.94,
            "ask_price": 150.06,
            "bid_size": 1000,
            "ask_size": 900,
            "last_price": 150.0,
            "volume_1m": 1800,
            "volatility_1h": 0.02
        }
    ]
    
    # 4. Analyze transaction costs
    analysis_result = TOOL_REGISTRY["tcost.analyze"](
        fills_data=fills_result.data["fills"],
        config_data=config_result.data["config"],
        market_data=market_data
    )
    assert analysis_result.ok
    
    # 5. Verify complete analysis
    summary = analysis_result.data["summary"]
    assert summary["fill_count"] == fills_result.data["summary"]["total_fills"]
    assert summary["total_cost_bps"] > 0  # Should have some cost
    assert abs(summary["aggressive_ratio"]) <= 1.001  # Valid ratio (allow for floating point precision)
    
    # Should have all cost components
    assert "commission_bps" in summary
    assert "spread_bps" in summary 
    assert "impact_bps" in summary
    assert "slippage_bps" in summary