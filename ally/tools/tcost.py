import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from ally.schemas.base import ToolResult
from ally.schemas.tcost import (
    Fill, OrderSide, FillQuality, TransactionCostConfig, 
    MarketMicrostructure, TransactionCostAnalysis
)
from ally.utils.tcost import analyze_transaction_costs
from . import register


@register("tcost.analyze")
def tcost_analyze(
    fills_data: List[Dict[str, Any]],
    config_data: Dict[str, Any],
    market_data: List[Dict[str, Any]]
) -> ToolResult:
    """
    Analyze transaction costs for a set of fills
    
    Args:
        fills_data: List of fill records
        config_data: Transaction cost configuration
        market_data: Market microstructure data
    """
    try:
        # Parse inputs
        config = TransactionCostConfig(**config_data)
        fills = [Fill(**fill_dict) for fill_dict in fills_data]
        market_structs = [MarketMicrostructure(**md) for md in market_data]
        
        # Perform analysis
        analysis = analyze_transaction_costs(fills, config, market_structs)
        
        return ToolResult(
            ok=True,
            data={
                "analysis": analysis.model_dump(),
                "summary": {
                    "total_cost_bps": analysis.total_cost_bps,
                    "commission_bps": analysis.commission_cost_bps,
                    "spread_bps": analysis.spread_cost_bps,
                    "impact_bps": analysis.market_impact_bps,
                    "slippage_bps": analysis.slippage_bps,
                    "fill_count": analysis.fill_count,
                    "aggressive_ratio": analysis.aggressive_fill_ratio,
                    "implementation_shortfall_bps": analysis.implementation_shortfall_bps
                }
            }
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Transaction cost analysis failed: {str(e)}"}
        )


@register("tcost.simulate_fills")
def tcost_simulate_fills(
    symbol: str,
    target_quantity: int,
    order_side: str,
    market_conditions: Dict[str, Any],
    execution_strategy: str = "twap",
    time_horizon_minutes: int = 30,
    seed: int = 42
) -> ToolResult:
    """
    Simulate realistic fill execution for transaction cost analysis
    
    Args:
        symbol: Trading symbol
        target_quantity: Target order size
        order_side: "buy" or "sell"
        market_conditions: Market parameters
        execution_strategy: Execution algorithm
        time_horizon_minutes: Execution time window
        seed: Random seed for deterministic results
    """
    try:
        np.random.seed(seed)
        
        side = OrderSide(order_side.lower())
        
        # Extract market conditions
        base_price = market_conditions.get("price", 100.0)
        volatility = market_conditions.get("volatility", 0.02)
        spread_bps = market_conditions.get("spread_bps", 10)
        avg_volume = market_conditions.get("avg_volume", 100000)
        
        # Generate fills based on strategy
        fills = []
        remaining_qty = target_quantity
        # Use deterministic timestamp based on seed for reproducibility (timezone-aware)
        from datetime import timezone
        current_time = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc) + timedelta(seconds=seed % 3600)
        
        if execution_strategy == "twap":
            # Time-weighted average price strategy
            num_slices = min(10, time_horizon_minutes // 3)  # Every 3 minutes
            slice_size = target_quantity // num_slices
            
            for i in range(num_slices):
                if remaining_qty <= 0:
                    break
                    
                fill_size = min(slice_size, remaining_qty)
                if i == num_slices - 1:  # Last slice gets remainder
                    fill_size = remaining_qty
                
                # Simulate price movement
                price_drift = np.random.normal(0, volatility * np.sqrt(3/60))  # 3-minute movement
                current_price = base_price * (1 + price_drift)
                
                # Add spread
                spread_half = (spread_bps / 10000) * current_price / 2
                bid_price = current_price - spread_half
                ask_price = current_price + spread_half
                
                # Determine fill quality based on market participation rate
                participation_rate = fill_size / (avg_volume / (24 * 60 / 3))  # 3-minute slice
                
                if participation_rate < 0.1:
                    quality = FillQuality.PASSIVE
                    fill_price = bid_price if side == OrderSide.SELL else ask_price
                elif participation_rate > 0.3:
                    quality = FillQuality.AGGRESSIVE  
                    fill_price = ask_price if side == OrderSide.BUY else bid_price
                else:
                    quality = FillQuality.MIXED
                    # Mixed execution between mid and spread
                    mid = (bid_price + ask_price) / 2
                    if side == OrderSide.BUY:
                        fill_price = mid + spread_half * np.random.uniform(0, 0.8)
                    else:
                        fill_price = mid - spread_half * np.random.uniform(0, 0.8)
                
                fill = Fill(
                    fill_id=f"fill_{i+1}",
                    parent_order_id="simulated_order",
                    symbol=symbol,
                    side=side,
                    fill_price=fill_price,
                    fill_size=fill_size,
                    fill_time=current_time + timedelta(minutes=i*3),
                    quality=quality,
                    commission=fill_size * 0.005,  # $0.005 per share
                    bid_at_fill=bid_price,
                    ask_at_fill=ask_price,
                    volume_before=int(avg_volume * np.random.uniform(0.8, 1.2))
                )
                
                fills.append(fill)
                remaining_qty -= fill_size
                current_time += timedelta(minutes=3)
        
        # Create deterministic hash for reproducibility
        fills_json = json.dumps([fill.model_dump() for fill in fills], sort_keys=True, default=str)
        fills_fingerprint = hashlib.sha1(fills_json.encode()).hexdigest()
        
        return ToolResult(
            ok=True,
            data={
                "fills": [fill.model_dump() for fill in fills],
                "summary": {
                    "total_fills": len(fills),
                    "total_quantity": sum(fill.fill_size for fill in fills),
                    "avg_fill_price": sum(fill.fill_price * fill.fill_size for fill in fills) / sum(fill.fill_size for fill in fills),
                    "execution_time_minutes": time_horizon_minutes,
                    "strategy": execution_strategy
                },
                "fills_fingerprint": fills_fingerprint
            }
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Fill simulation failed: {str(e)}"}
        )


@register("tcost.benchmark_config")
def tcost_benchmark_config(
    market_regime: str = "normal",
    asset_class: str = "equity"
) -> ToolResult:
    """
    Generate benchmark transaction cost configuration
    
    Args:
        market_regime: "normal", "volatile", "stressed"
        asset_class: "equity", "fx", "crypto"
    """
    try:
        # Base configurations by asset class
        base_configs = {
            "equity": {
                "commission_bps": 1.0,
                "spread_capture_rate": 0.5,
                "market_impact_alpha": 0.6,
                "market_impact_beta": 0.4,
                "slippage_tolerance_bps": 50,
                "min_fill_size": 100
            },
            "fx": {
                "commission_bps": 0.5,
                "spread_capture_rate": 0.7,
                "market_impact_alpha": 0.4,
                "market_impact_beta": 0.3,
                "slippage_tolerance_bps": 25,
                "min_fill_size": 10000
            },
            "crypto": {
                "commission_bps": 5.0,
                "spread_capture_rate": 0.3,
                "market_impact_alpha": 0.8,
                "market_impact_beta": 0.5,
                "slippage_tolerance_bps": 100,
                "min_fill_size": 1
            }
        }
        
        config_dict = base_configs.get(asset_class, base_configs["equity"]).copy()
        
        # Adjust for market regime
        if market_regime == "volatile":
            config_dict["market_impact_alpha"] *= 1.3
            config_dict["slippage_tolerance_bps"] *= 1.5
        elif market_regime == "stressed":
            config_dict["market_impact_alpha"] *= 1.8
            config_dict["slippage_tolerance_bps"] *= 2.0
            config_dict["spread_capture_rate"] *= 0.7
        
        config = TransactionCostConfig(**config_dict)
        
        # Create deterministic fingerprint
        config_json = json.dumps(config.model_dump(), sort_keys=True)
        config_hash = hashlib.sha1(config_json.encode()).hexdigest()
        
        return ToolResult(
            ok=True,
            data={
                "config": config.model_dump(),
                "regime": market_regime,
                "asset_class": asset_class,
                "config_hash": config_hash
            }
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Benchmark config generation failed: {str(e)}"}
        )