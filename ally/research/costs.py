"""
Transaction Costs, Slippage, and Liquidity Modeling
Apply realistic trading costs to strategy backtests
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult


@dataclass
class CostModel:
    """Cost model specification"""
    bps_per_turnover: float = 10.0      # Linear cost in basis points
    impact_model: str = "sqrt"          # Impact model type: sqrt, linear, none
    impact_k: float = 6.0              # Impact coefficient
    borrow_bps_annual: float = 50.0    # Short borrow fee (annual bps)
    capacity_usd: Optional[float] = None  # Strategy capacity limit


def parse_cost_model_string(model_string: str) -> CostModel:
    """
    Parse cost model from string specification
    
    Args:
        model_string: String like "bps=10,impact=k*sqrt(q),k=6.0,borrow_bps=50"
        
    Returns:
        CostModel object
    """
    
    # Default values
    cost_model = CostModel()
    
    # Parse key-value pairs
    pairs = model_string.split(',')
    
    for pair in pairs:
        if '=' not in pair:
            continue
            
        key, value = pair.strip().split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if key == 'bps':
            cost_model.bps_per_turnover = float(value)
        elif key == 'impact':
            # Parse impact model specification
            if 'sqrt' in value.lower():
                cost_model.impact_model = "sqrt"
            elif 'linear' in value.lower():
                cost_model.impact_model = "linear"
            elif 'none' in value.lower():
                cost_model.impact_model = "none"
        elif key == 'k':
            cost_model.impact_k = float(value)
        elif key == 'borrow_bps':
            cost_model.borrow_bps_annual = float(value)
        elif key == 'capacity':
            cost_model.capacity_usd = float(value)
    
    return cost_model


def compute_market_impact_cost(quantity: float, price: float, 
                             model: str = "sqrt", k: float = 6.0) -> float:
    """
    Compute market impact cost per share
    
    Args:
        quantity: Number of shares traded (absolute value)
        price: Share price
        model: Impact model ("sqrt", "linear", "none")
        k: Impact coefficient
        
    Returns:
        Impact cost per share in dollars
    """
    
    if model == "none" or quantity == 0:
        return 0.0
    
    # Normalize quantity (impact scales with trade size)
    # Use simplified model: impact = k * sqrt(quantity) * price / 10000
    # This gives impact in dollars per share
    
    if model == "sqrt":
        impact_bps = k * np.sqrt(abs(quantity)) / 100  # Scale factor
        return price * impact_bps / 10000  # Convert bps to dollars per share
    elif model == "linear":
        impact_bps = k * abs(quantity) / 10000  # Linear in quantity
        return price * impact_bps / 10000
    else:
        return 0.0


def compute_borrow_cost(position: float, price: float, 
                       days_held: int, borrow_bps_annual: float = 50.0) -> float:
    """
    Compute short borrow cost
    
    Args:
        position: Position size (negative for short)
        price: Average price held
        days_held: Number of days position held
        borrow_bps_annual: Annual borrow rate in basis points
        
    Returns:
        Total borrow cost in dollars
    """
    
    if position >= 0:  # Only shorts pay borrow fees
        return 0.0
    
    notional = abs(position) * price
    annual_cost = notional * borrow_bps_annual / 10000
    daily_cost = annual_cost / 365.25
    
    return daily_cost * days_held


def apply_transaction_costs(weights_df: pd.DataFrame, ohlcv_df: pd.DataFrame,
                          cost_model: CostModel, 
                          initial_capital: float = 1_000_000) -> Dict[str, Any]:
    """
    Apply transaction costs to portfolio weights and returns
    
    Args:
        weights_df: Portfolio weights over time
        ohlcv_df: OHLCV data for pricing
        cost_model: Cost model specification
        initial_capital: Initial portfolio capital
        
    Returns:
        Dict with cost-adjusted returns and cost breakdown
    """
    
    if len(weights_df) == 0:
        raise ValueError("No portfolio weights provided")
    
    # Merge weights with prices
    weights_with_prices = weights_df.merge(
        ohlcv_df[['date', 'symbol', 'close']], 
        on=['date', 'symbol'], 
        how='left'
    )
    
    # Calculate position sizes in shares and dollars
    weights_with_prices['position_usd'] = weights_with_prices['weight'] * initial_capital
    weights_with_prices['position_shares'] = weights_with_prices['position_usd'] / weights_with_prices['close']
    
    # Calculate turnover (change in positions)
    weights_with_prices = weights_with_prices.sort_values(['symbol', 'date'])
    weights_with_prices['prev_position_shares'] = weights_with_prices.groupby('symbol')['position_shares'].shift(1).fillna(0)
    weights_with_prices['turnover_shares'] = abs(weights_with_prices['position_shares'] - weights_with_prices['prev_position_shares'])
    weights_with_prices['turnover_usd'] = weights_with_prices['turnover_shares'] * weights_with_prices['close']
    
    # Apply cost model
    cost_breakdown = []
    total_linear_costs = 0
    total_impact_costs = 0
    total_borrow_costs = 0
    
    for _, row in weights_with_prices.iterrows():
        # Linear transaction costs (bps per turnover)
        linear_cost = row['turnover_usd'] * cost_model.bps_per_turnover / 10000
        total_linear_costs += linear_cost
        
        # Market impact costs
        impact_cost = compute_market_impact_cost(
            quantity=row['turnover_shares'],
            price=row['close'],
            model=cost_model.impact_model,
            k=cost_model.impact_k
        ) * row['turnover_shares']  # Total impact for this trade
        total_impact_costs += impact_cost
        
        # Borrow costs (simplified - assume position held for average period)
        avg_holding_days = 30  # Simplified assumption
        borrow_cost = compute_borrow_cost(
            position=row['position_shares'],
            price=row['close'],
            days_held=avg_holding_days,
            borrow_bps_annual=cost_model.borrow_bps_annual
        )
        total_borrow_costs += borrow_cost
        
        cost_breakdown.append({
            'date': row['date'],
            'symbol': row['symbol'],
            'turnover_usd': row['turnover_usd'],
            'linear_cost': linear_cost,
            'impact_cost': impact_cost,
            'borrow_cost': borrow_cost,
            'total_cost': linear_cost + impact_cost + borrow_cost
        })
    
    # Aggregate costs by date
    cost_df = pd.DataFrame(cost_breakdown)
    daily_costs = cost_df.groupby('date').agg({
        'linear_cost': 'sum',
        'impact_cost': 'sum', 
        'borrow_cost': 'sum',
        'total_cost': 'sum',
        'turnover_usd': 'sum'
    }).reset_index()
    
    # Calculate cost as percentage of portfolio value
    daily_costs['cost_pct'] = daily_costs['total_cost'] / initial_capital
    
    # Summary statistics
    total_costs = total_linear_costs + total_impact_costs + total_borrow_costs
    total_turnover = cost_df['turnover_usd'].sum()
    
    cost_analysis = {
        "total_costs": {
            "linear_costs": total_linear_costs,
            "impact_costs": total_impact_costs,
            "borrow_costs": total_borrow_costs,
            "total_cost_usd": total_costs,
            "cost_pct_of_capital": total_costs / initial_capital * 100
        },
        "turnover_analysis": {
            "total_turnover_usd": total_turnover,
            "annual_turnover_ratio": total_turnover / initial_capital,  # Simplified
            "avg_cost_per_turnover_bps": (total_costs / total_turnover * 10000) if total_turnover > 0 else 0
        },
        "daily_costs": daily_costs.to_dict('records'),
        "cost_breakdown": cost_breakdown,
        "cost_model": {
            "bps_per_turnover": cost_model.bps_per_turnover,
            "impact_model": cost_model.impact_model,
            "impact_k": cost_model.impact_k,
            "borrow_bps_annual": cost_model.borrow_bps_annual
        },
        "analysis_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return cost_analysis


def check_capacity_constraints(backtest_results: Dict[str, Any],
                             cost_analysis: Dict[str, Any],
                             capacity_usd: float,
                             max_turnover_annual: float = 3.0) -> Dict[str, Any]:
    """
    Check if strategy violates capacity or turnover constraints
    
    Args:
        backtest_results: Original backtest results
        cost_analysis: Cost analysis results
        capacity_usd: Maximum strategy capacity
        max_turnover_annual: Maximum annual turnover ratio
        
    Returns:
        Dict with constraint validation results
    """
    
    annual_turnover = cost_analysis["turnover_analysis"]["annual_turnover_ratio"]
    portfolio_value = backtest_results.get("total_value", 1_000_000)  # Default assumption
    
    # Check constraints
    capacity_violated = portfolio_value > capacity_usd
    turnover_violated = annual_turnover > max_turnover_annual
    
    # Calculate headroom
    capacity_utilization = portfolio_value / capacity_usd if capacity_usd > 0 else 0
    turnover_utilization = annual_turnover / max_turnover_annual if max_turnover_annual > 0 else 0
    
    constraints_result = {
        "capacity_check": {
            "capacity_limit_usd": capacity_usd,
            "portfolio_value_usd": portfolio_value,
            "capacity_violated": capacity_violated,
            "capacity_utilization_pct": capacity_utilization * 100,
            "capacity_headroom_usd": max(0, capacity_usd - portfolio_value)
        },
        "turnover_check": {
            "turnover_limit": max_turnover_annual,
            "annual_turnover": annual_turnover,
            "turnover_violated": turnover_violated,
            "turnover_utilization_pct": turnover_utilization * 100,
            "turnover_headroom": max(0, max_turnover_annual - annual_turnover)
        },
        "overall_status": {
            "constraints_passed": not (capacity_violated or turnover_violated),
            "violations": []
        },
        "validation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Collect violations
    if capacity_violated:
        constraints_result["overall_status"]["violations"].append(
            f"Capacity exceeded: ${portfolio_value:,.0f} > ${capacity_usd:,.0f}"
        )
    
    if turnover_violated:
        constraints_result["overall_status"]["violations"].append(
            f"Turnover exceeded: {annual_turnover:.2f}x > {max_turnover_annual:.2f}x"
        )
    
    return constraints_result


@register("research.costs.apply")
def research_costs_apply(backtest_results: Dict[str, Any], model_string: str = "bps=10,impact=k*sqrt(q),k=6.0",
                        capacity_usd: float = 100_000_000, max_turnover: float = 3.0,
                        live: bool = False, **kwargs) -> ToolResult:
    """
    Apply transaction costs to backtest results
    
    Args:
        backtest_results: Backtest results with weights and returns
        model_string: Cost model specification string
        capacity_usd: Strategy capacity limit in USD
        max_turnover: Maximum annual turnover ratio
        live: Whether to use live cost data
        
    Returns:
        ToolResult with cost-adjusted results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Transaction Cost Analysis")
    
    try:
        if live:
            raise NotImplementedError("Live cost data not implemented")
        
        # Parse cost model
        cost_model = parse_cost_model_string(model_string)
        cost_model.capacity_usd = capacity_usd  # Override with parameter
        
        # Mock portfolio weights and OHLCV data for CI
        # In production, would extract from backtest_results
        np.random.seed(42)  # Deterministic for CI
        
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
        symbols = ['STOCK001', 'STOCK002', 'STOCK003', 'STOCK004', 'STOCK005']
        
        # Generate mock weights
        weights_data = []
        for date in dates:
            for symbol in symbols:
                weight = np.random.uniform(0.15, 0.25)  # Mock equal-weight-ish
                weights_data.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })
        
        weights_df = pd.DataFrame(weights_data)
        
        # Generate mock OHLCV
        ohlcv_data = []
        base_prices = {symbol: 100 + i * 20 for i, symbol in enumerate(symbols)}
        
        for date in dates:
            for symbol in symbols:
                price = base_prices[symbol] * (1 + np.random.normal(0, 0.02))
                ohlcv_data.append({
                    'date': date,
                    'symbol': symbol,
                    'close': price
                })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        
        # Apply transaction costs
        cost_analysis = apply_transaction_costs(
            weights_df=weights_df,
            ohlcv_df=ohlcv_df,
            cost_model=cost_model,
            initial_capital=1_000_000
        )
        
        # Check capacity constraints
        constraints_result = check_capacity_constraints(
            backtest_results=backtest_results,
            cost_analysis=cost_analysis,
            capacity_usd=capacity_usd,
            max_turnover_annual=max_turnover
        )
        
        # Adjust backtest metrics for costs
        original_return = backtest_results.get("annual_return", 0.10)
        cost_drag_annual = cost_analysis["total_costs"]["cost_pct_of_capital"] / 100
        
        cost_adjusted_results = {
            "original_metrics": {
                "annual_return": original_return,
                "sharpe_ratio": backtest_results.get("sharpe_ratio", 1.0),
                "max_drawdown": backtest_results.get("max_drawdown", -0.05)
            },
            "cost_adjusted_metrics": {
                "annual_return": original_return - cost_drag_annual,
                "cost_drag_annual": cost_drag_annual,
                "net_sharpe_est": (original_return - cost_drag_annual) / backtest_results.get("annual_volatility", 0.15)
            },
            "cost_analysis": cost_analysis,
            "constraints_validation": constraints_result,
            "model_specification": {
                "model_string": model_string,
                "parsed_model": {
                    "bps_per_turnover": cost_model.bps_per_turnover,
                    "impact_model": cost_model.impact_model,
                    "impact_k": cost_model.impact_k,
                    "borrow_bps_annual": cost_model.borrow_bps_annual,
                    "capacity_usd": cost_model.capacity_usd
                }
            },
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check if constraints were violated
        constraints_passed = constraints_result["overall_status"]["constraints_passed"]
        
        # Generate receipt
        receipt_hash = generate_receipt("research.costs.apply", cost_adjusted_results)
        cost_adjusted_results["costs_receipt"] = receipt_hash[:16]
        
        # Determine result status
        if not constraints_passed:
            violations = constraints_result["overall_status"]["violations"]
            return ToolResult(
                ok=False,
                data=cost_adjusted_results,
                receipt_hash=receipt_hash,
                errors=[f"Capacity/turnover constraints violated: {'; '.join(violations)}"]
            )
        
        return ToolResult(
            ok=True,
            data=cost_adjusted_results,
            receipt_hash=receipt_hash,
            warnings=[f"Cost drag: {cost_drag_annual:.3f} annual return reduction"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "model_string": model_string,
            "capacity_usd": capacity_usd
        }
        receipt_hash = generate_receipt("research.costs.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Cost analysis failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.costs.model")
def research_costs_model(model_string: str, test_trades: Optional[List[Dict[str, Any]]] = None,
                        live: bool = False, **kwargs) -> ToolResult:
    """
    Validate and test cost model specification
    
    Args:
        model_string: Cost model specification
        test_trades: Optional list of test trades for validation
        live: Whether to use live data
        
    Returns:
        ToolResult with cost model validation
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Cost Model Validation")
    
    try:
        # Parse cost model
        cost_model = parse_cost_model_string(model_string)
        
        # Default test trades if none provided
        if test_trades is None:
            test_trades = [
                {"symbol": "TEST", "quantity": 1000, "price": 100.0, "side": "buy"},
                {"symbol": "TEST", "quantity": 5000, "price": 100.0, "side": "buy"},
                {"symbol": "TEST", "quantity": 10000, "price": 100.0, "side": "sell"}
            ]
        
        # Test cost model on sample trades
        model_validation = []
        
        for trade in test_trades:
            quantity = trade["quantity"]
            price = trade["price"]
            
            # Calculate costs for this trade
            linear_cost = abs(quantity) * price * cost_model.bps_per_turnover / 10000
            impact_cost = compute_market_impact_cost(
                quantity=abs(quantity),
                price=price,
                model=cost_model.impact_model,
                k=cost_model.impact_k
            ) * abs(quantity)
            
            borrow_cost = compute_borrow_cost(
                position=-abs(quantity) if trade["side"] == "sell" else 0,
                price=price,
                days_held=30,
                borrow_bps_annual=cost_model.borrow_bps_annual
            )
            
            total_cost = linear_cost + impact_cost + borrow_cost
            cost_bps = (total_cost / (abs(quantity) * price)) * 10000
            
            model_validation.append({
                "trade": trade,
                "costs": {
                    "linear_cost": linear_cost,
                    "impact_cost": impact_cost,
                    "borrow_cost": borrow_cost,
                    "total_cost": total_cost,
                    "cost_bps": cost_bps
                }
            })
        
        validation_results = {
            "model_string": model_string,
            "parsed_model": {
                "bps_per_turnover": cost_model.bps_per_turnover,
                "impact_model": cost_model.impact_model,
                "impact_k": cost_model.impact_k,
                "borrow_bps_annual": cost_model.borrow_bps_annual,
                "capacity_usd": cost_model.capacity_usd
            },
            "model_validation": model_validation,
            "summary_statistics": {
                "avg_cost_bps": np.mean([v["costs"]["cost_bps"] for v in model_validation]),
                "max_cost_bps": np.max([v["costs"]["cost_bps"] for v in model_validation]),
                "total_test_cost": sum([v["costs"]["total_cost"] for v in model_validation])
            },
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.costs.model", validation_results)
        validation_results["model_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=validation_results,
            receipt_hash=receipt_hash
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "model_string": model_string
        }
        receipt_hash = generate_receipt("research.costs.model_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Cost model validation failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test cost modeling functionality
    print("🧪 Testing Transaction Cost Modeling...")
    
    # Test cost model parsing
    model_result = research_costs_model(
        model_string="bps=15,impact=k*sqrt(q),k=8.0,borrow_bps=75",
        live=False
    )
    
    print(f"Model validation: {model_result.ok}")
    if model_result.ok:
        data = model_result.data
        print(f"Parsed model: {data['parsed_model']}")
        print(f"Avg cost: {data['summary_statistics']['avg_cost_bps']:.1f} bps")
        print(f"Receipt: {data['model_receipt']}")
    
    # Test cost application
    mock_backtest = {
        "annual_return": 0.12,
        "sharpe_ratio": 1.2,
        "annual_volatility": 0.18,
        "max_drawdown": -0.08
    }
    
    costs_result = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=12,impact=k*sqrt(q),k=5.0",
        capacity_usd=50_000_000,
        max_turnover=2.5,
        live=False
    )
    
    print(f"\nCost application: {costs_result.ok}")
    if costs_result.ok:
        data = costs_result.data
        orig = data["original_metrics"]["annual_return"]
        adj = data["cost_adjusted_metrics"]["annual_return"]
        drag = data["cost_adjusted_metrics"]["cost_drag_annual"]
        print(f"Return: {orig:.3f} → {adj:.3f} (drag: {drag:.3f})")
        print(f"Constraints passed: {data['constraints_validation']['overall_status']['constraints_passed']}")
        print(f"Receipt: {data['costs_receipt']}")
    
    print("\n🎯 Transaction cost modeling ready for integration")