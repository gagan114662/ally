import math
from typing import List, Tuple
from datetime import datetime
from ally.schemas.tcost import (
    Fill, OrderSide, FillQuality, TransactionCostConfig, 
    TransactionCostAnalysis, MarketMicrostructure
)


def calculate_market_impact(fill: Fill, config: TransactionCostConfig, 
                          market_data: MarketMicrostructure) -> float:
    """
    Calculate market impact in basis points using square-root model
    Impact = alpha * (volume / ADV) ^ beta * volatility
    """
    if market_data.volume_1m <= 0:
        return 0.0
    
    # Estimate ADV from 1-minute volume (rough approximation)
    estimated_adv = market_data.volume_1m * 60 * 24  # Scale to daily
    volume_ratio = fill.fill_size / max(estimated_adv, 1)
    
    impact_bps = (config.market_impact_alpha * 
                  (volume_ratio ** config.market_impact_beta) * 
                  market_data.volatility_1h * 10000)
    
    return min(impact_bps, 1000)  # Cap at 1000bps


def calculate_spread_cost(fill: Fill, config: TransactionCostConfig) -> float:
    """Calculate spread cost in basis points"""
    spread_bps = ((fill.ask_at_fill - fill.bid_at_fill) / fill.mid_at_fill) * 10000
    
    # Adjust based on fill quality and capture rate
    if fill.quality == FillQuality.PASSIVE:
        # Market making - can capture spread
        cost = spread_bps * (1 - config.spread_capture_rate)
    elif fill.quality == FillQuality.AGGRESSIVE:
        # Market taking - pays full spread
        cost = spread_bps
    else:  # MIXED
        # Partial spread cost
        cost = spread_bps * 0.7
    
    return cost


def calculate_slippage(fill: Fill, benchmark_price: float) -> float:
    """Calculate slippage vs benchmark in basis points"""
    if fill.side == OrderSide.BUY:
        slippage = ((fill.fill_price - benchmark_price) / benchmark_price) * 10000
    else:  # SELL
        slippage = ((benchmark_price - fill.fill_price) / benchmark_price) * 10000
    
    return max(slippage, 0)  # Only positive slippage


def calculate_commission_cost(fill: Fill, config: TransactionCostConfig) -> float:
    """Calculate commission cost in basis points"""
    return (fill.commission / fill.notional) * 10000 if fill.notional > 0 else 0


def analyze_transaction_costs(fills: List[Fill], config: TransactionCostConfig,
                            market_data_list: List[MarketMicrostructure]) -> TransactionCostAnalysis:
    """
    Comprehensive transaction cost analysis for a set of fills
    """
    if not fills:
        raise ValueError("No fills provided for analysis")
    
    # Create market data lookup
    market_lookup = {md.timestamp: md for md in market_data_list}
    
    total_notional = sum(fill.notional for fill in fills)
    total_commission = sum(fill.commission for fill in fills)
    
    # Calculate VWAP benchmark
    vwap = sum(fill.fill_price * fill.fill_size for fill in fills) / sum(fill.fill_size for fill in fills)
    
    # Cost components (weighted by notional)
    commission_cost_total = 0
    spread_cost_total = 0 
    impact_cost_total = 0
    slippage_total = 0
    
    aggressive_notional = 0
    
    for fill in fills:
        weight = fill.notional / total_notional
        
        # Find closest market data
        closest_market = min(market_data_list, key=lambda md: abs((md.timestamp - fill.fill_time).total_seconds()))
        
        commission_cost_total += calculate_commission_cost(fill, config) * weight
        spread_cost_total += calculate_spread_cost(fill, config) * weight
        impact_cost_total += calculate_market_impact(fill, config, closest_market) * weight
        slippage_total += calculate_slippage(fill, closest_market.mid_price) * weight
        
        if fill.quality == FillQuality.AGGRESSIVE:
            aggressive_notional += fill.notional
    
    total_cost_bps = commission_cost_total + spread_cost_total + impact_cost_total + slippage_total
    
    # Implementation shortfall (vs arrival price - use first market data as proxy)
    arrival_price = market_data_list[0].mid_price if market_data_list else vwap
    # Use deterministic benchmark spread for reproducible results
    benchmark_bid = arrival_price * 0.999  # 0.1% spread deterministically
    benchmark_ask = arrival_price * 1.001
    
    implementation_shortfall_bps = calculate_slippage(
        Fill(
            fill_id="benchmark", parent_order_id="", symbol=fills[0].symbol,
            side=fills[0].side, fill_price=vwap, fill_size=1, fill_time=fills[0].fill_time,
            quality=FillQuality.MIXED, bid_at_fill=benchmark_bid, ask_at_fill=benchmark_ask
        ), 
        arrival_price
    )
    
    # Calculate price improvement (negative means cost)
    price_improvement_bps = -slippage_total  # Negative slippage is improvement
    
    return TransactionCostAnalysis(
        config=config,
        fills=fills,
        analysis_time=datetime.now(),
        commission_cost_bps=commission_cost_total,
        spread_cost_bps=spread_cost_total, 
        market_impact_bps=impact_cost_total,
        slippage_bps=slippage_total,
        total_cost_bps=total_cost_bps,
        total_notional=total_notional,
        total_commission=total_commission,
        fill_count=len(fills),
        avg_fill_size=sum(fill.fill_size for fill in fills) / len(fills),
        aggressive_fill_ratio=aggressive_notional / total_notional,
        implementation_shortfall_bps=implementation_shortfall_bps,
        volume_weighted_price=vwap,
        price_improvement_bps=price_improvement_bps
    )