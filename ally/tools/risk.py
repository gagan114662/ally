"""
Risk Management Tool for Ally
Policy-driven risk checks for trading operations
"""

import hashlib
import json
from typing import List, Dict, Any


def simple_yaml_parse(yaml_str: str) -> Dict[str, Any]:
    """Simple YAML parser for basic policy configs"""
    result = {}
    current_dict = result
    lines = yaml_str.strip().split('\n')
    
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue
            
        # Handle nested dictionaries (basic support)
        if line.endswith(':') and not line.strip().startswith('-'):
            key = line.rstrip(':').strip()
            current_dict[key] = {}
            current_dict = result[key] if key in result else {}
            continue
            
        # Handle key-value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle different value types
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
            elif value.startswith('{') and value.endswith('}'):
                # Simple dict parsing like {BTCUSDT: 2}
                dict_content = value[1:-1].strip()
                if dict_content:
                    parts = dict_content.split(':')
                    if len(parts) == 2:
                        dict_key = parts[0].strip()
                        dict_val = parts[1].strip()
                        if dict_val.isdigit():
                            dict_val = int(dict_val)
                        elif dict_val.replace('.', '').isdigit():
                            dict_val = float(dict_val)
                        value = {dict_key: dict_val}
            elif value.startswith('[') and value.endswith(']'):
                # Simple list parsing
                list_content = value[1:-1].strip()
                if list_content:
                    items = [item.strip() for item in list_content.split(',')]
                    value = items
                else:
                    value = []
            
            current_dict[key] = value
    
    return result

from ..tools import register
from ..schemas.base import ToolResult, Meta
from ..schemas.risk import RiskCheckIn, RiskCheckOut, RiskViolation
from ..utils.hashing import hash_inputs, hash_code
from ..utils.serialization import convert_timestamps


@register("risk.check_limits")
def check_limits(**kwargs) -> ToolResult:
    """
    Check positions and orders against risk policy limits
    
    Args:
        **kwargs: Parameters matching RiskCheckIn schema
        
    Returns:
        ToolResult with RiskCheckOut containing violations and allow flag
    """
    try:
        params = RiskCheckIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    # Parse policy YAML
    try:
        policy = simple_yaml_parse(params.policy_yaml)
    except Exception as e:
        return ToolResult.error([f"Invalid policy YAML: {e}"])
    
    violations = []
    
    # Calculate current gross exposure
    gross_exposure = 0.0
    position_by_symbol = {}
    
    for pos in params.positions:
        symbol = pos.get('symbol', '')
        qty = abs(pos.get('qty', 0))
        price = params.prices.get(symbol, pos.get('price', 0))
        exposure = qty * price
        gross_exposure += exposure
        
        if symbol in position_by_symbol:
            position_by_symbol[symbol] += qty
        else:
            position_by_symbol[symbol] = qty
    
    # Check leverage limit
    if 'max_leverage' in policy and params.equity > 0:
        leverage = gross_exposure / params.equity
        max_leverage = policy['max_leverage']
        if leverage > max_leverage:
            severity = policy.get('severity', {}).get('leverage', 'hard')
            violations.append(RiskViolation(
                code="LEVERAGE_LIMIT",
                severity=severity,
                message=f"{leverage:.1f}x exceeds max {max_leverage}x",
                subject={
                    "metric": "leverage",
                    "value": leverage,
                    "limit": max_leverage
                }
            ))
    
    # Check gross exposure limit
    if 'max_gross_exposure' in policy:
        max_exposure = policy['max_gross_exposure']
        if gross_exposure > max_exposure:
            severity = policy.get('severity', {}).get('exposure', 'hard')
            violations.append(RiskViolation(
                code="EXPOSURE_LIMIT",
                severity=severity,
                message=f"Gross exposure ${gross_exposure:,.0f} exceeds max ${max_exposure:,.0f}",
                subject={
                    "metric": "gross_exposure",
                    "value": gross_exposure,
                    "limit": max_exposure
                }
            ))
    
    # Check orders
    for order in params.orders:
        symbol = order.get('symbol', '')
        side = order.get('side', 'buy')
        qty = abs(order.get('qty', 0))
        price = params.prices.get(symbol, order.get('price', params.prices.get(symbol, 0)))
        
        # Check denylist
        if 'denylist' in policy and symbol in policy['denylist']:
            violations.append(RiskViolation(
                code="DENYLIST",
                severity="hard",
                message=f"Symbol {symbol} is on denylist",
                subject={
                    "symbol": symbol,
                    "list": "denylist"
                }
            ))
        
        # Check allowlist
        if 'allowlist' in policy and policy['allowlist'] and symbol not in policy['allowlist']:
            violations.append(RiskViolation(
                code="ALLOWLIST",
                severity="hard",
                message=f"Symbol {symbol} is not on allowlist",
                subject={
                    "symbol": symbol,
                    "list": "allowlist"
                }
            ))
        
        # Check single order notional
        notional = qty * price
        if 'max_single_order_notional' in policy:
            max_notional = policy['max_single_order_notional']
            if notional > max_notional:
                severity = policy.get('severity', {}).get('single_notional', 'soft')
                violations.append(RiskViolation(
                    code="SINGLE_NOTIONAL",
                    severity=severity,
                    message=f"Order notional ${notional:,.0f} exceeds max ${max_notional:,.0f}",
                    subject={
                        "metric": "notional",
                        "value": notional,
                        "limit": max_notional,
                        "symbol": symbol
                    }
                ))
        
        # Check order quantity limit
        if 'max_order_qty' in policy:
            max_qty = policy['max_order_qty']
            if qty > max_qty:
                violations.append(RiskViolation(
                    code="ORDER_QTY",
                    severity="soft",
                    message=f"Order qty {qty} exceeds max {max_qty}",
                    subject={
                        "metric": "qty",
                        "value": qty,
                        "limit": max_qty,
                        "symbol": symbol
                    }
                ))
        
        # Check per-symbol position limit
        if 'max_position_per_symbol' in policy:
            symbol_limits = policy['max_position_per_symbol']
            if symbol in symbol_limits:
                max_pos = symbol_limits[symbol]
                current_pos = position_by_symbol.get(symbol, 0)
                
                # Calculate new position after order
                if side == 'buy':
                    new_pos = current_pos + qty
                else:
                    new_pos = current_pos - qty
                
                if abs(new_pos) > max_pos:
                    violations.append(RiskViolation(
                        code="POSITION_LIMIT",
                        severity="hard",
                        message=f"Position {abs(new_pos):.2f} would exceed max {max_pos} for {symbol}",
                        subject={
                            "metric": "position",
                            "current": current_pos,
                            "order_qty": qty,
                            "new_position": new_pos,
                            "limit": max_pos,
                            "symbol": symbol
                        }
                    ))
    
    # Check daily loss limit (would need P&L tracking - simplified here)
    if 'max_daily_loss' in policy:
        # This would normally check realized + unrealized P&L
        # For now, just include as a placeholder
        pass
    
    # Determine if action is allowed (no hard violations)
    hard_violations = [v for v in violations if v.severity == "hard"]
    allow = len(hard_violations) == 0
    
    # Generate audit hash
    audit_content = json.dumps({
        'inputs': params.model_dump(),
        'violations': [v.model_dump() for v in violations],
        'allow': allow
    }, sort_keys=True)
    audit_hash = hashlib.sha256(audit_content.encode()).hexdigest()
    
    # Create output
    output = RiskCheckOut(
        allow=allow,
        violations=violations,
        audit_hash=audit_hash
    )
    
    # Create metadata
    meta = Meta(
        tool="risk.check_limits",
        version="1.0.0",
        timestamp=None,
        provenance={
            "inputs_hash": hash_inputs(params.model_dump()),
            "code_hash": hash_code(check_limits),
            "violation_count": len(violations),
            "hard_violations": len(hard_violations)
        }
    )
    
    result = ToolResult(
        ok=True,
        data=output.model_dump(),
        meta=meta.model_dump()
    )
    
    return convert_timestamps(result)