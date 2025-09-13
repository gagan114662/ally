"""
QC Runtime Assertions - Indicator readiness, order fills, warmup guards
"""

from typing import List, Dict, Any


def generate_assert_helpers() -> str:
    """
    Generate runtime assertion helper functions for QC algorithms
    
    Returns:
        Python code string with assertion helpers
    """
    return '''
# Runtime Assertion Helpers
def assert_indicator_ready(self, indicator, name="indicator"):
    """Assert indicator is ready with sufficient data"""
    if not hasattr(indicator, 'IsReady'):
        self.Debug(f"WARNING: {name} has no IsReady property")
        return
    
    if not indicator.IsReady:
        msg = f"ASSERT_TRIP: {name} not ready at {self.Time}"
        self.Debug(msg)
        self.Log(msg)
        return False
    return True

def assert_warmup_complete(self):
    """Assert warmup period is complete"""
    if self.IsWarmingUp:
        msg = f"ASSERT_TRIP: Still warming up at {self.Time}"
        self.Debug(msg)
        return False
    return True

def assert_orders_filled(self, symbol, expected_quantity):
    """Assert orders filled as expected"""
    holdings = self.Portfolio[symbol].Quantity
    if abs(holdings - expected_quantity) > 0.01:
        msg = f"ASSERT_TRIP: Expected {expected_quantity} {symbol}, got {holdings}"
        self.Debug(msg)
        self.Log(msg)
        return False
    return True

def assert_history_available(self, symbol, bars=1):
    """Assert history data is available"""
    try:
        history = self.History(symbol, bars, Resolution.Daily)
        if history.empty:
            msg = f"ASSERT_TRIP: No history for {symbol} at {self.Time}"
            self.Debug(msg)
            self.Log(msg)
            return False
        return True
    except Exception as e:
        msg = f"ASSERT_TRIP: History fetch failed for {symbol}: {e}"
        self.Debug(msg)
        self.Log(msg)
        return False

def assert_portfolio_value(self, min_value):
    """Assert portfolio value above minimum"""
    total_value = self.Portfolio.TotalPortfolioValue
    if total_value < min_value:
        msg = f"ASSERT_TRIP: Portfolio value {total_value} below minimum {min_value}"
        self.Debug(msg)
        self.Log(msg)
        return False
    return True

# Bind assertion helpers to self
self.assert_indicator_ready = assert_indicator_ready.__get__(self, self.__class__)
self.assert_warmup_complete = assert_warmup_complete.__get__(self, self.__class__)
self.assert_orders_filled = assert_orders_filled.__get__(self, self.__class__)
self.assert_history_available = assert_history_available.__get__(self, self.__class__)
self.assert_portfolio_value = assert_portfolio_value.__get__(self, self.__class__)
'''


def inject_asserts_into_template(template_content: str) -> str:
    """
    Inject assertion helpers into QC algorithm template
    
    Args:
        template_content: Original template content
        
    Returns:
        Modified template with assertions
    """
    helpers = generate_assert_helpers()
    
    # Find Initialize method and inject helpers
    lines = template_content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Inject after Initialize method definition
        if 'def Initialize(self):' in line:
            # Add helpers after docstring if present
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''")):
                j += 1
            # Insert helpers with proper indentation
            indent = '        '  # 2 levels of indentation
            helper_lines = helpers.split('\n')
            for helper_line in helper_lines:
                if helper_line:
                    new_lines.append(indent + helper_line)
                else:
                    new_lines.append('')
    
    return '\n'.join(new_lines)


def count_assert_trips(log_content: str) -> Dict[str, Any]:
    """
    Count assertion trips in LEAN log output
    
    Args:
        log_content: LEAN log content
        
    Returns:
        Dictionary with assert trip counts and details
    """
    import re
    
    trips = []
    total_trips = 0
    
    # Find all ASSERT_TRIP lines
    pattern = r'ASSERT_TRIP: (.+)'
    matches = re.findall(pattern, log_content)
    
    for match in matches:
        trips.append(match)
        total_trips += 1
    
    # Categorize trips
    categories = {
        'indicator_not_ready': 0,
        'warmup_incomplete': 0,
        'order_mismatch': 0,
        'history_unavailable': 0,
        'portfolio_below_min': 0,
        'other': 0
    }
    
    for trip in trips:
        if 'not ready' in trip:
            categories['indicator_not_ready'] += 1
        elif 'warming up' in trip:
            categories['warmup_incomplete'] += 1
        elif 'Expected' in trip and 'got' in trip:
            categories['order_mismatch'] += 1
        elif 'No history' in trip or 'History fetch failed' in trip:
            categories['history_unavailable'] += 1
        elif 'Portfolio value' in trip:
            categories['portfolio_below_min'] += 1
        else:
            categories['other'] += 1
    
    return {
        'total_trips': total_trips,
        'categories': categories,
        'trip_messages': trips[:10]  # First 10 messages for debugging
    }