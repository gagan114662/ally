from __future__ import annotations
import re

def add_algorithmimports(text: str) -> str:
    """Add AlgorithmImports and ensure QCAlgorithm base class"""
    if "from AlgorithmImports import *" not in text:
        text = "from AlgorithmImports import *\n" + text
    # ensure class extends QCAlgorithm
    text = re.sub(r"class\s+(\w+)\s*\((.*?)\):", lambda m: f"class {m.group(1)}(QCAlgorithm):", text, count=1)
    return text

def fix_ondata_signature(text: str) -> str:
    """Fix OnData method signature to proper format"""
    return re.sub(r"def\s+OnData\s*\(.*\):", "def OnData(self, data: Slice):", text)

def replace_now_with_self_time(text: str) -> str:
    """Replace datetime.now() and similar with self.Time"""
    text = re.sub(r"(datetime|pd\.Timestamp)\.now\(\)", "self.Time", text)
    return re.sub(r"\.now\(\)", "self.Time", text)

def replace_transactions_orders(text: str) -> str:
    """Replace Transactions.Orders property with GetOrders() method"""
    return text.replace(".Transactions.Orders", ".Transactions.GetOrders()")

def schedule_everyday(text: str) -> str:
    """Fix DateRules.Every() with invalid arguments"""
    return re.sub(r"DateRules\.Every\s*\([^)]*\)", "DateRules.EveryDay()", text)

def normalize_bnb_to_usdt_binance(text: str) -> str:
    """Map unsupported crypto pairs to supported ones"""
    text = text.replace('AddCrypto("BNBUSD",', 'AddCrypto("BNBUSDT",')
    return text.replace("Market.Bitfinex", "Market.Binance")