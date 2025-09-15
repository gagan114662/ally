#!/usr/bin/env python3
"""
Alpha Vantage Live Demo - Shows expected behavior with receipts
This would be run with a real API key in live mode
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append('.')

def demo_offline_mode():
    """Demo Alpha Vantage in offline mode"""
    print("🔧 Alpha Vantage Offline Mode Demo")
    print()
    
    try:
        from ally.tools.data import data_load_ohlcv
        
        # Offline mode demo
        os.environ['ALLY_LIVE'] = '0'
        
        result = data_load_ohlcv(
            symbols=["AAPL"],
            interval="1d", 
            start="2024-01-01",
            end="2024-01-05",
            source="alpha_vantage",
            live=False
        )
        
        print(f"✅ Result: {result.ok}")
        print(f"📊 Symbols loaded: {result.data['summary']['symbols_loaded']}")
        print(f"📈 Total rows: {result.data['summary']['total_rows']}")
        print(f"🏷️  Provider: {result.data['summary']['provider']}")
        print(f"🎯 Live mode: {result.data['panel']['metadata']['live_mode']}")
        print(f"🧾 Receipt hashes: {len(result.data['summary']['receipt_hashes'])} (offline = 0)")
        
        # Show provider info
        provider_info = result.data['panel']['metadata']['provider_info']
        if 'AAPL' in provider_info:
            print(f"📡 AAPL provider: {provider_info['AAPL']['provider']}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_live_mode_gating():
    """Demo live mode gating (will fail without ALLY_LIVE=1)"""
    print("🚦 Alpha Vantage Live Mode Gating Demo")
    print()
    
    try:
        from ally.tools.data import data_load_ohlcv
        
        # This should fail due to gating
        os.environ['ALLY_LIVE'] = '0'  # Keep offline
        
        result = data_load_ohlcv(
            symbols=["AAPL"],
            interval="1d",
            start="2024-01-01",
            end="2024-01-05",
            source="alpha_vantage", 
            live=True,  # Request live mode
            api_key="demo_api_key"
        )
        
        if result.ok:
            print("❌ Expected gating to block this request")
            return False
        else:
            print("✅ Gating correctly blocked live mode request")
            print(f"🔒 Error: {result.errors[0]}")
            return True
            
    except Exception as e:
        if "ALLY_LIVE" in str(e):
            print("✅ Gating correctly blocked live mode request") 
            print(f"🔒 Exception: {e}")
            return True
        else:
            print(f"❌ Unexpected error: {e}")
            return False


def simulate_offline_receipt_demo():
    """Demo offline receipt hashing (mock payload only - no network)"""
    print("🧾 Offline Receipt Demo (Mock Payload)")
    print("ℹ️  Mode: offline (no network; mock payload hashed)")
    print()
    
    try:
        from ally.utils.receipts import store_tool_receipt
        from ally.utils.hashing import hash_inputs
        
        # Simulate Alpha Vantage API response
        simulated_response = {
            "Meta Data": {
                "1. Information": "Intraday (1min) open, high, low, close prices and volume",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2024-01-05 19:55:00",
                "4. Interval": "1min",
                "5. Output Size": "Full size",
                "6. Time Zone": "US/Eastern"
            },
            "Time Series (1min)": {
                "2024-01-05 19:55:00": {
                    "1. open": "182.01",
                    "2. high": "182.15", 
                    "3. low": "182.00",
                    "4. close": "182.12",
                    "5. volume": "15423"
                },
                "2024-01-05 19:54:00": {
                    "1. open": "182.05",
                    "2. high": "182.10",
                    "3. low": "181.98",
                    "4. close": "182.01", 
                    "5. volume": "12891"
                }
            }
        }
        
        # Create tool inputs
        tool_inputs = {
            "symbol": "AAPL",
            "interval": "1min",
            "start": "2024-01-05",
            "end": "2024-01-05",
            "source": "alpha_vantage"
        }
        
        # Store receipt for simulated response
        receipt_hash = store_tool_receipt(
            tool_name="data.alpha_vantage.fetch_symbol",
            inputs=tool_inputs,
            raw_payload=simulated_response
        )
        
        print("✅ Receipt stored successfully")
        print(f"🆔 Receipt hash: {receipt_hash}")
        print(f"🔢 Args hash: {hash_inputs(tool_inputs)[:8]}")
        print(f"📅 Timestamp: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
        
        # Show what PROOF line would look like
        args_hash = hash_inputs(tool_inputs, algorithm="sha256")[:8]
        proof_line = f"PROOF:run:data.alpha_vantage.fetch_symbol@{args_hash}:{receipt_hash}"
        print(f"🏷️  PROOF line: {proof_line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_demo():
    """Run all demo scenarios"""
    print("🚀 Alpha Vantage Integration Demo")
    print("=" * 50)
    print()
    
    success_count = 0
    
    # Test offline mode
    if demo_offline_mode():
        success_count += 1
    
    # Test gating
    if demo_live_mode_gating():
        success_count += 1
    
    # Demo offline receipts  
    if simulate_offline_receipt_demo():
        success_count += 1
    
    print("=" * 50)
    print(f"📊 Demo Results: {success_count}/3 scenarios passed")
    
    if success_count == 3:
        print("🎉 Alpha Vantage integration working correctly!")
        print()
        print("🔑 To use with real data:")
        print("1. Get Alpha Vantage API key from https://www.alphavantage.co/")
        print("2. Set ALPHA_VANTAGE_API_KEY=your_real_key")
        print("3. Set ALLY_LIVE=1")
        print("4. Run: ally run data.load_ohlcv '{...\"live\":true}'")
        return True
    else:
        print("❌ Some demo scenarios failed")
        return False


if __name__ == "__main__":
    success = run_demo()
    exit(0 if success else 1)