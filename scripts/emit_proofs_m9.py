#!/usr/bin/env python3
"""
M9 Orchestrator Proof Generation Script
Runs orchestrator with portfolio integration and emits PROOF lines
"""

from ally.orchestrator.run import orchestrator_run

def main():
    """Run M9 orchestrator and emit proofs"""
    print("=== M9 Orchestrator Integration with Portfolio ===")
    
    # Run orchestrator with portfolio integration
    summary = orchestrator_run(
        symbols=["SPY", "QQQ", "TLT", "GLD"],
        start_date="2020-01-01", 
        end_date="2020-01-05"
    )
    
    # Additional orchestrator-specific proofs
    print(f"PROOF:ORCH_SYMBOLS: {len(summary.symbols)}")
    print(f"PROOF:ORCH_SUCCESS: {summary.execution_success}")
    
    # Portfolio proofs are already emitted by orchestrator_run
    print(f"=== Portfolio Weights ===")
    for symbol, weight in summary.port_weights.items():
        print(f"{symbol}: {weight:.4f}")
    
    print(f"=== Summary ===")
    print(f"Symbols: {summary.symbols}")
    print(f"Portfolio weights sum: {summary.port_weights_sum}")
    print(f"Attribution OK: {summary.attribution_ok}")
    print(f"Deterministic hash: {summary.port_det_hash}")

if __name__ == "__main__":
    main()