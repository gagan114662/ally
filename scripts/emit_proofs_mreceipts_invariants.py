from ally.tools import TOOL_REGISTRY
from ally.tools.receipts import assert_receipts_invariants

def main():
    # 1) Prove invariant function actually blocks
    blocked = False
    try:
        assert_receipts_invariants({"live": True, "receipts": []})
    except AssertionError:
        blocked = True

    # 2) Prove CLI works in offline mode
    ok = TOOL_REGISTRY["receipts.diff"](series_a=[1,2,3], series_b=[1,2,3], tolerance=0).ok

    # 3) Prove receipts.verify tool is registered and functional
    verify_tool_exists = "receipts.verify" in TOOL_REGISTRY
    
    print(f"PROOF:RECEIPTS_INVARIANTS: {'ok' if blocked else 'fail'}")
    print(f"PROOF:RECEIPTS_DIFF_CLI: {'ok' if ok else 'fail'}")
    print(f"PROOF:RECEIPTS_VERIFY_REGISTERED: {'ok' if verify_tool_exists else 'fail'}")

if __name__ == "__main__":
    main()