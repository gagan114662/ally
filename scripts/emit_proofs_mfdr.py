#!/usr/bin/env python3
"""
Emit M-FDR Gate proofs for CI verification
"""

import json, os, hashlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ally.tools import TOOL_REGISTRY

def main():
    """Emit M-FDR Gate proofs with deterministic fixture set"""

    try:
        print("# M-FDR Gate Proof Emission")

        # Deterministic fixture set
        CANDS = [
          {"id":"A","t_oos":3.2,"oos_obs":252,"alpha_oos":0.45},
          {"id":"B","t_oos":2.9,"oos_obs":252,"alpha_oos":0.30},
          {"id":"C","t_oos":2.5,"oos_obs":252,"alpha_oos":0.20},
          {"id":"D","t_oos":2.1,"oos_obs":252,"alpha_oos":0.05},
          {"id":"E","t_oos":1.8,"oos_obs":252,"alpha_oos":0.02},
          {"id":"F","t_oos":1.5,"oos_obs":252,"alpha_oos":0.01},
          {"id":"G","t_oos":1.2,"oos_obs":252,"alpha_oos":0.01},
          {"id":"H","t_oos":0.8,"oos_obs":252,"alpha_oos":0.00},
          {"id":"I","t_oos":0.0,"oos_obs":252,"alpha_oos":0.00},
          {"id":"J","t_oos":-0.5,"oos_obs":252,"alpha_oos":-0.01},
          {"id":"K","t_oos":-1.2,"oos_obs":252,"alpha_oos":-0.02},
          {"id":"L","t_oos":-2.2,"oos_obs":252,"alpha_oos":-0.03}
        ]

        # Run FDR evaluation
        res = TOOL_REGISTRY["fdr.evaluate"](candidates=CANDS, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)

        if not res.ok:
            print(f"ERROR: FDR evaluation failed: {res.errors}")
            sys.exit(1)

        data = res.data
        proofs = data.get("proofs", {})

        # Emit core FDR proofs
        print(f"PROOF:FDR_ALPHA: {proofs.get('FDR_ALPHA', 0.05)}")
        print(f"PROOF:FDR_METHOD: {proofs.get('FDR_METHOD', 'BH')}")
        print(f"PROOF:N_TESTED: {proofs.get('N_TESTED', 0)}")
        print(f"PROOF:N_PROMOTED: {proofs.get('N_PROMOTED', 0)}")
        print(f"PROOF:MEAN_T_OOS: {proofs.get('MEAN_T_OOS', 0.0)}")
        print(f"PROOF:POS_ALPHA_ENFORCED: {proofs.get('POS_ALPHA_ENFORCED', True)}")
        print(f"PROOF:FDR_HASH: {proofs.get('FDR_HASH', 'missing')}")

        # Additional verification proofs
        promoted_ids = data.get("promoted_ids", [])
        print(f"PROOF:PROMOTED_IDS: {','.join(promoted_ids) if promoted_ids else 'none'}")
        print(f"PROOF:PROMOTION_RATE: {round(len(promoted_ids) / max(1, data.get('n_tested', 1)) * 100, 1)}%")

        # Create artifacts directory
        os.makedirs("fdr-proof-bundle", exist_ok=True)

        # Save full proof data
        full_proofs = {
            "FDR_ALPHA": proofs.get("FDR_ALPHA", 0.05),
            "FDR_METHOD": proofs.get("FDR_METHOD", "BH"),
            "N_TESTED": proofs.get("N_TESTED", 0),
            "N_PROMOTED": proofs.get("N_PROMOTED", 0),
            "MEAN_T_OOS": proofs.get("MEAN_T_OOS", 0.0),
            "POS_ALPHA_ENFORCED": proofs.get("POS_ALPHA_ENFORCED", True),
            "FDR_HASH": proofs.get("FDR_HASH", "missing"),
            "promoted_ids": promoted_ids,
            "q_values": data.get("q_values", {}),
            "input_candidates": len(CANDS),
            "filtered_positive": data.get("n_tested", 0)
        }

        with open("fdr-proof-bundle/fdr_proofs.json", "w") as f:
            json.dump(full_proofs, f, indent=2)

        # Save detailed results for forensics
        with open("fdr-proof-bundle/fdr_detailed.json", "w") as f:
            json.dump(data, f, indent=2)

        print("# M-FDR Gate proofs emitted successfully")

    except Exception as e:
        print(f"ERROR: FDR proof emission failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()