import os
import random
import time
import numpy as np


def set_determinism(seed: int = 1337) -> None:
    # Global seeds
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Timezone & locale
    os.environ.setdefault("TZ", "UTC")
    try:
        time.tzset()
    except Exception:
        pass  # windows

    # Single-threaded math for stable FP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Optional: torch determinism if present
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass