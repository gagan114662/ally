from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Tuple

@dataclass
class WalkForwardConfig:
    window_train: int           # bars in train
    window_test: int            # bars in test
    mode: str = "expanding"     # "expanding" or "sliding"
    step: int = None            # advance step; default=window_test

def make_walkforward_splits(index: pd.DatetimeIndex, cfg: WalkForwardConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(index)
    step = cfg.step or cfg.window_test
    splits = []
    start_train, end_train = 0, cfg.window_train
    while True:
        start_test = end_train
        end_test = min(start_test + cfg.window_test, n)
        if end_test - start_test < cfg.window_test: break
        train_idx = np.arange(0, end_train) if cfg.mode == "expanding" else np.arange(start_train, end_train)
        test_idx = np.arange(start_test, end_test)
        splits.append((train_idx, test_idx))
        if cfg.mode == "sliding":
            start_train += step
        end_train += step
        if end_train >= n: break
    return splits