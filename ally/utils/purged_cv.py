from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

@dataclass
class PurgedKFold:
    n_splits: int = 5
    embargo_frac: float = 0.01  # 1% embargo by default

    def split(self, index: pd.DatetimeIndex) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(index)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(np.r_[0, fold_sizes[:-1]])
        for i, start in enumerate(starts):
            stop = start + fold_sizes[i]
            test_idx = np.arange(start, stop)

            # embargo: remove neighbors around test set from train
            emb = max(1, int(self.embargo_frac * n))
            left_cut = max(0, start - emb)
            right_cut = min(n, stop + emb)
            train_idx = np.r_[np.arange(0, left_cut), np.arange(right_cut, n)]
            yield train_idx, test_idx