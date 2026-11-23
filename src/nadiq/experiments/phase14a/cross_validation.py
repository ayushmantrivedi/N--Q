"""
NĀḌĪ-Q Phase 14A: Stratified Multi-Seed Cross-Validation Protocol.
Executes strict 5-fold stratified cross-validation across 3 seeds [42, 123, 456]
with complete isolation of fold test splits.
"""

from __future__ import annotations

from typing import Generator, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedMultiSeedCV:
    """
    Generator yielding (seed, fold_idx, train_idx, test_idx) for 3 seeds x 5 folds = 15 evaluation splits.
    """

    def __init__(
        self,
        n_splits: int = 5,
        seeds: Optional[List[int]] = None,
    ) -> None:
        self.n_splits = n_splits
        self.seeds = seeds or [42, 123, 456]

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Generator[Tuple[int, int, np.ndarray, np.ndarray], None, None]:
        """
        Yields:
            seed: int
            fold_idx: int (0 to n_splits - 1)
            train_idx: np.ndarray
            test_idx: np.ndarray
        """
        y_arr = np.asarray(y, dtype=int)
        for seed in self.seeds:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_arr)):
                yield seed, fold_idx, train_idx, test_idx
