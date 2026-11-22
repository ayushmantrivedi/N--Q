"""
Median Imputation for Nāḍī-Q.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nadiq.preprocessing.base import BaseTransformer


class MedianImputer(BaseTransformer):
    """
    Imputes missing values (NaN) with the median of each feature column
    computed on the training set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.medians_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MedianImputer":
        # Compute median ignoring NaNs
        # If an entire column is NaN, the median will be NaN. We fill with 0.0 in that case.
        medians = np.nanmedian(X, axis=0)
        medians[np.isnan(medians)] = 0.0
        
        self.medians_ = medians
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before calling transform.")
        assert self.medians_ is not None

        X_transformed = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X_transformed[:, i])
            X_transformed[mask, i] = self.medians_[i]
            
        return X_transformed

    def get_params(self) -> dict[str, Any]:
        if not self.is_fitted_:
            return {}
        return {
            "medians": self.medians_,
        }
