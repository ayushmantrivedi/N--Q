"""
Variance Filtering for Nāḍī-Q.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nadiq.preprocessing.base import BaseTransformer


class VarianceFilter(BaseTransformer):
    """
    Removes features with variance below a specified threshold.
    """

    def __init__(self, threshold: float = 1e-4) -> None:
        super().__init__()
        self.threshold = threshold
        self.variances_: np.ndarray | None = None
        self.retained_indices_: np.ndarray | None = None
        self.original_n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "VarianceFilter":
        self.original_n_features_ = X.shape[1]
        self.variances_ = np.nanvar(X, axis=0)
        
        # Identify features to keep
        self.retained_indices_ = np.where(self.variances_ > self.threshold)[0]
        
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before calling transform.")
        assert self.retained_indices_ is not None and self.original_n_features_ is not None
        
        if X.shape[1] != self.original_n_features_:
            raise ValueError(f"Expected {self.original_n_features_} features, got {X.shape[1]}")
            
        return X[:, self.retained_indices_]

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transforming a variance filter pads the removed features with zeros.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before calling inverse_transform.")
        assert self.retained_indices_ is not None and self.original_n_features_ is not None
        
        if X.shape[1] != len(self.retained_indices_):
            raise ValueError(f"Expected {len(self.retained_indices_)} features, got {X.shape[1]}")
            
        X_inv = np.zeros((X.shape[0], self.original_n_features_), dtype=X.dtype)
        X_inv[:, self.retained_indices_] = X
        
        return X_inv

    def get_params(self) -> dict[str, Any]:
        if not self.is_fitted_:
            return {"threshold": self.threshold}
        return {
            "threshold": self.threshold,
            "variances": self.variances_,
            "retained_indices": self.retained_indices_,
        }
