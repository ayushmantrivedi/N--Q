"""
Z-Score Normalization for Nāḍī-Q.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nadiq.preprocessing.base import BaseTransformer


class ZScoreNormalizer(BaseTransformer):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ZScoreNormalizer":
        # Ignore NaNs during fit
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        
        # Handle zero variance
        self.scale_[self.scale_ < self.epsilon] = 1.0
        
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before calling transform.")
        assert self.mean_ is not None and self.scale_ is not None
        
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before calling inverse_transform.")
        assert self.mean_ is not None and self.scale_ is not None
        
        return (X * self.scale_) + self.mean_

    def get_params(self) -> dict[str, Any]:
        if not self.is_fitted_:
            return {"epsilon": self.epsilon}
        return {
            "epsilon": self.epsilon,
            "mean": self.mean_,
            "scale": self.scale_,
        }
