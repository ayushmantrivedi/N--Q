"""
NĀḌĪ-Q Classical Control Baselines for Fair Quantum Comparison.
Implements Linear PCA, Non-linear Kernel PCA, Random Projections, and Matched Fourier Expansions.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection


class ClassicalTrigonometricControl(BaseEstimator, TransformerMixin):
    """
    Classical Order-1 and Order-2 Trigonometric / Fourier Basis Expansion.
    Evaluates whether non-linear quantum advantage exceeds classical harmonic expansions.
    """
    def __init__(self, target_dim: int = 23):
        self.target_dim = target_dim

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (N, d) assumed rescaled to [0, pi]
        d = X.shape[1]
        features = []

        # Order 1 harmonics: cos(x_i), sin(x_i)
        for i in range(d):
            features.append(np.cos(X[:, i]))
            features.append(np.sin(X[:, i]))

        # Order 2 harmonics: cos(2*x_i)
        for i in range(d):
            features.append(np.cos(2.0 * X[:, i]))

        # Bivariate cross-harmonics: cos(x_i - x_{i+1})
        for i in range(d - 1):
            features.append(np.cos(X[:, i] - X[:, i + 1]))

        stacked = np.column_stack(features)
        if stacked.shape[1] >= self.target_dim:
            return stacked[:, :self.target_dim]
        else:
            pad = np.zeros((len(X), self.target_dim - stacked.shape[1]))
            return np.column_stack([stacked, pad])
