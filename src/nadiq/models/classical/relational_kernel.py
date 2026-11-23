"""
NĀḌĪ-Q Phase 14A: Matched Classical Relational Kernel.
Builds explicit classical relational representations:
    phi_classical(x) = [theta_0, ..., theta_5, r_0, ..., r_m]
where r_jk = theta_j * theta_k for biological interaction pairs.
Computes RBF or normalized linear kernels over this identical relational feature space.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from nadiq.models.quantum.brqk.feature_map import (
    FeatureStandardizer,
    compute_angles,
    compute_interaction_terms,
)


class ClassicalRelationalKernel:
    """
    Matched Classical Relational Kernel serving as direct classical control for BRQK.

    Features:
    - Standardized input features mapped to bounded tanh angles theta in [-pi, pi].
    - Explicit biological interaction terms r_jk = theta_j * theta_k matching BRQK ZZ couplings.
    - Full relational feature vector phi in R^{d + |E|}.
    - Kernel evaluation: RBF or Cosine/Normalized Linear.
    """

    def __init__(
        self,
        interaction_pairs: Optional[List[Tuple[int, int]]] = None,
        angle_scale: float = np.pi,
        kernel_type: str = "rbf",
        gamma: Optional[float] = None,
        epsilon: float = 1e-6,
    ) -> None:
        self.interaction_pairs = interaction_pairs or [
            (0, 2), (1, 2), (0, 3), (1, 3),
            (2, 4), (3, 4), (3, 5), (0, 4),
        ]
        self.angle_scale = angle_scale
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.epsilon = epsilon
        self.standardizer = FeatureStandardizer(epsilon=self.epsilon)
        self.is_fitted = False
        self._learned_gamma: Optional[float] = None

    def fit(self, X: np.ndarray) -> "ClassicalRelationalKernel":
        """
        Fit standardizer strictly on training data and compute default gamma.
        """
        self.standardizer.fit(X)
        self.is_fitted = True

        # Compute default gamma if RBF
        if self.gamma is None and self.kernel_type == "rbf":
            phi_train = self.compute_relational_features(X)
            var = float(np.var(phi_train))
            n_features = phi_train.shape[1]
            self._learned_gamma = 1.0 / (n_features * var + self.epsilon) if var > 0 else 1.0 / n_features
        else:
            self._learned_gamma = self.gamma

        return self

    def compute_relational_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw inputs into explicit relational features [theta, r].
        """
        if not self.is_fitted:
            raise ValueError("ClassicalRelationalKernel must be fitted before transforming.")
        X_scaled = self.standardizer.transform(X)
        thetas = compute_angles(X_scaled, angle_scale=self.angle_scale)
        interactions = compute_interaction_terms(thetas, self.interaction_pairs)
        return np.column_stack([thetas, interactions])

    def compute_gram_matrix(
        self,
        X_train: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute classical relational Gram matrix.
        If X_eval is None: returns K(X_train, X_train) of shape (N_tr, N_tr).
        If X_eval is provided: returns K(X_eval, X_train) of shape (N_eval, N_tr).
        """
        if not self.is_fitted:
            self.fit(X_train)

        phi_train = self.compute_relational_features(X_train)

        if X_eval is None:
            if self.kernel_type == "rbf":
                gamma = self._learned_gamma or 0.1
                # Squared Euclidean distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u^T v
                sq_norm = np.sum(phi_train ** 2, axis=1, keepdims=True)
                dists_sq = sq_norm + sq_norm.T - 2.0 * np.dot(phi_train, phi_train.T)
                dists_sq = np.maximum(0.0, dists_sq)
                K = np.exp(-gamma * dists_sq)
                np.fill_diagonal(K, 1.0)
                return 0.5 * (K + K.T)
            elif self.kernel_type == "linear":
                # Normalized cosine similarity to guarantee unit diagonal
                norms = np.linalg.norm(phi_train, axis=1, keepdims=True) + self.epsilon
                phi_norm = phi_train / norms
                K = np.dot(phi_norm, phi_norm.T)
                np.fill_diagonal(K, 1.0)
                return 0.5 * (K + K.T)
            else:
                raise ValueError(f"Unknown kernel_type '{self.kernel_type}'. Supported: 'rbf', 'linear'.")
        else:
            phi_eval = self.compute_relational_features(X_eval)
            if self.kernel_type == "rbf":
                gamma = self._learned_gamma or 0.1
                sq_norm_eval = np.sum(phi_eval ** 2, axis=1, keepdims=True)
                sq_norm_train = np.sum(phi_train ** 2, axis=1, keepdims=True)
                dists_sq = sq_norm_eval + sq_norm_train.T - 2.0 * np.dot(phi_eval, phi_train.T)
                dists_sq = np.maximum(0.0, dists_sq)
                return np.exp(-gamma * dists_sq)
            elif self.kernel_type == "linear":
                norms_eval = np.linalg.norm(phi_eval, axis=1, keepdims=True) + self.epsilon
                norms_train = np.linalg.norm(phi_train, axis=1, keepdims=True) + self.epsilon
                phi_eval_norm = phi_eval / norms_eval
                phi_train_norm = phi_train / norms_train
                return np.dot(phi_eval_norm, phi_train_norm.T)
            else:
                raise ValueError(f"Unknown kernel_type '{self.kernel_type}'.")
