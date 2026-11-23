"""
NĀḌĪ-Q Phase 14A: Biological Relational Quantum Feature Map.
Encodes multimodal biological features into 6-qubit quantum states using:
1. Feature standardization (fit strictly on training folds).
2. Tanh angle bounding: theta_k = scale * tanh(x_scaled_k).
3. Biological interaction terms: r_jk = theta_j * theta_k.
4. Unitary circuit: H^{otimes n} -> R_Y(theta) -> prod_{(j,k)} exp(-i r_jk Z_j Z_k) -> R_Z(theta).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import List, Optional, Tuple


class FeatureStandardizer:
    """
    Standardizer that fits strictly on the training fold and transforms features.
    Guarantees no test-patient statistics leak into feature processing.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureStandardizer":
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        # Avoid division by zero on constant features
        self.scale_[self.scale_ < self.epsilon] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("FeatureStandardizer must be fitted before transform.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / (self.scale_ + self.epsilon)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def compute_angles(x_scaled: np.ndarray, angle_scale: float = np.pi) -> np.ndarray:
    """
    Map scaled features to bounded rotation angles in [-angle_scale, angle_scale].
    theta_k = angle_scale * tanh(x_scaled_k).
    """
    return angle_scale * np.tanh(np.asarray(x_scaled, dtype=np.float64))


def compute_interaction_terms(
    theta: np.ndarray,
    interaction_pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute pairwise relational interaction terms r_jk = theta_j * theta_k.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim == 1:
        return np.array([theta_arr[j] * theta_arr[k] for j, k in interaction_pairs], dtype=np.float64)
    elif theta_arr.ndim == 2:
        return np.column_stack([
            theta_arr[:, j] * theta_arr[:, k] for j, k in interaction_pairs
        ])
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {theta_arr.shape}")


def biological_feature_map(
    theta: np.ndarray,
    interaction_pairs: List[Tuple[int, int]],
    entanglement_type: str = "biological",
) -> None:
    """
    Execute the Biological Relational Feature Map unitary U_phi(x).
    
    Unitary sequence:
    1. H^{otimes n} on all qubits
    2. R_Y(theta_k) on each qubit k
    3. Pairwise biological entanglement: exp(-i r_jk Z_j Z_k) via IsingZZ(2 * r_jk)
    4. R_Z(theta_k) on each qubit k
    """
    n_qubits = len(theta)

    # 1. Uniform superposition
    for k in range(n_qubits):
        qml.Hadamard(wires=k)

    # 2. First single-qubit rotation layer (R_Y)
    for k in range(n_qubits):
        qml.RY(theta[k], wires=k)

    # 3. Two-qubit relational entanglement layer
    if entanglement_type != "none":
        for j, k in interaction_pairs:
            r_jk = theta[j] * theta[k]
            # exp(-i r_jk Z_j Z_k) corresponds to IsingZZ(2 * r_jk)
            qml.IsingZZ(2.0 * float(r_jk), wires=[j, k])

    # 4. Second single-qubit rotation layer (R_Z)
    for k in range(n_qubits):
        qml.RZ(theta[k], wires=k)


def adjoint_biological_feature_map(
    theta: np.ndarray,
    interaction_pairs: List[Tuple[int, int]],
    entanglement_type: str = "biological",
) -> None:
    """
    Execute the adjoint (inverse) unitary U_phi(x)^dagger.
    
    Reversed sequence with inverted gate angles:
    1. R_Z(-theta_k) on each qubit k
    2. Inverted pairwise biological entanglement in reverse order: exp(+i r_jk Z_j Z_k)
    3. R_Y(-theta_k) on each qubit k
    4. H^{otimes n} on all qubits
    """
    n_qubits = len(theta)

    # 1. Inverted R_Z
    for k in range(n_qubits):
        qml.RZ(-theta[k], wires=k)

    # 2. Inverted entanglement in reverse order
    if entanglement_type != "none":
        for j, k in reversed(interaction_pairs):
            r_jk = theta[j] * theta[k]
            qml.IsingZZ(-2.0 * float(r_jk), wires=[j, k])

    # 3. Inverted R_Y
    for k in range(n_qubits):
        qml.RY(-theta[k], wires=k)

    # 4. Hadamard (self-inverse)
    for k in range(n_qubits):
        qml.Hadamard(wires=k)
