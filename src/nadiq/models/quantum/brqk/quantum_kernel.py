"""
NĀḌĪ-Q Phase 14A: Biological Relational Quantum Kernel (BRQK).
Computes exact statevector quantum kernels and supports circuit overlap QNodes.
Guarantees unit diagonal, mathematical symmetry, and positive semi-definiteness via Schur product.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import List, Optional, Tuple, Union

from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.feature_map import (
    FeatureStandardizer,
    compute_angles,
    biological_feature_map,
    adjoint_biological_feature_map,
)


class BiologicalRelationalQuantumKernel:
    """
    Biological Relational Quantum Kernel (BRQK) implementation.

    Computes the quantum transition amplitude kernel:
        K_Q(x_i, x_j) = |<0^{otimes n}| U_phi(x_j)^dagger U_phi(x_i) |0^{otimes n}>|^2
                      = |<psi(x_j) | psi(x_i)>|^2

    Features:
    - High-throughput exact statevector simulation via PennyLane 'default.qubit'.
    - Exact unit diagonal: K_ii = 1.0.
    - Exact matrix symmetry: K_ij = K_ji.
    - Exact PSD property: Gram matrix of statevectors in complex projective Hilbert space.
    - Zero data leakage: standardizer fitted strictly on training data.
    """

    def __init__(self, config: Optional[BRQKConfig] = None) -> None:
        self.config = config or BRQKConfig()
        self.n_qubits = self.config.n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.standardizer = FeatureStandardizer(epsilon=self.config.epsilon)
        self.is_fitted = False

        # Build statevector QNode
        @qml.qnode(self.dev, interface="numpy", diff_method=None)
        def _statevector_qnode(theta: np.ndarray):
            biological_feature_map(
                theta=theta,
                interaction_pairs=self.config.interaction_pairs,
                entanglement_type=self.config.entanglement_type,
            )
            return qml.state()

        self._statevector_qnode = _statevector_qnode

        # Build pairwise overlap QNode (for circuit-level verification / finite-shot)
        @qml.qnode(self.dev, interface="numpy", diff_method=None)
        def _overlap_qnode(theta_i: np.ndarray, theta_j: np.ndarray):
            biological_feature_map(
                theta=theta_i,
                interaction_pairs=self.config.interaction_pairs,
                entanglement_type=self.config.entanglement_type,
            )
            adjoint_biological_feature_map(
                theta=theta_j,
                interaction_pairs=self.config.interaction_pairs,
                entanglement_type=self.config.entanglement_type,
            )
            return qml.probs(wires=range(self.n_qubits))

        self._overlap_qnode = _overlap_qnode

    def fit(self, X: np.ndarray) -> "BiologicalRelationalQuantumKernel":
        """
        Fit feature standardizer strictly on training data.
        """
        self.standardizer.fit(X)
        self.is_fitted = True
        return self

    def transform_to_angles(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize and map features to rotation angles in [-angle_scale, angle_scale].
        """
        if not self.is_fitted:
            raise ValueError("Kernel has not been fitted. Call fit(X_train) first.")
        X_scaled = self.standardizer.transform(X)
        return compute_angles(X_scaled, angle_scale=self.config.angle_scale)

    def compute_statevector(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the 2^n statevector |psi(x)> for a single sample.
        """
        return np.asarray(self._statevector_qnode(theta), dtype=np.complex128)

    def compute_statevectors_batch(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute statevectors for all samples in batch: shape (N, 2^n).
        """
        N = len(thetas)
        dim = 2 ** self.n_qubits
        statevectors = np.empty((N, dim), dtype=np.complex128)
        for i in range(N):
            statevectors[i] = self.compute_statevector(thetas[i])
        return statevectors

    def compute_overlap_qnode(self, theta_i: np.ndarray, theta_j: np.ndarray) -> float:
        """
        Compute transition probability via the inverse feature map QNode:
        |<0| U_phi(x_j)^dagger U_phi(x_i) |0>|^2 = P(|00...0>).
        """
        probs = self._overlap_qnode(theta_i, theta_j)
        return float(probs[0])

    def compute_gram_matrix(
        self,
        X_train: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute kernel Gram matrix.
        If X_eval is None: computes K_train,train of shape (N_tr, N_tr).
        If X_eval is provided: computes K_eval,train of shape (N_eval, N_tr).
        """
        if not self.is_fitted:
            self.fit(X_train)

        thetas_train = self.transform_to_angles(X_train)
        psi_train = self.compute_statevectors_batch(thetas_train)

        if X_eval is None:
            # Pairwise inner products: S_ij = <psi_i | psi_j> = sum_k psi_i[k]^* psi_j[k]
            # S = psi_train @ psi_train.conj().T
            S = np.matmul(psi_train, psi_train.conj().T)
            K = np.abs(S) ** 2
            # Symmetrize and enforce exact unit diagonal
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)
            return K
        else:
            thetas_eval = self.transform_to_angles(X_eval)
            psi_eval = self.compute_statevectors_batch(thetas_eval)
            # S_test_train = psi_eval @ psi_train.conj().T
            S = np.matmul(psi_eval, psi_train.conj().T)
            K = np.abs(S) ** 2
            return K
