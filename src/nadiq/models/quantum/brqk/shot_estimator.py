"""
NĀḌĪ-Q Phase 14A: Finite-Shot Quantum Kernel Estimator.
Estimates quantum transition amplitudes under realistic shot noise:
    K_hat(x_i, x_j) = N_{00...0} / M
with theoretical binomial sampling variance sigma^2 = p(1-p)/M.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Dict, List, Optional, Tuple

from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.feature_map import (
    biological_feature_map,
    adjoint_biological_feature_map,
)


class FiniteShotKernelEstimator:
    """
    Evaluates quantum kernel overlaps using shot-based sampling.
    """

    def __init__(self, config: Optional[BRQKConfig] = None) -> None:
        self.config = config or BRQKConfig()
        self.n_qubits = self.config.n_qubits
        self.n_shots = self.config.n_shots

        # Create shot-enabled device
        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=self.n_shots,
            seed=self.config.seed,
        )

        @qml.qnode(self.dev, diff_method=None)
        def _shot_overlap_qnode(theta_i: np.ndarray, theta_j: np.ndarray):
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
            return qml.counts()

        self._shot_overlap_qnode = _shot_overlap_qnode

    def estimate_overlap(
        self,
        theta_i: np.ndarray,
        theta_j: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate transition amplitude between two angle vectors using finite shots.

        Returns:
            Dict containing 'estimated_kernel', 'shot_count', 'ground_state_counts', and 'theoretical_std'.
        """
        counts = self._shot_overlap_qnode(theta_i, theta_j)
        zero_key = "0" * self.n_qubits
        n_zeros = int(counts.get(zero_key, 0))
        p_hat = n_zeros / float(self.n_shots)
        # Theoretical standard deviation under binomial sampling
        std_err = float(np.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / self.n_shots))

        return {
            "estimated_kernel": float(p_hat),
            "shot_count": self.n_shots,
            "ground_state_counts": n_zeros,
            "theoretical_std": std_err,
        }
