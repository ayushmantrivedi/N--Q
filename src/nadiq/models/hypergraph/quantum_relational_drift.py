"""
NĀḌĪ-Q Phase 10B: Quantum Relational Drift Calculator.
Computes quantum state fidelity overlap F_Q and relational drift D_Q between consecutive visits,
alongside matched classical distance benchmarks (Euclidean, Cosine, RBF).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumRelationalDriftCalculator:
    """
    Computes quantum relational fidelity and drift between sequential visits:
    F_Q(t-1, t) = |<Psi_(t-1) | Psi_t>|^2
    D_Q(t) = 1 - F_Q(t-1, t)
    Alongside classical metric benchmarks.
    """

    def __init__(self, rbf_gamma: float = 0.05):
        self.rbf_gamma = rbf_gamma

    def compute_quantum_drift(
        self,
        psi_prev: Optional[torch.Tensor],
        psi_curr: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Computes state overlap fidelity and quantum relational drift.
        Returns: (F_Q, D_Q)
        """
        if psi_prev is None:
            # Baseline visit: no prior visit
            return 1.0, 0.0

        # Compute inner product <psi_prev | psi_curr>
        inner_prod = torch.vdot(psi_prev.flatten(), psi_curr.flatten())
        fidelity = float(torch.abs(inner_prod).item() ** 2)
        fidelity = max(0.0, min(1.0, fidelity))
        drift = 1.0 - fidelity
        return fidelity, float(drift)

    def compute_classical_distances(
        self,
        x_prev: Optional[np.ndarray],
        x_curr: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes Euclidean, Cosine, and RBF metrics between consecutive observations.
        """
        if x_prev is None:
            return {
                "euclidean_distance": 0.0,
                "cosine_distance": 0.0,
                "rbf_similarity": 1.0
            }

        diff = x_curr - x_prev
        euc = float(np.linalg.norm(diff))
        
        norm_p = np.linalg.norm(x_prev)
        norm_c = np.linalg.norm(x_curr)
        if norm_p > 1e-6 and norm_c > 1e-6:
            cos_sim = float(np.dot(x_prev, x_curr) / (norm_p * norm_c))
            cos_dist = float(1.0 - np.clip(cos_sim, -1.0, 1.0))
        else:
            cos_dist = 0.0

        rbf = float(np.exp(-self.rbf_gamma * (euc ** 2)))

        return {
            "euclidean_distance": euc,
            "cosine_distance": cos_dist,
            "rbf_similarity": rbf
        }
