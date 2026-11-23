"""
NĀḌĪ-Q Phase 13: Biological QUBO Builder.
Constructs mathematically verified QUBO matrices from MeTTa symbolic constraints
and translates QUBO formulations into Ising spin Hamiltonians for QAOA optimization.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from nadiq.models.metta_quantum.types import BiologicalQUBOConfig, CandidateRelationship


class BiologicalQUBOBuilder:
    """
    Translates evidence-supported biological relationship selection into a Quadratic Unconstrained
    Binary Optimization (QUBO) problem and an equivalent Ising spin Hamiltonian:

    Maximization Objective:
        Obj_max(x) = sum_i U_i x_i - lambda sum_{i<j} R_ij x_i x_j - mu sum_{i<j} C_ij x_i x_j - gamma (sum_i x_i - K)^2

    Minimization Energy Function:
        H(x) = - Obj_max(x) = sum_i q_i x_i + sum_{i<j} Q_ij x_i x_j + const

    Where:
        q_i = -U_i + gamma * (1 - 2K)
        Q_ij = lambda * R_ij + mu * C_ij + 2 * gamma  (for i < j)
        const = gamma * K^2

    Ising Mapping:
        x_i = (I - Z_i) / 2
        H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + c_0 I
    """

    def __init__(
        self,
        candidates: List[CandidateRelationship],
        redundancy_map: Dict[Tuple[int, int], float],
        contradiction_pairs: Set[Tuple[int, int]],
        config: BiologicalQUBOConfig = BiologicalQUBOConfig(),
    ):
        self.candidates = candidates
        self.redundancy_map = redundancy_map
        self.contradiction_pairs = contradiction_pairs
        self.config = config
        self.n_vars = len(candidates)

        self._build_qubo_matrices()
        self._build_ising_coefficients()

    def _build_qubo_matrices(self):
        """Constructs linear vector q and upper-triangular quadratic matrix Q."""
        N = self.n_vars
        K = self.config.budget_k
        lam = self.config.lambda_redundancy
        mu = self.config.mu_contradiction
        gamma = self.config.gamma_budget

        self.U = np.array([c.utility for c in self.candidates], dtype=np.float64)

        # q_i = -U_i + gamma * (1 - 2*K)
        self.q = -self.U + gamma * (1.0 - 2.0 * K)

        # Upper triangular Q_ij
        self.Q = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = self.redundancy_map.get((i, j), 0.0)
                c_ij = 1.0 if (i, j) in self.contradiction_pairs or (j, i) in self.contradiction_pairs else 0.0
                self.Q[i, j] = lam * r_ij + mu * c_ij + 2.0 * gamma

        self.qubo_offset = gamma * (K ** 2)

    def _build_ising_coefficients(self):
        """
        Maps binary QUBO variables x_i in {0, 1} to Pauli-Z spin operators Z_i in {+1, -1}.
        Using the transformation: x_i = (I - Z_i) / 2.
        """
        N = self.n_vars
        self.h = np.zeros(N, dtype=np.float64)
        self.J = np.zeros((N, N), dtype=np.float64)

        # Single-qubit Pauli-Z coefficients:
        # h_i = -0.5 * q_i - 0.25 * sum_{j != i} Q_symm[i, j]
        for i in range(N):
            q_term = -0.5 * self.q[i]
            quad_sum = 0.0
            for j in range(N):
                if i < j:
                    quad_sum += self.Q[i, j]
                elif i > j:
                    quad_sum += self.Q[j, i]
            self.h[i] = q_term - 0.25 * quad_sum

        # Two-qubit ZZ coupling coefficients:
        # J_ij = 0.25 * Q_ij (for i < j)
        for i in range(N):
            for j in range(i + 1, N):
                self.J[i, j] = 0.25 * self.Q[i, j]

        # Constant offset:
        # c_0 = qubo_offset + 0.5 * sum_i q_i + 0.25 * sum_{i < j} Q_ij
        self.ising_offset = (
            self.qubo_offset
            + 0.5 * np.sum(self.q)
            + 0.25 * np.sum(self.Q)
        )

    def evaluate_energy(self, x: np.ndarray) -> float:
        """
        Computes QUBO minimization energy H(x) = sum_i q_i x_i + sum_{i < j} Q_ij x_i x_j + const.
        Lower is better.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        linear = np.dot(self.q, x)
        quad = 0.0
        N = len(x)
        for i in range(N):
            for j in range(i + 1, N):
                quad += self.Q[i, j] * x[i] * x[j]
        return float(linear + quad + self.qubo_offset)

    def evaluate_utility(self, x: np.ndarray) -> float:
        """
        Computes original maximization objective Obj_max(x) = -H(x).
        Higher is better.
        """
        return -self.evaluate_energy(x)

    def evaluate_ising(self, z: np.ndarray) -> float:
        """
        Evaluates Ising Hamiltonian energy for spin configuration z in {+1, -1}^N.
        H_C = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + ising_offset.
        """
        z = np.asarray(z, dtype=np.float64).flatten()
        linear = np.dot(self.h, z)
        quad = 0.0
        N = len(z)
        for i in range(N):
            for j in range(i + 1, N):
                quad += self.J[i, j] * z[i] * z[j]
        return float(linear + quad + self.ising_offset)

    def verify_mathematical_equivalence(self, n_samples: int = 100) -> bool:
        """
        Verifies that H(x) == evaluate_ising(1 - 2x) across test bitstrings.
        Guarantees exact mathematical translation with zero formulation error.
        """
        rng = np.random.default_rng(42)
        N = self.n_vars
        for _ in range(n_samples):
            x = rng.integers(0, 2, size=N)
            z = 1 - 2 * x  # 0 -> +1, 1 -> -1
            e_qubo = self.evaluate_energy(x)
            e_ising = self.evaluate_ising(z)
            if not np.isclose(e_qubo, e_ising, atol=1e-9):
                return False
        return True

    def find_exact_optimum(self) -> Tuple[np.ndarray, float, float]:
        """
        Exhaustively evaluates all 2^N bitstrings to find the exact global minimum.
        Returns (best_bitstring, min_energy, max_utility).
        """
        N = self.n_vars
        best_x = None
        best_energy = float("inf")

        for bits in itertools.product([0, 1], repeat=N):
            x = np.array(bits, dtype=np.int32)
            e = self.evaluate_energy(x)
            if e < best_energy:
                best_energy = e
                best_x = x

        return best_x, best_energy, -best_energy

    def check_feasibility(self, x: np.ndarray) -> Tuple[bool, int, int]:
        """
        Checks whether bitstring x satisfies:
        1. sum(x) == budget_k (budget error = |sum(x) - K|)
        2. x_i * x_j == 0 for all contradiction pairs
        Returns (is_feasible, budget_error, num_contradictions).
        """
        x = np.asarray(x, dtype=np.int32).flatten()
        k_error = abs(int(np.sum(x)) - self.config.budget_k)

        contra_count = 0
        for i, j in self.contradiction_pairs:
            if x[i] == 1 and x[j] == 1:
                contra_count += 1

        is_feasible = (k_error == 0) and (contra_count == 0)
        return is_feasible, k_error, contra_count
