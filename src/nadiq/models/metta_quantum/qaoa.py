"""
NĀḌĪ-Q Phase 13: Quantum Approximate Optimization Algorithm (QAOA) Optimizer.
Implements a PennyLane-based QAOA circuit for evidence-grown biological relationship selection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.types import QuantumSelectionOutput


class QAOAOptimizer:
    """
    Solves the biological graph selection QUBO using QAOA on PennyLane:
    - Layer count p in {1, 2}
    - Cost Hamiltonian H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset
    - Mixer Hamiltonian H_M = sum_i X_i
    - Exact statevector simulation via default.qubit
    - Multi-restart parameter optimization
    """

    def __init__(
        self,
        qubo_builder: BiologicalQUBOBuilder,
        p_layers: int = 1,
        n_restarts: int = 3,
        n_steps: int = 30,
        lr: float = 0.08,
        seed: int = 42,
    ):
        self.builder = qubo_builder
        self.n_vars = qubo_builder.n_vars
        self.p_layers = p_layers
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr
        self.seed = seed

        self.dev = qml.device("default.qubit", wires=self.n_vars)
        self._precompute_energy_vector()

    def _precompute_energy_vector(self):
        """Precomputes exact QUBO energy for all 2^N basis states for fast, differentiable expectation computation."""
        N = self.n_vars
        n_states = 1 << N
        energies = np.zeros(n_states, dtype=np.float32)

        for idx in range(n_states):
            # Bit representation (big-endian: qubit 0 is MSB)
            bits = np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)], dtype=np.int32)
            energies[idx] = self.builder.evaluate_energy(bits)

        self.basis_energies = torch.tensor(energies, dtype=torch.float32)

    def _build_circuit(self):
        """Defines the parameterized QAOA quantum node returning state probabilities."""
        N = self.n_vars
        p = self.p_layers
        h_const = torch.tensor(self.builder.h, dtype=torch.float32)
        J_const = torch.tensor(self.builder.J, dtype=torch.float32)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor):
            # 1. Equal Superposition State
            for i in range(N):
                qml.Hadamard(wires=i)

            # 2. Alternating Cost and Mixer Layers
            for layer in range(p):
                gamma = params[0, layer]
                beta = params[1, layer]

                # Cost unitary: e^{-i gamma H_C}
                for i in range(N):
                    if abs(h_const[i]) > 1e-8:
                        qml.RZ(2.0 * gamma * h_const[i], wires=i)

                for i in range(N):
                    for j in range(i + 1, N):
                        if abs(J_const[i, j]) > 1e-8:
                            qml.IsingZZ(2.0 * gamma * J_const[i, j], wires=[i, j])

                # Mixer unitary: e^{-i beta H_M}
                for i in range(N):
                    qml.RX(2.0 * beta, wires=i)

            return qml.probs(wires=range(N))

        return circuit

    def optimize(self) -> QuantumSelectionOutput:
        """
        Runs multi-start gradient-based QAOA parameter optimization.
        Returns the best QuantumSelectionOutput.
        """
        t0 = time.time()
        circuit = self._build_circuit()
        exact_x, exact_min_e, _ = self.builder.find_exact_optimum()

        best_cost = float("inf")
        best_params = None
        best_probs = None
        n_evals = 0

        # Run multi-start optimization
        for restart in range(self.n_restarts):
            torch.manual_seed(self.seed + restart * 100)
            # Initialize (gamma, beta) in [0, pi] as a proper leaf tensor
            init_params = (torch.rand((2, self.p_layers), dtype=torch.float32) * np.pi).clone().detach().requires_grad_(True)
            optimizer = optim.Adam([init_params], lr=self.lr)

            for step in range(self.n_steps):
                optimizer.zero_grad()
                probs = circuit(init_params)
                n_evals += 1
                # Expected cost: sum_k P(k) * E(k)
                exp_cost = torch.sum(probs * self.basis_energies)
                exp_cost.backward()
                optimizer.step()

            with torch.no_grad():
                final_probs = circuit(init_params)
                final_cost = float(torch.sum(final_probs * self.basis_energies).item())
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_params = init_params.detach().clone()
                    best_probs = final_probs.detach().cpu().numpy()

        # Selection: Pick the bitstring with highest probability among states that minimize energy
        # Inspect top 10 most probable states
        top_indices = np.argsort(best_probs)[::-1][:16]
        best_bitstring = None
        best_sample_energy = float("inf")
        best_sample_prob = 0.0

        N = self.n_vars
        for idx in top_indices:
            bits = np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)], dtype=np.int32)
            e = self.builder.evaluate_energy(bits)
            if e < best_sample_energy:
                best_sample_energy = e
                best_bitstring = bits
                best_sample_prob = float(best_probs[idx])

        # If highest probability state is different from min-energy sampled state, record both
        max_prob_idx = int(np.argmax(best_probs))
        max_prob_bits = np.array([(max_prob_idx >> (N - 1 - i)) & 1 for i in range(N)], dtype=np.int32)
        max_prob_energy = self.builder.evaluate_energy(max_prob_bits)

        # Standard selection chooses the lowest energy discovered state
        selected_x = best_bitstring
        selected_e = best_sample_energy
        selected_prob = best_sample_prob

        selected_indices = [i for i, b in enumerate(selected_x) if b == 1]
        selected_edges = [self.builder.candidates[i] for i in selected_indices]
        is_feasible, k_err, contra_cnt = self.builder.check_feasibility(selected_x)
        gap = float(selected_e - exact_min_e)
        utility = self.builder.evaluate_utility(selected_x)

        # Calculate circuit specs (depth and gate counts)
        circuit_depth = self.p_layers * 3 + 1
        n_hadamards = N
        n_rz = self.p_layers * N
        n_zz = self.p_layers * int(np.sum(np.abs(self.builder.J) > 1e-8))
        n_rx = self.p_layers * N
        gate_counts = {
            "Hadamard": n_hadamards,
            "RZ": n_rz,
            "IsingZZ": n_zz,
            "RX": n_rx,
            "Total": n_hadamards + n_rz + n_zz + n_rx,
        }

        runtime = time.time() - t0
        bitstring_str = "".join(str(b) for b in selected_x)

        return QuantumSelectionOutput(
            method=f"QAOA_p{self.p_layers}",
            bitstring=bitstring_str,
            selected_indices=selected_indices,
            selected_edges=selected_edges,
            energy=float(selected_e),
            utility_score=float(utility),
            is_feasible=is_feasible,
            budget_error=k_err,
            num_contradictions=contra_cnt,
            optimality_gap=gap,
            probability=selected_prob,
            circuit_depth=circuit_depth,
            gate_count=gate_counts,
            n_evaluations=n_evals,
            runtime_sec=runtime,
        )
