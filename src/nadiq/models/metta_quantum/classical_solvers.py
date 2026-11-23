"""
NĀḌĪ-Q Phase 13: Classical Control Solvers for Biological QUBO.
Implements:
- M0: Exhaustive Search (Exact Global Optimum)
- M1: Classical Simulated Annealing (SA)
- M2: Greedy Evidence-Based Selection
- M3: Classical Tabu Search / Quadratic Local Optimization
All solvers optimize the exact same objective and constraint penalties.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.types import QuantumSelectionOutput


class ClassicalSolvers:
    """Suite of classical optimization controls matched to the biological QUBO objective."""

    def __init__(self, qubo_builder: BiologicalQUBOBuilder):
        self.builder = qubo_builder
        self.n_vars = qubo_builder.n_vars

    def solve_exhaustive(self) -> QuantumSelectionOutput:
        """
        M0: Exhaustive Search / Exact Optimum.
        Guarantees finding the global minimum with zero optimality gap.
        """
        t0 = time.time()
        best_x, best_e, best_u = self.builder.find_exact_optimum()
        runtime = time.time() - t0

        selected_indices = [i for i, b in enumerate(best_x) if b == 1]
        selected_edges = [self.builder.candidates[i] for i in selected_indices]
        is_feasible, k_err, contra_cnt = self.builder.check_feasibility(best_x)

        return QuantumSelectionOutput(
            method="M0_Exact_Exhaustive",
            bitstring="".join(str(b) for b in best_x),
            selected_indices=selected_indices,
            selected_edges=selected_edges,
            energy=float(best_e),
            utility_score=float(best_u),
            is_feasible=is_feasible,
            budget_error=k_err,
            num_contradictions=contra_cnt,
            optimality_gap=0.0,
            probability=1.0,
            circuit_depth=0,
            gate_count={},
            n_evaluations=1 << self.n_vars,
            runtime_sec=runtime,
        )

    def solve_simulated_annealing(
        self,
        n_steps: int = 500,
        t_init: float = 2.0,
        t_min: float = 0.01,
        seed: int = 42,
    ) -> QuantumSelectionOutput:
        """
        M1: Classical Simulated Annealing (SA).
        Stochastic exploration using Metropolis-Hastings transitions and geometric cooling.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)
        N = self.n_vars
        _, exact_min_e, _ = self.builder.find_exact_optimum()

        # Random initial state
        current_x = rng.integers(0, 2, size=N)
        current_e = self.builder.evaluate_energy(current_x)

        best_x = current_x.copy()
        best_e = current_e
        n_evals = 1

        decay = (t_min / t_init) ** (1.0 / max(n_steps, 1))
        t = t_init

        for step in range(n_steps):
            # Propose 1-bit flip
            flip_idx = rng.integers(0, N)
            candidate_x = current_x.copy()
            candidate_x[flip_idx] = 1 - candidate_x[flip_idx]

            candidate_e = self.builder.evaluate_energy(candidate_x)
            n_evals += 1
            delta = candidate_e - current_e

            # Metropolis acceptance
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-8)):
                current_x = candidate_x
                current_e = candidate_e

                if current_e < best_e:
                    best_e = current_e
                    best_x = current_x.copy()

            t *= decay

        runtime = time.time() - t0
        selected_indices = [i for i, b in enumerate(best_x) if b == 1]
        selected_edges = [self.builder.candidates[i] for i in selected_indices]
        is_feasible, k_err, contra_cnt = self.builder.check_feasibility(best_x)
        gap = float(best_e - exact_min_e)
        utility = self.builder.evaluate_utility(best_x)

        return QuantumSelectionOutput(
            method="M1_Simulated_Annealing",
            bitstring="".join(str(b) for b in best_x),
            selected_indices=selected_indices,
            selected_edges=selected_edges,
            energy=float(best_e),
            utility_score=float(utility),
            is_feasible=is_feasible,
            budget_error=k_err,
            num_contradictions=contra_cnt,
            optimality_gap=gap,
            probability=1.0,
            circuit_depth=0,
            gate_count={},
            n_evaluations=n_evals,
            runtime_sec=runtime,
        )

    def solve_greedy(self) -> QuantumSelectionOutput:
        """
        M2: Greedy Evidence-Based Selection.
        Sequentially adds candidate with the best marginal energy improvement until K items are selected.
        """
        t0 = time.time()
        N = self.n_vars
        K = self.builder.config.budget_k
        _, exact_min_e, _ = self.builder.find_exact_optimum()

        selected = set()
        n_evals = 0

        for _ in range(K):
            best_cand = None
            best_cand_e = float("inf")

            for i in range(N):
                if i in selected:
                    continue
                test_set = selected | {i}
                x = np.zeros(N, dtype=np.int32)
                for idx in test_set:
                    x[idx] = 1

                e = self.builder.evaluate_energy(x)
                n_evals += 1

                if e < best_cand_e:
                    best_cand_e = e
                    best_cand = i

            if best_cand is not None:
                selected.add(best_cand)

        final_x = np.zeros(N, dtype=np.int32)
        for idx in selected:
            final_x[idx] = 1

        final_e = self.builder.evaluate_energy(final_x)
        runtime = time.time() - t0

        selected_indices = sorted(list(selected))
        selected_edges = [self.builder.candidates[i] for i in selected_indices]
        is_feasible, k_err, contra_cnt = self.builder.check_feasibility(final_x)
        gap = float(final_e - exact_min_e)
        utility = self.builder.evaluate_utility(final_x)

        return QuantumSelectionOutput(
            method="M2_Greedy_Selection",
            bitstring="".join(str(b) for b in final_x),
            selected_indices=selected_indices,
            selected_edges=selected_edges,
            energy=float(final_e),
            utility_score=float(utility),
            is_feasible=is_feasible,
            budget_error=k_err,
            num_contradictions=contra_cnt,
            optimality_gap=gap,
            probability=1.0,
            circuit_depth=0,
            gate_count={},
            n_evaluations=n_evals,
            runtime_sec=runtime,
        )

    def solve_tabu_search(
        self,
        n_iters: int = 150,
        tabu_tenure: int = 4,
        seed: int = 42,
    ) -> QuantumSelectionOutput:
        """
        M3: Classical Tabu Search / 1-Flip Local Search.
        Explores the full 1-bit flip neighborhood with a recency-based tabu memory to escape local minima.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)
        N = self.n_vars
        _, exact_min_e, _ = self.builder.find_exact_optimum()

        # Start from greedy solution or random
        current_x = rng.integers(0, 2, size=N)
        current_e = self.builder.evaluate_energy(current_x)

        best_x = current_x.copy()
        best_e = current_e

        tabu_list = np.zeros(N, dtype=np.int32)
        n_evals = 1

        for it in range(n_iters):
            best_move = None
            best_move_e = float("inf")

            # Evaluate all N neighboring 1-flip states
            for i in range(N):
                cand_x = current_x.copy()
                cand_x[i] = 1 - cand_x[i]
                cand_e = self.builder.evaluate_energy(cand_x)
                n_evals += 1

                # Aspiration criterion: allow tabu if better than global best
                is_tabu = tabu_list[i] > it
                if cand_e < best_e or not is_tabu:
                    if cand_e < best_move_e:
                        best_move_e = cand_e
                        best_move = i

            if best_move is None:
                # Fallback: pick any random move
                best_move = rng.integers(0, N)
                cand_x = current_x.copy()
                cand_x[best_move] = 1 - cand_x[best_move]
                best_move_e = self.builder.evaluate_energy(cand_x)

            current_x[best_move] = 1 - current_x[best_move]
            current_e = best_move_e
            tabu_list[best_move] = it + tabu_tenure

            if current_e < best_e:
                best_e = current_e
                best_x = current_x.copy()

        runtime = time.time() - t0
        selected_indices = [i for i, b in enumerate(best_x) if b == 1]
        selected_edges = [self.builder.candidates[i] for i in selected_indices]
        is_feasible, k_err, contra_cnt = self.builder.check_feasibility(best_x)
        gap = float(best_e - exact_min_e)
        utility = self.builder.evaluate_utility(best_x)

        return QuantumSelectionOutput(
            method="M3_Tabu_Local_Search",
            bitstring="".join(str(b) for b in best_x),
            selected_indices=selected_indices,
            selected_edges=selected_edges,
            energy=float(best_e),
            utility_score=float(utility),
            is_feasible=is_feasible,
            budget_error=k_err,
            num_contradictions=contra_cnt,
            optimality_gap=gap,
            probability=1.0,
            circuit_depth=0,
            gate_count={},
            n_evaluations=n_evals,
            runtime_sec=runtime,
        )
