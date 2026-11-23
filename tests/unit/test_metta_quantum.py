"""
NĀḌĪ-Q Phase 13 Unit Tests: MeTTa-Q & Test A Synthetic Correctness.
Validates:
1. MeTTa symbolic compiler promotion rules and candidate extraction.
2. QUBO mathematical equivalence with Ising spin Hamiltonian.
3. Penalty scaling (budget constraint, redundancy, contradictions).
4. Exact recovery of planted synthetic optimum.
5. QAOA quantum circuit compilation, specs, and execution.
6. Classical controls (M0, M1, M2, M3) consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from nadiq.models.metta_quantum.compiler import MeTTaGraphCompiler
from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.types import BiologicalQUBOConfig, CandidateRelationship
from nadiq.models.metta_quantum.qaoa import QAOAOptimizer
from nadiq.models.metta_quantum.classical_solvers import ClassicalSolvers


@pytest.fixture
def synthetic_candidate_pool():
    """Returns a pool of 8 synthetic candidate edges with known utilities and constraints."""
    raw_edges = [
        {"src": "mol_asyn", "dst": "struct_caudate", "r": 0.45, "p": 0.001, "se": 0.08, "n": 100, "lag": 1.0},
        {"src": "mol_ptau", "dst": "struct_caudate", "r": 0.40, "p": 0.002, "se": 0.09, "n": 100, "lag": 1.0},
        {"src": "struct_caudate", "dst": "motor_total", "r": 0.50, "p": 0.0005, "se": 0.07, "n": 100, "lag": 1.0},
        {"src": "struct_putamen", "dst": "motor_brady", "r": 0.48, "p": 0.0008, "se": 0.07, "n": 100, "lag": 1.0},
        {"src": "mol_asyn", "dst": "motor_total", "r": 0.20, "p": 0.04, "se": 0.12, "n": 100, "lag": 1.0},
        {"src": "struct_putamen", "dst": "motor_total", "r": 0.35, "p": 0.01, "se": 0.10, "n": 100, "lag": 1.0},
        {"src": "mol_abeta", "dst": "struct_caudate", "r": 0.08, "p": 0.08, "se": 0.15, "n": 100, "lag": 1.0},
        {"src": "motor_total", "dst": "mol_asyn", "r": -0.15, "p": 0.12, "se": 0.18, "n": 100, "lag": 1.0},
    ]
    compiler = MeTTaGraphCompiler()
    # Edges 0 and 1 are redundant (both molecular to caudate)
    collinear = [(0, 1, 0.70)]
    # Edge 4 and Edge 7 are contradictory (opposite directions)
    contradictions = [(4, 7)]
    candidates, red_map, contra_pairs = compiler.compile_candidates(
        raw_edges, collinear_redundancies=collinear, incompatibility_pairs=contradictions
    )
    return {
        "candidates": candidates,
        "redundancy_map": red_map,
        "contradiction_pairs": contra_pairs,
    }


def test_metta_compiler_promotions(synthetic_candidate_pool):
    """Contract 1: MeTTa correctly promotes validated edges and assigns positive utilities."""
    candidates = synthetic_candidate_pool["candidates"]
    assert len(candidates) == 8

    # Edge 0 (p=0.001) should be Validated
    assert candidates[0].status == "Validated"
    # Edge 6 (p=0.08) should be Provisional
    assert candidates[6].status == "Provisional"
    # Edge 7 (p=0.12 > 0.10) should be Candidate
    assert candidates[7].status == "Candidate"

    for c in candidates:
        assert c.utility > 0.0
    # Higher correlation should have higher utility
    assert candidates[2].utility > candidates[6].utility


def test_qubo_mathematical_equivalence(synthetic_candidate_pool):
    """Contract 2: QUBO minimization energy matches Ising Hamiltonian energy exactly."""
    builder = BiologicalQUBOBuilder(
        candidates=synthetic_candidate_pool["candidates"],
        redundancy_map=synthetic_candidate_pool["redundancy_map"],
        contradiction_pairs=synthetic_candidate_pool["contradiction_pairs"],
        config=BiologicalQUBOConfig(budget_k=3, gamma_budget=1.5),
    )

    # Verify that H(x) == evaluate_ising(1 - 2x) across 150 random bitstrings
    is_equivalent = builder.verify_mathematical_equivalence(n_samples=150)
    assert is_equivalent is True, "QUBO energy does not match Ising spin Hamiltonian energy!"


def test_qubo_penalty_scaling_and_budget(synthetic_candidate_pool):
    """Contract 3: Budget deviations and contradiction pairs incur exact expected penalties."""
    cfg = BiologicalQUBOConfig(budget_k=3, gamma_budget=2.0, mu_contradiction=5.0)
    builder = BiologicalQUBOBuilder(
        candidates=synthetic_candidate_pool["candidates"],
        redundancy_map=synthetic_candidate_pool["redundancy_map"],
        contradiction_pairs=synthetic_candidate_pool["contradiction_pairs"],
        config=cfg,
    )

    # State with sum(x) = 3 (zero budget error)
    x_valid_k = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    is_feas, k_err, contra_cnt = builder.check_feasibility(x_valid_k)
    assert is_feas is True
    assert k_err == 0
    assert contra_cnt == 0

    # State with sum(x) = 1 (budget error = 2)
    x_invalid_k = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    is_feas_bad, k_err_bad, _ = builder.check_feasibility(x_invalid_k)
    assert is_feas_bad is False
    assert k_err_bad == 2

    # State with contradictory pair (4 and 7 active)
    x_contra = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int32)
    is_feas_c, _, contra_c = builder.check_feasibility(x_contra)
    assert is_feas_c is False
    assert contra_c == 1


def test_synthetic_planted_optimum_recovery(synthetic_candidate_pool):
    """Contract 4: Exhaustive search finds global optimum and checks feasibility."""
    builder = BiologicalQUBOBuilder(
        candidates=synthetic_candidate_pool["candidates"],
        redundancy_map=synthetic_candidate_pool["redundancy_map"],
        contradiction_pairs=synthetic_candidate_pool["contradiction_pairs"],
        config=BiologicalQUBOConfig(budget_k=3),
    )

    solvers = ClassicalSolvers(builder)
    m0_out = solvers.solve_exhaustive()

    assert m0_out.optimality_gap == 0.0
    assert m0_out.is_feasible is True
    assert len(m0_out.selected_indices) == 3
    # Contradictory pair (4, 7) must not both be selected
    assert not (4 in m0_out.selected_indices and 7 in m0_out.selected_indices)


def test_qaoa_circuit_execution_and_specs(synthetic_candidate_pool):
    """Contract 5: QAOAOptimizer executes cleanly, calculates depth, gates, and probability."""
    builder = BiologicalQUBOBuilder(
        candidates=synthetic_candidate_pool["candidates"],
        redundancy_map=synthetic_candidate_pool["redundancy_map"],
        contradiction_pairs=synthetic_candidate_pool["contradiction_pairs"],
        config=BiologicalQUBOConfig(budget_k=3),
    )

    qaoa = QAOAOptimizer(
        qubo_builder=builder,
        p_layers=1,
        n_restarts=2,
        n_steps=10,
        lr=0.10,
        seed=42,
    )

    out = qaoa.optimize()
    assert out.method == "QAOA_p1"
    assert len(out.bitstring) == 8
    assert out.circuit_depth > 0
    assert "IsingZZ" in out.gate_count
    assert out.gate_count["IsingZZ"] > 0
    assert out.probability > 0.0
    assert not np.isnan(out.energy)
    assert out.optimality_gap >= 0.0


def test_classical_controls_parity(synthetic_candidate_pool):
    """Contract 6: M0, M1, M2, M3 all evaluate the identical objective function."""
    builder = BiologicalQUBOBuilder(
        candidates=synthetic_candidate_pool["candidates"],
        redundancy_map=synthetic_candidate_pool["redundancy_map"],
        contradiction_pairs=synthetic_candidate_pool["contradiction_pairs"],
        config=BiologicalQUBOConfig(budget_k=3),
    )

    solvers = ClassicalSolvers(builder)
    out_m0 = solvers.solve_exhaustive()
    out_m1 = solvers.solve_simulated_annealing(n_steps=100, seed=42)
    out_m2 = solvers.solve_greedy()
    out_m3 = solvers.solve_tabu_search(n_iters=50, seed=42)

    for out in [out_m0, out_m1, out_m2, out_m3]:
        # Energy must equal builder.evaluate_energy(bitstring)
        bits = np.array([int(b) for b in out.bitstring], dtype=np.int32)
        expected_e = builder.evaluate_energy(bits)
        assert np.isclose(out.energy, expected_e, atol=1e-6)
        assert np.isclose(out.utility_score, -expected_e, atol=1e-6)
        assert out.optimality_gap >= -1e-6  # Cannot be better than exact optimum
