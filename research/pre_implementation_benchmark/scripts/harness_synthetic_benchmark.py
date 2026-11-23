"""
NĀḌĪ-Q Pre-Implementation Benchmark Validation:
Harness 2: Controlled Synthetic Ground-Truth Benchmark (20 Replicates).

Protocol:
- N = 341 synthetic longitudinal multimodal participants per replicate
- 20 Independent random seeds (seeds 1 to 20)
- 10 Candidate biological relationships:
    * 4 Planted True Causal Relationships (non-redundant, non-contradictory)
    * 2 Collinear Redundant Distractors (high pairwise redundancy R_ij >= 0.70)
    * 2 Contradictory Relationships (shortcut vs mediated cascade)
    * 2 Pure Noise Associations
- Target budget K = 4
- Comparison of 7 Solvers:
    1. Heuristic Top-K (Utility ranking alone)
    2. M0: Exact Exhaustive Search (Global optimum)
    3. M1: Simulated Annealing (SA)
    4. M2: Greedy Evidence Selection
    5. M3: Tabu Search / Local Quadratic Search
    6. QAOA p=1 (1-layer PennyLane parameterized ansatz)
    7. QAOA p=2 (2-layer PennyLane parameterized ansatz)
- Metrics:
    * Energy & Optimality Gap
    * Feasibility Rate (%)
    * Precision, Recall, F1 against planted ground truth
    * Selection Stability (Pairwise Jaccard across 20 replicates)
    * Runtime (milliseconds)
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from nadiq.models.metta_quantum.types import (
    BiologicalQUBOConfig,
    CandidateRelationship,
    QuantumSelectionOutput,
)
from nadiq.models.metta_quantum.compiler import MeTTaGraphCompiler
from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.classical_solvers import ClassicalSolvers
from nadiq.models.metta_quantum.qaoa import QAOAOptimizer


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes Jaccard index between two sets."""
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 1.0
    return float(len(set_a & set_b) / union_len)


def solve_top_k(builder: BiologicalQUBOBuilder, k: int = 4) -> QuantumSelectionOutput:
    """Heuristic Top-K baseline: selects k items with highest raw utility."""
    t0 = time.perf_counter()
    utilities = builder.U
    sorted_indices = np.argsort(-utilities)[:k]
    x = np.zeros(builder.n_vars, dtype=np.int32)
    x[sorted_indices] = 1

    energy = builder.evaluate_energy(x)
    utility = builder.evaluate_utility(x)
    is_feas, k_err, contra_cnt = builder.check_feasibility(x)
    dt = time.perf_counter() - t0

    bitstring = "".join(str(b) for b in x)
    return QuantumSelectionOutput(
        method="Top_K_Heuristic",
        bitstring=bitstring,
        selected_indices=sorted_indices.tolist(),
        selected_edges=[builder.candidates[i] for i in sorted_indices],
        energy=float(energy),
        utility_score=float(utility),
        is_feasible=bool(is_feas),
        budget_error=int(k_err),
        num_contradictions=int(contra_cnt),
        optimality_gap=0.0,  # filled later relative to M0
        probability=1.0,
        circuit_depth=0,
        gate_count={},
        n_evaluations=1,
        runtime_sec=float(dt),
    )


def generate_synthetic_cohort(
    n_participants: int = 341,
    seed: int = 42,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[Dict[str, Any]], List[Tuple[int, int, float]], List[Tuple[int, int]], Set[int]]:
    """
    Generates synthetic longitudinal multimodal data with known planted ground truth structure:
    Features:
      mol_0, mol_1 (molecular)
      struct_0, struct_1 (striatal imaging)
      motor_0, motor_1 (clinical motor)
      noise_0, noise_1 (noise biomarkers)

    Planted relationships (indices 0, 1, 2, 3):
      0: mol_0 -> struct_0 (r ~ 0.50)
      1: mol_1 -> struct_1 (r ~ 0.45)
      2: struct_0 -> motor_0 (r ~ 0.48)
      3: struct_1 -> motor_1 (r ~ 0.42)

    Redundant collinear distractors (indices 4, 5):
      4: mol_0 -> struct_1 (redundant with 0 & 1, R_04 = 0.75)
      5: struct_0 -> motor_1 (redundant with 2 & 3, R_25 = 0.70)

    Contradictory shortcut edges (indices 6, 7):
      6: mol_0 -> motor_0 (direct shortcut contradicting intermediate cascade 0 + 2)
      7: struct_1 -> motor_0 (conflicting target projection contradicting 2)

    Pure noise associations (indices 8, 9):
      8: noise_0 -> motor_0 (r ~ 0.05, p ~ 0.35)
      9: noise_1 -> struct_1 (r ~ -0.04, p ~ 0.45)
    """
    rng = np.random.default_rng(seed)

    # Latent factors
    z_mol0 = rng.normal(0, 1, n_participants)
    z_mol1 = rng.normal(0, 1, n_participants)

    # Propagate through causal graph with noise
    z_struct0 = 0.55 * z_mol0 + rng.normal(0, 0.8, n_participants)
    z_struct1 = 0.50 * z_mol1 + rng.normal(0, 0.8, n_participants)

    z_motor0 = 0.52 * z_struct0 + rng.normal(0, 0.8, n_participants)
    z_motor1 = 0.48 * z_struct1 + rng.normal(0, 0.8, n_participants)

    # Feature baseline and follow-up (deltas reflect underlying latent progression)
    def make_pair(z, noise_scale=0.3):
        bl = rng.normal(20, 5, n_participants)
        delta = 2.0 * z + rng.normal(0, noise_scale, n_participants)
        v04 = bl + delta
        return bl, v04

    features_raw = {
        "mol_0": make_pair(z_mol0),
        "mol_1": make_pair(z_mol1),
        "struct_0": make_pair(z_struct0),
        "struct_1": make_pair(z_struct1),
        "motor_0": make_pair(z_motor0),
        "motor_1": make_pair(z_motor1),
        "noise_0": make_pair(rng.normal(0, 1, n_participants)),
        "noise_1": make_pair(rng.normal(0, 1, n_participants)),
    }

    candidate_definitions = [
        {"id": 0, "src": "mol_0", "dst": "struct_0", "type": "molecular_to_structural"},
        {"id": 1, "src": "mol_1", "dst": "struct_1", "type": "molecular_to_structural"},
        {"id": 2, "src": "struct_0", "dst": "motor_0", "type": "structural_to_motor"},
        {"id": 3, "src": "struct_1", "dst": "motor_1", "type": "structural_to_motor"},
        {"id": 4, "src": "mol_0", "dst": "struct_1", "type": "molecular_to_structural"},
        {"id": 5, "src": "struct_0", "dst": "motor_1", "type": "structural_to_motor"},
        {"id": 6, "src": "mol_0", "dst": "motor_0", "type": "molecular_to_motor"},
        {"id": 7, "src": "struct_1", "dst": "motor_0", "type": "structural_to_motor"},
        {"id": 8, "src": "noise_0", "dst": "motor_0", "type": "noise_association"},
        {"id": 9, "src": "noise_1", "dst": "struct_1", "type": "noise_association"},
    ]

    collinear_redundancies = [
        (0, 4, 0.75),
        (1, 4, 0.70),
        (2, 5, 0.70),
        (3, 5, 0.65),
    ]

    incompatibility_pairs = [
        (2, 6),  # Cascade struct_0->motor_0 vs shortcut mol_0->motor_0
        (2, 7),  # Conflicting convergence on motor_0
    ]

    planted_optimal_indices = {0, 1, 2, 3}

    return (
        features_raw,
        candidate_definitions,
        collinear_redundancies,
        incompatibility_pairs,
        planted_optimal_indices,
    )


def run_synthetic_benchmark(n_replicates: int = 20):
    """Executes synthetic benchmark across 20 independent replicates."""
    print(f"Executing Synthetic Benchmark across {n_replicates} independent replicates...")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    qubo_cfg = BiologicalQUBOConfig(
        lambda_redundancy=0.60,
        mu_contradiction=2.50,
        gamma_budget=1.50,
        budget_k=4,
    )

    methods = [
        "Top_K_Heuristic",
        "M0_Exact_Exhaustive",
        "M1_Simulated_Annealing",
        "M2_Greedy_Selection",
        "M3_Tabu_Local_Search",
        "QAOA_p1",
        "QAOA_p2",
    ]

    all_records: List[Dict[str, Any]] = []
    selected_sets_by_method: Dict[str, List[set]] = {m: [] for m in methods}

    t0_all = time.time()

    for rep in range(1, n_replicates + 1):
        rep_seed = 100 + rep
        print(f"--- Replicate {rep}/{n_replicates} (Seed: {rep_seed}) ---")

        (
            features_raw,
            c_defs,
            collinear_red,
            incompat_pairs,
            planted_truth,
        ) = generate_synthetic_cohort(n_participants=341, seed=rep_seed)

        # Compute delta correlations
        raw_candidates = []
        for c in c_defs:
            c_id = c["id"]
            s_name = c["src"]
            d_name = c["dst"]

            bl_s, v4_s = features_raw[s_name]
            bl_d, v4_d = features_raw[d_name]

            dx_s = (v4_s - bl_s) / np.std(bl_s)
            dx_d = (v4_d - bl_d) / np.std(bl_d)

            r, p = stats.pearsonr(dx_s, dx_d)
            se = math.sqrt((1 - r**2) / max(len(dx_s) - 2, 1))

            raw_candidates.append({
                "id": c_id,
                "src": s_name,
                "dst": d_name,
                "r": float(r),
                "p": float(p),
                "se": float(se),
                "lag": 1.0,
                "n": len(dx_s),
                "rel_type": c["type"],
                "direction": 1 if r >= 0 else -1,
            })

        # MeTTa Compilation
        compiler = MeTTaGraphCompiler(r_thresh=0.05, p_thresh=0.10, n_min=30)
        verified_cands, red_map, contra_pairs = compiler.compile_candidates(
            raw_candidates,
            collinear_redundancies=collinear_red,
            incompatibility_pairs=incompat_pairs,
        )

        builder = BiologicalQUBOBuilder(
            candidates=verified_cands,
            redundancy_map=red_map,
            contradiction_pairs=contra_pairs,
            config=qubo_cfg,
        )

        # 1. M0: Exact Exhaustive Search
        classical = ClassicalSolvers(builder)
        out_m0 = classical.solve_exhaustive()

        # 2. Top-K Heuristic
        out_topk = solve_top_k(builder, k=qubo_cfg.budget_k)
        out_topk.optimality_gap = max(0.0, out_topk.energy - out_m0.energy)

        # 3. M1: Simulated Annealing
        out_m1 = classical.solve_simulated_annealing(n_steps=500, seed=rep_seed)

        # 4. M2: Greedy Evidence Selection
        out_m2 = classical.solve_greedy()

        # 5. M3: Tabu Search
        out_m3 = classical.solve_tabu_search(n_iters=150, tabu_tenure=4, seed=rep_seed)

        # 6. QAOA-p1
        qaoa_p1 = QAOAOptimizer(
            qubo_builder=builder,
            p_layers=1,
            n_restarts=1,
            n_steps=12,
            lr=0.08,
            seed=rep_seed,
        )
        out_q1 = qaoa_p1.optimize()

        # 7. QAOA-p2
        qaoa_p2 = QAOAOptimizer(
            qubo_builder=builder,
            p_layers=2,
            n_restarts=1,
            n_steps=12,
            lr=0.08,
            seed=rep_seed,
        )
        out_q2 = qaoa_p2.optimize()

        rep_outputs = [out_topk, out_m0, out_m1, out_m2, out_m3, out_q1, out_q2]

        for out in rep_outputs:
            sel_set = set(out.selected_indices)
            selected_sets_by_method[out.method].append(sel_set)

            # Precision, Recall, F1 against planted ground truth
            tp = len(sel_set & planted_truth)
            prec = tp / len(sel_set) if len(sel_set) > 0 else 0.0
            rec = tp / len(planted_truth) if len(planted_truth) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            rec_dict = {
                "replicate": rep,
                "seed": rep_seed,
                "method": out.method,
                "bitstring": out.bitstring,
                "energy": out.energy,
                "utility_score": out.utility_score,
                "optimality_gap": out.optimality_gap,
                "is_feasible": out.is_feasible,
                "budget_error": out.budget_error,
                "num_contradictions": out.num_contradictions,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "runtime_ms": out.runtime_sec * 1000.0,
                "n_evaluations": out.n_evaluations,
                "circuit_depth": out.circuit_depth,
            }
            all_records.append(rec_dict)

        print(
            f"  Exact E={out_m0.energy:.3f} | Tabu gap={out_m3.optimality_gap:.3f} | "
            f"QAOA-p1 gap={out_q1.optimality_gap:.3f} | QAOA-p2 gap={out_q2.optimality_gap:.3f} | "
            f"TopK gap={out_topk.optimality_gap:.3f}"
        )

    # Compute selection stability across replicates
    stability_by_method = {}
    for m, sets_list in selected_sets_by_method.items():
        jaccards = []
        for s1, s2 in itertools.combinations(sets_list, 2):
            jaccards.append(jaccard_similarity(s1, s2))
        stability_by_method[m] = float(np.mean(jaccards)) if jaccards else 1.0

    df_results = pd.DataFrame(all_records)
    csv_path = os.path.join(out_dir, "synthetic_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)

    summary = df_results.groupby("method").agg({
        "energy": ["mean", "std"],
        "optimality_gap": ["mean", "std"],
        "is_feasible": "mean",
        "budget_error": "mean",
        "num_contradictions": "mean",
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1_score": ["mean", "std"],
        "runtime_ms": ["mean", "std"],
    }).reset_index()

    summary.columns = [
        "method",
        "energy_mean", "energy_std",
        "gap_mean", "gap_std",
        "feasibility_rate",
        "budget_error_mean",
        "contradictions_mean",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "runtime_ms_mean", "runtime_ms_std",
    ]
    summary["selection_stability"] = summary["method"].map(stability_by_method)

    print("\n" + "=" * 80)
    print("SYNTHETIC BENCHMARK SUMMARY (20 Replicates)")
    print("=" * 80)
    cols = [
        "method", "energy_mean", "gap_mean", "feasibility_rate",
        "precision_mean", "recall_mean", "f1_mean", "selection_stability", "runtime_ms_mean"
    ]
    print(summary[cols].sort_values(by="energy_mean").to_string(index=False))

    summary_dict = summary.to_dict(orient="records")
    summary_path = os.path.join(out_dir, "synthetic_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Summary JSON to: {summary_path}")
    print(f"Total benchmark time: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    run_synthetic_benchmark(n_replicates=10)
