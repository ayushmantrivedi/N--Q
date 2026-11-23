"""
NĀḌĪ-Q Phase 13 Master Experiment:
MeTTa-Q Rapid Feasibility Test.

Primary Research Hypothesis:
Can quantum approximate optimization (QAOA) help MeTTa select informative,
non-redundant and temporally consistent biological relationship combinations
from an evidence-grown multimodal knowledge graph?

Protocol:
- Authoritative PPMI Longitudinal Cohort (N=341)
- 3 Random Seeds (42, 123, 456) x 5-Fold Stratified Cross-Validation
- Candidate relationships extracted strictly from training participants in each fold
- QUBO construction with symbolic redundancy, contradiction penalties, and budget K=4
- Comparison across 6 solvers:
    * M0: Exhaustive Search (Exact Global Optimum)
    * M1: Classical Simulated Annealing (SA)
    * M2: Greedy Evidence-Based Selection
    * M3: Classical Tabu Search / Local Quadratic Search
    * QAOA-p1: 1-Layer QAOA
    * QAOA-p2: 2-Layer QAOA
- Held-out biological validation on test folds:
    * Replication rate (replicated correlation and p <= 0.10)
    * Effect direction consistency (sign(r_test) == sign(r_train))
    * Selected set stability (Jaccard similarity across folds)
    * Optimality gap versus exact optimum
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch

from nadiq.models.metta_quantum.types import (
    CandidateRelationship,
    BiologicalQUBOConfig,
    QuantumSelectionOutput,
)
from nadiq.models.metta_quantum.compiler import MeTTaGraphCompiler
from nadiq.models.metta_quantum.qubo import BiologicalQUBOBuilder
from nadiq.models.metta_quantum.qaoa import QAOAOptimizer
from nadiq.models.metta_quantum.classical_solvers import ClassicalSolvers

# Paths
LOCAL_OUT_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase13"
AUDIT_OUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_13"
COHORT_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
CACHE_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
os.makedirs(AUDIT_OUT_DIR, exist_ok=True)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes Jaccard index between two sets."""
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 1.0
    return float(len(set_a & set_b) / union_len)


def main():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 13: MeTTa-Q RAPID FEASIBILITY TEST")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Load Authoritative Multimodal PPMI Cohort
    print("\n[1/6] Loading PPMI multimodal cohort and cache...")
    df_cohort = pd.read_parquet(COHORT_PARQUET)
    df_cache = pd.read_parquet(CACHE_PARQUET)

    pats = df_cohort["PATNO"].tolist()
    y_all_future = (
        (df_cohort["delta_NP3TOT_5yr"] - df_cohort["early_trajectory_delta_1yr"]).values >= 5.0
    ).astype(int)
    N = len(pats)
    print(f"Cohort size: N={N} participants")

    # Align baseline (BL) and V04 records
    c341 = df_cache[df_cache["PATNO"].isin(pats)].copy()
    c341["visit_date_dt"] = pd.to_datetime(c341["visit_date"], errors="coerce")

    bl_df = c341[c341["EVENT_ID"] == "BL"].set_index("PATNO").reindex(pats)
    v04_df = c341[c341["EVENT_ID"] == "V04"].set_index("PATNO").reindex(pats)

    # 1. Molecular features
    s_asyn_bl = bl_df["asyn"].fillna(bl_df["asyn"].median())
    s_asyn_v04 = v04_df["asyn"].combine_first(s_asyn_bl)

    s_ptau_bl = bl_df["ptau"].fillna(bl_df["ptau"].median())
    s_ptau_v04 = v04_df["ptau"].combine_first(s_ptau_bl)

    s_abeta_bl = bl_df["abeta"].fillna(bl_df["abeta"].median())
    s_abeta_v04 = v04_df["abeta"].combine_first(s_abeta_bl)

    # 2. Structural features
    s_caud_bl = bl_df["MIA_CAUDATE_BILAT"].fillna(bl_df["MIA_CAUDATE_BILAT"].mean())
    s_caud_v04 = v04_df["MIA_CAUDATE_BILAT"].combine_first(s_caud_bl)

    s_put_bl = bl_df["MIA_PUTAMEN_BILAT"].fillna(bl_df["MIA_PUTAMEN_BILAT"].mean())
    s_put_v04 = v04_df["MIA_PUTAMEN_BILAT"].combine_first(s_put_bl)

    # 3. Motor & Functional features
    s_cohort_np3 = df_cohort.set_index("PATNO")["NP3TOT_BL"].reindex(pats)
    s_cohort_delta = df_cohort.set_index("PATNO")["early_trajectory_delta_1yr"].reindex(pats)

    s_motor_bl = bl_df["updrs3_score"].combine_first(s_cohort_np3)
    s_motor_v04 = v04_df["updrs3_score"].combine_first(s_motor_bl + s_cohort_delta)

    s_brady_bl = df_cohort.set_index("PATNO")["NP3BRADY"].reindex(pats).fillna(0.0)
    s_brady_v04 = s_brady_bl + (s_motor_v04 - s_motor_bl) * 0.40  # bradykinesia subscore proxy

    s_gait_bl = df_cohort.set_index("PATNO")["NP3GAIT"].reindex(pats).fillna(0.0)
    s_gait_v04 = s_gait_bl + (s_motor_v04 - s_motor_bl) * 0.25  # gait subscore proxy

    dt_days = (v04_df["visit_date_dt"] - bl_df["visit_date_dt"]).dt.days.values
    dt_years = np.where(np.isnan(dt_days) | (dt_days <= 0), 365.25, dt_days) / 365.25

    # Assemble feature delta dictionary
    features_raw = {
        "asyn": (s_asyn_bl.values.astype(float), s_asyn_v04.values.astype(float)),
        "ptau": (s_ptau_bl.values.astype(float), s_ptau_v04.values.astype(float)),
        "abeta": (s_abeta_bl.values.astype(float), s_abeta_v04.values.astype(float)),
        "caudate": (s_caud_bl.values.astype(float), s_caud_v04.values.astype(float)),
        "putamen": (s_put_bl.values.astype(float), s_put_v04.values.astype(float)),
        "motor_total": (s_motor_bl.values.astype(float), s_motor_v04.values.astype(float)),
        "motor_brady": (s_brady_bl.values.astype(float), s_brady_v04.values.astype(float)),
        "motor_gait": (s_gait_bl.values.astype(float), s_gait_v04.values.astype(float)),
    }

    # Define the 10 candidate biological relationships across modalities
    candidate_definitions = [
        {"id": 0, "src": "asyn", "dst": "caudate", "type": "molecular_to_structural"},
        {"id": 1, "src": "ptau", "dst": "caudate", "type": "molecular_to_structural"},
        {"id": 2, "src": "abeta", "dst": "caudate", "type": "molecular_to_structural"},
        {"id": 3, "src": "asyn", "dst": "putamen", "type": "molecular_to_structural"},
        {"id": 4, "src": "ptau", "dst": "putamen", "type": "molecular_to_structural"},
        {"id": 5, "src": "caudate", "dst": "motor_total", "type": "structural_to_motor"},
        {"id": 6, "src": "putamen", "dst": "motor_total", "type": "structural_to_motor"},
        {"id": 7, "src": "putamen", "dst": "motor_brady", "type": "structural_to_motor"},
        {"id": 8, "src": "caudate", "dst": "motor_gait", "type": "structural_to_motor"},
        {"id": 9, "src": "asyn", "dst": "motor_total", "type": "molecular_to_motor"},
    ]

    # Explicit symbolic domain knowledge:
    # Collinear redundancies:
    # (0, 1): asyn->caudate vs ptau->caudate
    # (3, 4): asyn->putamen vs ptau->putamen
    # (5, 6): caudate->motor vs putamen->motor
    # (6, 7): putamen->motor vs putamen->brady
    collinear_redundancies = [
        (0, 1, 0.70),
        (3, 4, 0.70),
        (5, 6, 0.60),
        (6, 7, 0.65),
    ]

    # Contradictions / Mutual Exclusions:
    # Direct shortcut (Edge 9: asyn -> motor_total) contradicts intermediate cascade (Edge 0 and Edge 5)
    # in a strictly mediated parsimonious model
    incompatibility_pairs = [
        (5, 9),  # intermediate caudate->motor vs direct shortcut asyn->motor
    ]

    seeds = [42, 123, 456]
    qubo_cfg = BiologicalQUBOConfig(
        lambda_redundancy=0.50,
        mu_contradiction=2.50,
        gamma_budget=1.50,
        budget_k=4,
    )

    methods = [
        "M0_Exact_Exhaustive",
        "M1_Simulated_Annealing",
        "M2_Greedy_Selection",
        "M3_Tabu_Local_Search",
        "QAOA_p1",
        "QAOA_p2",
    ]

    all_fold_records: List[Dict[str, Any]] = []
    selected_sets_by_method: Dict[str, List[set]] = {m: [] for m in methods}

    print("\n[2/6] Executing Phase 13 MeTTa-Q Optimization across 3 Seeds & 5 Folds...")
    t_bench = time.time()

    for seed in seeds:
        print(f"\n==================== Random Seed {seed} ====================")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all_future)):
            # 1. Compute delta features strictly on training fold
            # Scale features within train fold
            deltas_train = {}
            deltas_test = {}

            for fname, (bl_vals, v04_vals) in features_raw.items():
                scaler = RobustScaler() if fname in ["asyn", "ptau", "abeta"] else StandardScaler()
                scaler.fit(bl_vals[train_idx].reshape(-1, 1))

                tr_bl = scaler.transform(bl_vals[train_idx].reshape(-1, 1)).flatten()
                tr_v04 = scaler.transform(v04_vals[train_idx].reshape(-1, 1)).flatten()
                deltas_train[fname] = tr_v04 - tr_bl

                te_bl = scaler.transform(bl_vals[test_idx].reshape(-1, 1)).flatten()
                te_v04 = scaler.transform(v04_vals[test_idx].reshape(-1, 1)).flatten()
                deltas_test[fname] = te_v04 - te_bl

            dt_train = dt_years[train_idx]
            dt_test = dt_years[test_idx]

            # 2. Extract genuine candidate correlations on training fold
            raw_candidates_train = []
            test_stats = {}

            for c_def in candidate_definitions:
                c_id = c_def["id"]
                s_feat = c_def["src"]
                d_feat = c_def["dst"]

                # Training fold correlation
                dx_s_tr = deltas_train[s_feat]
                dx_d_tr = deltas_train[d_feat]
                r_tr, p_tr = stats.pearsonr(dx_s_tr, dx_d_tr)
                se_tr = math.sqrt((1 - r_tr**2) / max(len(dx_s_tr) - 2, 1))

                raw_candidates_train.append({
                    "id": c_id,
                    "src": s_feat,
                    "dst": d_feat,
                    "r": float(r_tr),
                    "p": float(p_tr),
                    "se": float(se_tr),
                    "lag": float(np.mean(dt_train)),
                    "n": len(dx_s_tr),
                    "rel_type": c_def["type"],
                    "direction": 1 if r_tr >= 0 else -1,
                })

                # Held-out test fold correlation
                dx_s_te = deltas_test[s_feat]
                dx_d_te = deltas_test[d_feat]
                r_te, p_te = stats.pearsonr(dx_s_te, dx_d_te)
                test_stats[c_id] = {
                    "r_test": float(r_te),
                    "p_test": float(p_te),
                    "replicated": bool(p_te <= 0.10 and (r_te * r_tr > 0)),
                    "direction_consistent": bool(r_te * r_tr > 0),
                }

            # 3. MeTTa Symbolic Compilation (Training Fold Only)
            compiler = MeTTaGraphCompiler(r_thresh=0.05, p_thresh=0.10, n_min=30)
            verified_cands, red_map, contra_pairs = compiler.compile_candidates(
                raw_candidates_train,
                collinear_redundancies=collinear_redundancies,
                incompatibility_pairs=incompatibility_pairs,
            )

            # 4. Construct Biological QUBO
            builder = BiologicalQUBOBuilder(
                candidates=verified_cands,
                redundancy_map=red_map,
                contradiction_pairs=contra_pairs,
                config=qubo_cfg,
            )

            # 5. Execute Solvers
            classical = ClassicalSolvers(builder)

            # M0: Exact
            out_m0 = classical.solve_exhaustive()

            # M1: Simulated Annealing
            out_m1 = classical.solve_simulated_annealing(n_steps=500, seed=seed + fold)

            # M2: Greedy Evidence Selection
            out_m2 = classical.solve_greedy()

            # M3: Tabu Search
            out_m3 = classical.solve_tabu_search(n_iters=150, tabu_tenure=4, seed=seed + fold)

            # QAOA-p1
            qaoa_p1 = QAOAOptimizer(
                qubo_builder=builder,
                p_layers=1,
                n_restarts=3,
                n_steps=25,
                lr=0.08,
                seed=seed + fold,
            )
            out_q1 = qaoa_p1.optimize()

            # QAOA-p2
            qaoa_p2 = QAOAOptimizer(
                qubo_builder=builder,
                p_layers=2,
                n_restarts=3,
                n_steps=25,
                lr=0.08,
                seed=seed + fold,
            )
            out_q2 = qaoa_p2.optimize()

            fold_outputs = [out_m0, out_m1, out_m2, out_m3, out_q1, out_q2]

            # 6. Evaluate Biological Replication on Held-Out Test Fold
            for out in fold_outputs:
                sel_indices = set(out.selected_indices)
                selected_sets_by_method[out.method].append(sel_indices)

                # Replication metrics
                if len(sel_indices) > 0:
                    n_replicated = sum(1 for idx in sel_indices if test_stats[idx]["replicated"])
                    n_dir_consistent = sum(1 for idx in sel_indices if test_stats[idx]["direction_consistent"])
                    rep_rate = n_replicated / len(sel_indices)
                    dir_rate = n_dir_consistent / len(sel_indices)
                else:
                    rep_rate = 0.0
                    dir_rate = 0.0

                record = {
                    "seed": seed,
                    "fold": fold,
                    "method": out.method,
                    "bitstring": out.bitstring,
                    "energy": out.energy,
                    "utility_score": out.utility_score,
                    "optimality_gap": out.optimality_gap,
                    "is_feasible": out.is_feasible,
                    "budget_error": out.budget_error,
                    "num_contradictions": out.num_contradictions,
                    "replication_rate": rep_rate,
                    "direction_consistency": dir_rate,
                    "runtime_sec": out.runtime_sec,
                    "n_evaluations": out.n_evaluations,
                    "circuit_depth": out.circuit_depth,
                    "probability": out.probability,
                }
                all_fold_records.append(record)

            print(
                f"Seed {seed} Fold {fold}: Exact E={out_m0.energy:.4f}, "
                f"Tabu E={out_m3.energy:.4f} (gap={out_m3.optimality_gap:.4f}), "
                f"QAOA-p1 E={out_q1.energy:.4f} (gap={out_q1.optimality_gap:.4f}), "
                f"QAOA-p2 E={out_q2.energy:.4f} (gap={out_q2.optimality_gap:.4f})"
            )

    print(f"\n[3/6] Optimization completed across all folds in {time.time() - t_bench:.1f}s.")

    # Step 3: Compute Selection Stability (Pairwise Jaccard Across Folds)
    stability_by_method = {}
    for m, sets_list in selected_sets_by_method.items():
        jaccards = []
        for s1, s2 in itertools.combinations(sets_list, 2):
            jaccards.append(jaccard_similarity(s1, s2))
        stability_by_method[m] = float(np.mean(jaccards)) if jaccards else 1.0

    # Step 4: Aggregate Statistical Results
    df_results = pd.DataFrame(all_fold_records)

    summary = df_results.groupby("method").agg({
        "energy": ["mean", "std"],
        "utility_score": ["mean", "std"],
        "optimality_gap": ["mean", "std"],
        "is_feasible": "mean",
        "replication_rate": ["mean", "std"],
        "direction_consistency": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
        "n_evaluations": "mean",
        "circuit_depth": "mean",
    }).reset_index()

    summary.columns = [
        "method",
        "energy_mean", "energy_std",
        "utility_mean", "utility_std",
        "gap_mean", "gap_std",
        "feasibility_rate",
        "replication_mean", "replication_std",
        "dir_consistency_mean", "dir_consistency_std",
        "runtime_mean", "runtime_std",
        "n_evals_mean",
        "circuit_depth",
    ]
    summary["selection_stability"] = summary["method"].map(stability_by_method)

    print("\n" + "=" * 80)
    print("PHASE 13 BENCHMARK RESULTS (MeTTa-Q vs Classical Controls, 15 Folds)")
    print("=" * 80)
    cols_display = [
        "method", "energy_mean", "gap_mean", "feasibility_rate",
        "replication_mean", "dir_consistency_mean", "selection_stability", "runtime_mean"
    ]
    print(summary[cols_display].sort_values(by="energy_mean").to_string(index=False))

    # Step 5: Evaluate Official Verdict Criteria
    print("\n[5/6] Determining Official Scientific Verdict...")

    exact_gap = summary.set_index("method")["gap_mean"]["M0_Exact_Exhaustive"]
    tabu_gap = summary.set_index("method")["gap_mean"]["M3_Tabu_Local_Search"]
    sa_gap = summary.set_index("method")["gap_mean"]["M1_Simulated_Annealing"]
    greedy_gap = summary.set_index("method")["gap_mean"]["M2_Greedy_Selection"]

    q1_gap = summary.set_index("method")["gap_mean"]["QAOA_p1"]
    q2_gap = summary.set_index("method")["gap_mean"]["QAOA_p2"]

    q1_rep = summary.set_index("method")["replication_mean"]["QAOA_p1"]
    q2_rep = summary.set_index("method")["replication_mean"]["QAOA_p2"]
    tabu_rep = summary.set_index("method")["replication_mean"]["M3_Tabu_Local_Search"]
    exact_rep = summary.set_index("method")["replication_mean"]["M0_Exact_Exhaustive"]

    q_better_search = (q2_gap < tabu_gap - 1e-4) or (q1_gap < tabu_gap - 1e-4)
    q_better_bio = (max(q1_rep, q2_rep) > max(tabu_rep, exact_rep) + 0.05)
    classical_matches_or_beats = (tabu_gap <= min(q1_gap, q2_gap) + 1e-4) and (exact_rep >= max(q1_rep, q2_rep) - 1e-4)

    # Determine Verdict:
    if q_better_bio and q_better_search:
        verdict_status = "GO — METTA-Q QUANTUM RESEARCH"
    elif q_better_bio and not q_better_search:
        verdict_status = "GO — HYBRID SYMBOLIC-QUANTUM OPTIMIZATION, NO ADVANTAGE YET"
    elif classical_matches_or_beats:
        verdict_status = "CLASSICAL OPTIMIZATION SUFFICIENT"
    elif min(q1_gap, q2_gap) > 1.0:
        verdict_status = "NO-GO"
    else:
        verdict_status = "CLASSICAL OPTIMIZATION SUFFICIENT"

    print(f"\nOptimization Summary:")
    print(f"  Exact Optimum Energy: {summary.set_index('method')['energy_mean']['M0_Exact_Exhaustive']:.4f}")
    print(f"  Tabu Search Optimality Gap: {tabu_gap:.4f} (Runtime: {summary.set_index('method')['runtime_mean']['M3_Tabu_Local_Search']*1000:.1f}ms)")
    print(f"  Simulated Annealing Gap: {sa_gap:.4f} (Runtime: {summary.set_index('method')['runtime_mean']['M1_Simulated_Annealing']*1000:.1f}ms)")
    print(f"  QAOA-p1 Optimality Gap: {q1_gap:.4f} (Runtime: {summary.set_index('method')['runtime_mean']['QAOA_p1']*1000:.1f}ms)")
    print(f"  QAOA-p2 Optimality Gap: {q2_gap:.4f} (Runtime: {summary.set_index('method')['runtime_mean']['QAOA_p2']*1000:.1f}ms)")
    print(f"  Biological Replication Rate: Exact={exact_rep*100:.1f}%, Tabu={tabu_rep*100:.1f}%, QAOA-p1={q1_rep*100:.1f}%, QAOA-p2={q2_rep*100:.1f}%")

    # Step 6: Save Deliverables
    print("\n[6/6] Writing Deliverables and Audit Records...")

    csv_local = os.path.join(LOCAL_OUT_DIR, "PHASE13_RESULTS.csv")
    csv_audit = os.path.join(AUDIT_OUT_DIR, "PHASE13_RESULTS.csv")
    summary.to_csv(csv_local, index=False)
    summary.to_csv(csv_audit, index=False)

    verdict_payload = {
        "phase": "PHASE_13",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort_size": N,
        "n_seeds": len(seeds),
        "cv_folds": 5,
        "num_candidate_relationships": 10,
        "budget_k": qubo_cfg.budget_k,
        "verdict_status": verdict_status,
        "claims_distinction": {
            "search_quality_advantage": bool(q_better_search),
            "biological_replication_advantage": bool(q_better_bio),
            "computational_speedup": False,  # Exact statevector simulation on classical hardware
        },
        "model_summary": summary.to_dict(orient="records"),
        "detailed_fold_metrics": all_fold_records,
    }

    json_local = os.path.join(LOCAL_OUT_DIR, "PHASE13_VERDICT.json")
    json_audit = os.path.join(AUDIT_OUT_DIR, "PHASE13_VERDICT.json")
    with open(json_local, "w") as f:
        json.dump(verdict_payload, f, indent=2)
    with open(json_audit, "w") as f:
        json.dump(verdict_payload, f, indent=2)

    print(f"\nDeliverables written:")
    print(f"  [1] {csv_local} & {csv_audit}")
    print(f"  [2] {json_local} & {json_audit}")
    print("\n" + "=" * 80)
    print(f"OFFICIAL SCIENTIFIC VERDICT: {verdict_status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
