"""
NĀḌĪ-Q Pre-Implementation Benchmark Validation:
Harness 3: PPMI Longitudinal Multimodal Evidence Discovery Benchmark.

Protocol:
- Authoritative PPMI Longitudinal Cohort (N = 341 participants)
- 3 Random Seeds (42, 123, 456) x 5-Fold Stratified Cross-Validation (15 folds)
- Cross-validation split strictly at the participant level (PATNO)
- Stratified by 5-year future progression (delta >= 5.0)
- Candidate feature pairs across 3 modalities (Molecular, Structural MRI, Motor):
    * Molecular: CSF alpha-synuclein, p-Tau181, amyloid-beta
    * Structural: Bilateral Caudate, Bilateral Putamen
    * Motor: UPDRS-III Total, Bradykinesia, Gait
- 8 Evidence Discovery Protocols:
    1. Raw_Pearson (Uncorrected p <= 0.10)
    2. Pearson_FDR_BH (Benjamini-Hochberg FDR q <= 0.05)
    3. MeTTa_FDR (MeTTa symbolic promotion with BH FDR q <= 0.05)
    4. Top_K_Heuristic (Top K=4 by training utility, no graph optimization)
    5. M0_Exact_Exhaustive (Exact QUBO global optimum, K=4)
    6. M3_Tabu_Local_Search (Tabu local quadratic search, K=4)
    7. QAOA_p1 (1-layer parameterized quantum ansatz)
    8. QAOA_p2 (2-layer parameterized quantum ansatz)
- Validation on Held-Out Test Folds:
    * Replicated edge count & Replication rate (p_test <= 0.10 and matching sign)
    * Effect direction consistency (sign(r_test) == sign(r_train))
    * Multimodal domain coverage (number of modality pairs)
    * Selection stability (Pairwise Jaccard across all 15 folds)
    * Energy & Optimality gap vs exact solver
    * Wall-clock runtime
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

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

# Paths
LOCAL_PPMI_COHORT = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
LOCAL_CACHE_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Computes Benjamini-Hochberg False Discovery Rate control.
    Returns boolean array indicating which hypotheses are rejected at FDR q <= alpha.
    """
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Find largest k such that P_(k) <= (k / m) * alpha
    k_max = -1
    for k in range(m - 1, -1, -1):
        rank = k + 1
        threshold = (rank / m) * alpha
        if sorted_p[k] <= threshold:
            k_max = k
            break

    is_significant_sorted = np.zeros(m, dtype=bool)
    if k_max >= 0:
        is_significant_sorted[: k_max + 1] = True

    is_significant = np.zeros(m, dtype=bool)
    is_significant[sorted_indices] = is_significant_sorted
    return is_significant.tolist()


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Computes Jaccard index between two sets."""
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 1.0
    return float(len(set_a & set_b) / union_len)


def run_ppmi_discovery_benchmark():
    print("=" * 80)
    print("HARNESS 3: PPMI LONGITUDINAL MULTIMODAL EVIDENCE DISCOVERY BENCHMARK")
    print("=" * 80)
    t_start = time.time()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/5] Loading PPMI multimodal cohort and cache...")
    df_cohort = pd.read_parquet(LOCAL_PPMI_COHORT)
    df_cache = pd.read_parquet(LOCAL_CACHE_PARQUET)

    pats = df_cohort["PATNO"].tolist()
    y_all_future = (
        (df_cohort["delta_NP3TOT_5yr"] - df_cohort["early_trajectory_delta_1yr"]).values >= 5.0
    ).astype(int)
    N = len(pats)
    print(f"Cohort size: N={N} participants | Future progression rate: {np.mean(y_all_future)*100:.1f}%")

    # Align baseline and follow-up records
    c341 = df_cache[df_cache["PATNO"].isin(pats)].copy()
    c341["visit_date_dt"] = pd.to_datetime(c341["visit_date"], errors="coerce")

    bl_df = c341[c341["EVENT_ID"] == "BL"].set_index("PATNO").reindex(pats)
    v04_df = c341[c341["EVENT_ID"] == "V04"].set_index("PATNO").reindex(pats)

    s_asyn_bl = bl_df["asyn"].fillna(bl_df["asyn"].median())
    s_asyn_v04 = v04_df["asyn"].combine_first(s_asyn_bl)

    s_ptau_bl = bl_df["ptau"].fillna(bl_df["ptau"].median())
    s_ptau_v04 = v04_df["ptau"].combine_first(s_ptau_bl)

    s_abeta_bl = bl_df["abeta"].fillna(bl_df["abeta"].median())
    s_abeta_v04 = v04_df["abeta"].combine_first(s_abeta_bl)

    s_caud_bl = bl_df["MIA_CAUDATE_BILAT"].fillna(bl_df["MIA_CAUDATE_BILAT"].mean())
    s_caud_v04 = v04_df["MIA_CAUDATE_BILAT"].combine_first(s_caud_bl)

    s_put_bl = bl_df["MIA_PUTAMEN_BILAT"].fillna(bl_df["MIA_PUTAMEN_BILAT"].mean())
    s_put_v04 = v04_df["MIA_PUTAMEN_BILAT"].combine_first(s_put_bl)

    s_cohort_np3 = df_cohort.set_index("PATNO")["NP3TOT_BL"].reindex(pats)
    s_cohort_delta = df_cohort.set_index("PATNO")["early_trajectory_delta_1yr"].reindex(pats)

    s_motor_bl = bl_df["updrs3_score"].combine_first(s_cohort_np3)
    s_motor_v04 = v04_df["updrs3_score"].combine_first(s_motor_bl + s_cohort_delta)

    s_brady_bl = df_cohort.set_index("PATNO")["NP3BRADY"].reindex(pats).fillna(0.0)
    s_brady_v04 = s_brady_bl + (s_motor_v04 - s_motor_bl) * 0.40

    s_gait_bl = df_cohort.set_index("PATNO")["NP3GAIT"].reindex(pats).fillna(0.0)
    s_gait_v04 = s_gait_bl + (s_motor_v04 - s_motor_bl) * 0.25

    dt_days = (v04_df["visit_date_dt"] - bl_df["visit_date_dt"]).dt.days.values
    dt_years = np.where(np.isnan(dt_days) | (dt_days <= 0), 365.25, dt_days) / 365.25

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

    collinear_redundancies = [
        (0, 1, 0.70),
        (3, 4, 0.70),
        (5, 6, 0.60),
        (6, 7, 0.65),
    ]

    incompatibility_pairs = [
        (5, 9),  # intermediate caudate->motor vs direct shortcut asyn->motor
    ]

    qubo_cfg = BiologicalQUBOConfig(
        lambda_redundancy=0.50,
        mu_contradiction=2.50,
        gamma_budget=1.50,
        budget_k=4,
    )

    methods = [
        "Raw_Pearson_p10",
        "Pearson_FDR_BH",
        "MeTTa_FDR",
        "Top_K_Heuristic",
        "M0_Exact_Exhaustive",
        "M3_Tabu_Local_Search",
        "QAOA_p1",
        "QAOA_p2",
    ]

    all_fold_records: List[Dict[str, Any]] = []
    selected_sets_by_method: Dict[str, List[set]] = {m: [] for m in methods}

    seeds = [42, 123, 456]
    print("\n[2/5] Executing 15-Fold PPMI Discovery Benchmark across 8 protocols...")

    for seed in seeds:
        print(f"\n--- Random Seed {seed} ---")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all_future)):
            # Normalize features strictly on train fold
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

            # Compute training fold and test fold stats for all 10 candidates
            raw_candidates_train = []
            test_stats = {}
            p_values_train = []

            for c_def in candidate_definitions:
                c_id = c_def["id"]
                s_feat = c_def["src"]
                d_feat = c_def["dst"]

                dx_s_tr = deltas_train[s_feat]
                dx_d_tr = deltas_train[d_feat]
                r_tr, p_tr = stats.pearsonr(dx_s_tr, dx_d_tr)
                se_tr = math.sqrt((1 - r_tr**2) / max(len(dx_s_tr) - 2, 1))
                p_values_train.append(p_tr)

                raw_candidates_train.append({
                    "id": c_id,
                    "src": s_feat,
                    "dst": d_feat,
                    "r": float(r_tr),
                    "p": float(p_tr),
                    "se": float(se_tr),
                    "lag": float(np.mean(dt_years[train_idx])),
                    "n": len(dx_s_tr),
                    "rel_type": c_def["type"],
                    "direction": 1 if r_tr >= 0 else -1,
                })

                dx_s_te = deltas_test[s_feat]
                dx_d_te = deltas_test[d_feat]
                r_te, p_te = stats.pearsonr(dx_s_te, dx_d_te)
                test_stats[c_id] = {
                    "r_test": float(r_te),
                    "p_test": float(p_te),
                    "replicated": bool(p_te <= 0.10 and (r_te * r_tr > 0)),
                    "direction_consistent": bool(r_te * r_tr > 0),
                    "type": c_def["type"],
                }

            # 1. Raw Pearson (p <= 0.10)
            t0 = time.perf_counter()
            sel_raw = {c["id"] for c in raw_candidates_train if c["p"] <= 0.10}
            t_raw = time.perf_counter() - t0

            # 2. Pearson + Benjamini-Hochberg FDR (q <= 0.05)
            t0 = time.perf_counter()
            fdr_mask = benjamini_hochberg_fdr(p_values_train, alpha=0.05)
            sel_fdr = {i for i, sig in enumerate(fdr_mask) if sig}
            t_fdr = time.perf_counter() - t0

            # 3. MeTTa + FDR
            t0 = time.perf_counter()
            compiler_fdr = MeTTaGraphCompiler(r_thresh=0.05, p_thresh=0.05, n_min=30)
            verified_cands_fdr, _, _ = compiler_fdr.compile_candidates(
                raw_candidates_train,
                collinear_redundancies=collinear_redundancies,
                incompatibility_pairs=incompatibility_pairs,
            )
            # Must also pass FDR
            sel_metta_fdr = {c.id for c in verified_cands_fdr if fdr_mask[c.id]}
            t_metta = time.perf_counter() - t0

            # 4. Standard MeTTa Graph Compilation & QUBO setup for optimization models
            compiler = MeTTaGraphCompiler(r_thresh=0.05, p_thresh=0.10, n_min=30)
            verified_cands, red_map, contra_pairs = compiler.compile_candidates(
                raw_candidates_train,
                collinear_redundancies=collinear_redundancies,
                incompatibility_pairs=incompatibility_pairs,
            )
            builder = BiologicalQUBOBuilder(
                candidates=verified_cands,
                redundancy_map=red_map,
                contradiction_pairs=contra_pairs,
                config=qubo_cfg,
            )

            # 4. Top-K Heuristic
            t0 = time.perf_counter()
            top_k_indices = np.argsort(-builder.U)[: qubo_cfg.budget_k].tolist()
            x_topk = np.zeros(builder.n_vars, dtype=np.int32)
            x_topk[top_k_indices] = 1
            e_topk = builder.evaluate_energy(x_topk)
            t_topk = time.perf_counter() - t0
            sel_topk = set(top_k_indices)

            # 5. M0: Exact Exhaustive Search
            classical = ClassicalSolvers(builder)
            out_m0 = classical.solve_exhaustive()
            sel_m0 = set(out_m0.selected_indices)

            # 6. M3: Tabu Search
            out_m3 = classical.solve_tabu_search(n_iters=150, tabu_tenure=4, seed=seed + fold)
            sel_m3 = set(out_m3.selected_indices)

            # Load existing Phase 13 fold runs if available to avoid re-simulating slow quantum circuits
            phase13_json = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase13\PHASE13_VERDICT.json"
            p13_records = {}
            if os.path.exists(phase13_json):
                with open(phase13_json, "r") as f:
                    p13_data = json.load(f)
                    for r in p13_data.get("detailed_fold_metrics", []):
                        p13_records[(r["seed"], r["fold"], r["method"])] = r

            # 7. QAOA p=1
            p13_q1 = p13_records.get((seed, fold, "QAOA_p1"))
            if p13_q1:
                bit_q1 = p13_q1["bitstring"]
                sel_q1 = {i for i, b in enumerate(bit_q1) if b == "1"}
                e_q1 = p13_q1["energy"]
                dt_q1 = p13_q1["runtime_sec"]
                gap_q1 = p13_q1["optimality_gap"]
            else:
                qaoa_p1 = QAOAOptimizer(qubo_builder=builder, p_layers=1, n_restarts=1, n_steps=10, lr=0.08, seed=seed + fold)
                out_q1 = qaoa_p1.optimize()
                sel_q1 = set(out_q1.selected_indices)
                e_q1 = out_q1.energy
                dt_q1 = out_q1.runtime_sec
                gap_q1 = out_q1.optimality_gap

            # 8. QAOA p=2
            p13_q2 = p13_records.get((seed, fold, "QAOA_p2"))
            if p13_q2:
                bit_q2 = p13_q2["bitstring"]
                sel_q2 = {i for i, b in enumerate(bit_q2) if b == "1"}
                e_q2 = p13_q2["energy"]
                dt_q2 = p13_q2["runtime_sec"]
                gap_q2 = p13_q2["optimality_gap"]
            else:
                qaoa_p2 = QAOAOptimizer(qubo_builder=builder, p_layers=2, n_restarts=1, n_steps=10, lr=0.08, seed=seed + fold)
                out_q2 = qaoa_p2.optimize()
                sel_q2 = set(out_q2.selected_indices)
                e_q2 = out_q2.energy
                dt_q2 = out_q2.runtime_sec
                gap_q2 = out_q2.optimality_gap

            protocol_runs = [
                ("Raw_Pearson_p10", sel_raw, None, t_raw, 0.0),
                ("Pearson_FDR_BH", sel_fdr, None, t_fdr, 0.0),
                ("MeTTa_FDR", sel_metta_fdr, None, t_metta, 0.0),
                ("Top_K_Heuristic", sel_topk, e_topk, t_topk, max(0.0, e_topk - out_m0.energy)),
                ("M0_Exact_Exhaustive", sel_m0, out_m0.energy, out_m0.runtime_sec, 0.0),
                ("M3_Tabu_Local_Search", sel_m3, out_m3.energy, out_m3.runtime_sec, out_m3.optimality_gap),
                ("QAOA_p1", sel_q1, e_q1, dt_q1, gap_q1),
                ("QAOA_p2", sel_q2, e_q2, dt_q2, gap_q2),
            ]

            for m_name, sel_set, energy, dt, gap in protocol_runs:
                selected_sets_by_method[m_name].append(sel_set)

                total_sel = len(sel_set)
                if total_sel > 0:
                    n_rep = sum(1 for idx in sel_set if test_stats[idx]["replicated"])
                    n_dir = sum(1 for idx in sel_set if test_stats[idx]["direction_consistent"])
                    rep_rate = n_rep / total_sel
                    dir_rate = n_dir / total_sel
                    unique_types = len({test_stats[idx]["type"] for idx in sel_set})
                else:
                    n_rep = 0
                    n_dir = 0
                    rep_rate = 0.0
                    dir_rate = 0.0
                    unique_types = 0

                rec = {
                    "seed": seed,
                    "fold": fold,
                    "method": m_name,
                    "n_selected": total_sel,
                    "n_replicated": n_rep,
                    "replication_fraction": f"{n_rep}/{total_sel}",
                    "replication_rate": rep_rate,
                    "direction_consistency": dir_rate,
                    "modality_coverage": unique_types,
                    "energy": energy if energy is not None else np.nan,
                    "optimality_gap": gap,
                    "runtime_ms": dt * 1000.0,
                }
                all_fold_records.append(rec)

            print(
                f"Seed {seed} Fold {fold}: "
                f"Raw={len(sel_raw)} edges | FDR={len(sel_fdr)} edges | "
                f"Exact={len(sel_m0)} (rep={all_fold_records[-4]['replication_rate']*100:.1f}%) | "
                f"QAOA-p2 gap={gap_q2:.3f}"
            )

    # Compute selection stability across all 15 folds
    stability_by_method = {}
    for m, sets_list in selected_sets_by_method.items():
        jaccards = []
        for s1, s2 in itertools.combinations(sets_list, 2):
            jaccards.append(jaccard_similarity(s1, s2))
        stability_by_method[m] = float(np.mean(jaccards)) if jaccards else 1.0

    df_results = pd.DataFrame(all_fold_records)
    csv_path = os.path.join(out_dir, "ppmi_discovery_results.csv")
    df_results.to_csv(csv_path, index=False)

    summary = df_results.groupby("method").agg({
        "n_selected": ["mean", "std"],
        "n_replicated": ["mean", "std"],
        "replication_rate": ["mean", "std"],
        "direction_consistency": ["mean", "std"],
        "modality_coverage": ["mean", "std"],
        "energy": ["mean", "std"],
        "optimality_gap": ["mean", "std"],
        "runtime_ms": ["mean", "std"],
    }).reset_index()

    summary.columns = [
        "method",
        "n_selected_mean", "n_selected_std",
        "n_replicated_mean", "n_replicated_std",
        "replication_rate_mean", "replication_rate_std",
        "dir_consistency_mean", "dir_consistency_std",
        "modality_coverage_mean", "modality_coverage_std",
        "energy_mean", "energy_std",
        "gap_mean", "gap_std",
        "runtime_ms_mean", "runtime_ms_std",
    ]
    summary["selection_stability"] = summary["method"].map(stability_by_method)

    print("\n" + "=" * 90)
    print("PPMI LONGITUDINAL MULTIMODAL EVIDENCE DISCOVERY SUMMARY (15 Folds)")
    print("=" * 90)
    cols = [
        "method", "n_selected_mean", "replication_rate_mean",
        "dir_consistency_mean", "modality_coverage_mean",
        "selection_stability", "gap_mean", "runtime_ms_mean"
    ]
    print(summary[cols].to_string(index=False))

    summary_dict = summary.to_dict(orient="records")
    summary_path = os.path.join(out_dir, "ppmi_discovery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Summary JSON to: {summary_path}")
    print(f"Total benchmark execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_ppmi_discovery_benchmark()
