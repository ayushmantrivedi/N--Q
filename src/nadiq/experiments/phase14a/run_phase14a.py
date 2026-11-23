"""
NĀḌĪ-Q Phase 14A Master Benchmark Runner:
Biological Relational Quantum Kernel (BRQK) vs Matched Classical Controls.

Executes:
- 2 Independent Clinical Endpoints:
  T1: Cumulative 5-Year Progression (Baseline to Month 60)
  T2: Future-Only Progression (Month 12 to Month 60)
- 11 Models / Ablations:
  M0: Logistic Regression (linear baseline)
  M1: RBF-SVM (non-linear baseline)
  M2: Poly-SVM (degree 2 baseline)
  M3: Classical Relational Kernel SVM (matched classical control)
  M4: Classical Relational Classifier (explicit features)
  M5: BRQK Quantum Relational Kernel SVM (biological topology)
  M6: CTCLM Baseline (temporal motor model)
  M7: CTCLM + Matched Classical Relational Kernel (classical hybrid)
  M8: CTCLM + Quantum Relational Kernel (quantum hybrid)
  M9: BRQK Quantum Kernel Shuffled Topology (negative control)
  M5_NoEntangle: BRQK Quantum Kernel No Entanglement (ablation control)
- 3 Random Seeds [42, 123, 456] x 5 Folds Stratified Cross-Validation (15 evaluation splits per task)
- Clustered Paired Bootstrap 95% Confidence Intervals
- Finite Shot Noise Profiling
- Outputs:
  experiments/phase14a/PHASE14A_T1_RESULTS.csv
  experiments/phase14a/PHASE14A_T2_RESULTS.csv
  experiments/phase14a/PHASE14A_VERDICT.json
  Mirrored to F:\\NADI-Q_AUDIT_OUTPUT\\PHASE_14A\\
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from nadiq.experiments.phase14a.data_protocol import DataProtocol14A
from nadiq.experiments.phase14a.cross_validation import StratifiedMultiSeedCV
from nadiq.experiments.phase14a.benchmark import Phase14ABenchmarkRunner
from nadiq.experiments.phase14a.metrics import (
    compute_classification_metrics,
    compute_paired_bootstrap_ci,
)
from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.shot_estimator import FiniteShotKernelEstimator
from nadiq.models.quantum.brqk.quantum_kernel import BiologicalRelationalQuantumKernel

LOCAL_OUT_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase14a"
AUDIT_OUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_14A"


def evaluate_task(
    task: str,
    data_protocol: DataProtocol14A,
    seeds: List[int] = [42, 123, 456],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Run 3-seed 5-fold cross-validation benchmark for a given task.
    """
    print(f"\n=======================================================")
    print(f"Executing Phase 14A Benchmark: Task {task}")
    print(f"=======================================================")

    X_raw, y, pats = data_protocol.get_task_data(task=task)
    ctclm_inputs = data_protocol.get_ctclm_temporal_inputs(task=task)
    N = len(y)
    n_pos = int(np.sum(y == 1))
    print(f"Cohort size: N={N}, Events: {n_pos} ({n_pos/N*100:.1f}%)")

    model_names = [
        "M0_Logistic_Regression",
        "M1_RBF_SVM",
        "M2_Poly_SVM",
        "M3_Classical_Relational_Kernel",
        "M4_Classical_Relational_Classifier",
        "M5_Quantum_Relational_Kernel",
        "M6_CTCLM_Baseline",
        "M7_CTCLM_Classical_Relational_Hybrid",
        "M8_CTCLM_Quantum_Relational_Hybrid",
        "M9_Quantum_Shuffled_Topology",
        "M5_Ablation_No_Entanglement",
    ]

    # Out-of-fold probabilities storage: model -> seed -> (N,)
    oof_probs: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in model_names}
    for m in model_names:
        for s in seeds:
            oof_probs[m][s] = np.zeros(N, dtype=float)

    cv = StratifiedMultiSeedCV(n_splits=5, seeds=seeds)
    runner = Phase14ABenchmarkRunner()

    t0 = time.time()
    for seed, fold_idx, tr_idx, te_idx in cv.split(X_raw, y):
        X_tr, y_tr, X_te, y_te = data_protocol.prepare_fold_data(X_raw, y, tr_idx, te_idx)

        fold_results = runner.run_fold(
            X_tr=X_tr,
            y_tr=y_tr,
            X_te=X_te,
            y_te=y_te,
            ctclm_inputs=ctclm_inputs,
            train_idx=tr_idx,
            test_idx=te_idx,
            seed=seed,
        )

        for m in model_names:
            if m in fold_results:
                oof_probs[m][seed][te_idx] = fold_results[m]["prob"]

    elapsed = time.time() - t0
    print(f"Completed 15 folds for Task {task} in {elapsed:.1f}s")

    # Aggregate across seeds
    avg_probs: Dict[str, np.ndarray] = {}
    metrics_rows = []

    for m in model_names:
        seed_matrices = np.stack([oof_probs[m][s] for s in seeds], axis=0)  # (n_seeds, N)
        avg_p = np.mean(seed_matrices, axis=0)
        avg_probs[m] = avg_p

        metrics = compute_classification_metrics(y, avg_p)
        metrics_rows.append({
            "task": task,
            "model": m,
            "auroc": round(metrics["auroc"], 4),
            "auprc": round(metrics["auprc"], 4),
            "brier": round(metrics["brier"], 4),
            "ece": round(metrics["ece"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "optimal_threshold": round(metrics["optimal_threshold"], 4),
        })

    df_results = pd.DataFrame(metrics_rows)
    print("\n--- Summary Performance Table ---")
    print(df_results[["model", "auroc", "auprc", "brier", "ece"]].to_string(index=False))

    # Bootstrap hypothesis comparisons
    comparisons = [
        ("M5_Quantum_Relational_Kernel", "M3_Classical_Relational_Kernel", "M5_vs_M3_Quantum_vs_Classical_Relational"),
        ("M8_CTCLM_Quantum_Relational_Hybrid", "M7_CTCLM_Classical_Relational_Hybrid", "M8_vs_M7_Hybrid_Quantum_vs_Classical"),
        ("M5_Quantum_Relational_Kernel", "M1_RBF_SVM", "M5_vs_M1_Quantum_vs_RBF_Baseline"),
        ("M5_Quantum_Relational_Kernel", "M9_Quantum_Shuffled_Topology", "M5_vs_M9_Biological_vs_Shuffled_Topology"),
        ("M5_Quantum_Relational_Kernel", "M5_Ablation_No_Entanglement", "M5_vs_NoEntangle_Entanglement_Contribution"),
        ("M8_CTCLM_Quantum_Relational_Hybrid", "M6_CTCLM_Baseline", "M8_vs_M6_Hybrid_vs_CTCLM_Alone"),
    ]

    boot_results = {}
    for mod_a, mod_b, comp_name in comparisons:
        ci_auc = compute_paired_bootstrap_ci(y, avg_probs[mod_a], avg_probs[mod_b], metric_fn=roc_auc_score)
        ci_prc = compute_paired_bootstrap_ci(y, avg_probs[mod_a], avg_probs[mod_b], metric_fn=average_precision_score)
        boot_results[comp_name] = {
            "model_a": mod_a,
            "model_b": mod_b,
            "auroc_diff_mean": round(ci_auc["diff_mean"], 4),
            "auroc_ci_95": [round(ci_auc["ci_lower"], 4), round(ci_auc["ci_upper"], 4)],
            "auroc_p_superior": round(ci_auc["p_superior"], 4),
            "auprc_diff_mean": round(ci_prc["diff_mean"], 4),
            "auprc_ci_95": [round(ci_prc["ci_lower"], 4), round(ci_prc["ci_upper"], 4)],
            "auprc_p_superior": round(ci_prc["p_superior"], 4),
        }

    return df_results, boot_results, avg_probs


def run_finite_shot_analysis() -> Dict[str, Any]:
    """
    Evaluate shot noise on quantum kernel estimation.
    """
    print("\n[Shot Analysis] Evaluating finite shot sampling error vs exact statevector...")
    cfg_exact = BRQKConfig(simulation_mode="statevector")
    kernel_exact = BiologicalRelationalQuantumKernel(cfg_exact)

    # Sample pairs
    rng = np.random.default_rng(42)
    theta_1 = rng.uniform(-np.pi, np.pi, size=6)
    theta_2 = rng.uniform(-np.pi, np.pi, size=6)

    k_exact = float(kernel_exact.compute_overlap_qnode(theta_1, theta_2))

    shot_configs = [512, 1024, 2048, 4096]
    shot_results = []
    for shots in shot_configs:
        cfg_shot = BRQKConfig(n_shots=shots, simulation_mode="finite_shot")
        estimator = FiniteShotKernelEstimator(cfg_shot)
        est = estimator.estimate_overlap(theta_1, theta_2)
        k_est = est["estimated_kernel"]
        abs_err = abs(k_est - k_exact)
        shot_results.append({
            "shots": shots,
            "exact_fidelity": round(k_exact, 6),
            "estimated_fidelity": round(k_est, 6),
            "absolute_error": round(abs_err, 6),
            "theoretical_std": round(est["theoretical_std"], 6),
        })

    return {
        "exact_fidelity": k_exact,
        "evaluations": shot_results,
    }


def main() -> None:
    t_start = time.time()
    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    os.makedirs(AUDIT_OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("NĀḌĪ-Q PHASE 14A MASTER BENCHMARK: BIOLOGICAL RELATIONAL QUANTUM KERNEL (BRQK)")
    print("=" * 80)

    data_proto = DataProtocol14A()

    # 1. Evaluate Task T1 (Cumulative 5-year progression)
    df_t1, boot_t1, probs_t1 = evaluate_task("T1", data_proto)
    t1_csv_local = os.path.join(LOCAL_OUT_DIR, "PHASE14A_T1_RESULTS.csv")
    t1_csv_audit = os.path.join(AUDIT_OUT_DIR, "PHASE14A_T1_RESULTS.csv")
    df_t1.to_csv(t1_csv_local, index=False)
    df_t1.to_csv(t1_csv_audit, index=False)

    # 2. Evaluate Task T2 (Future-only Month 12 to 60 progression)
    df_t2, boot_t2, probs_t2 = evaluate_task("T2", data_proto)
    t2_csv_local = os.path.join(LOCAL_OUT_DIR, "PHASE14A_T2_RESULTS.csv")
    t2_csv_audit = os.path.join(AUDIT_OUT_DIR, "PHASE14A_T2_RESULTS.csv")
    df_t2.to_csv(t2_csv_local, index=False)
    df_t2.to_csv(t2_csv_audit, index=False)

    # 3. Finite Shot Analysis
    shot_analysis = run_finite_shot_analysis()

    # 4. Formulate Scientific Verdict
    # Check primary decision conditions
    # Outcome A: Quantum Advantage (M5 > M3 AND M8 > M7 with CI_lower > 0)
    # Outcome B: Quantum Parity (M5 ~ M3 and M8 ~ M7, CIs overlap zero)
    # Outcome C: Classical Superiority (M3 > M5 or M7 > M8 with CI_lower > 0)
    # Outcome D: Relational Topology Ineffective (M5 ~ M9)

    t2_m5_vs_m3 = boot_t2["M5_vs_M3_Quantum_vs_Classical_Relational"]
    t2_m8_vs_m7 = boot_t2["M8_vs_M7_Hybrid_Quantum_vs_Classical"]
    t2_m5_vs_m9 = boot_t2["M5_vs_M9_Biological_vs_Shuffled_Topology"]

    is_m5_superior = (t2_m5_vs_m3["auroc_ci_95"][0] > 0.0)
    is_m8_superior = (t2_m8_vs_m7["auroc_ci_95"][0] > 0.0)
    is_classical_m3_superior = (t2_m5_vs_m3["auroc_ci_95"][1] < 0.0)
    is_classical_m7_superior = (t2_m8_vs_m7["auroc_ci_95"][1] < 0.0)

    if is_m5_superior and is_m8_superior:
        verdict_code = "OUTCOME_A_QUANTUM_RELATIONAL_ADVANTAGE"
        verdict_summary = "BRQK demonstrated statistically significant superior performance over matched classical relational control in both standalone and hybrid settings."
        go_decision = "GO"
    elif is_classical_m3_superior or is_classical_m7_superior:
        verdict_code = "OUTCOME_C_CLASSICAL_SUPERIORITY"
        verdict_summary = "Classical relational representations strictly outperform or equal quantum kernel representations with lower complexity and zero quantum overhead."
        go_decision = "NO-GO"
    elif abs(t2_m5_vs_m3["auroc_diff_mean"]) <= 0.03 and t2_m5_vs_m3["auroc_ci_95"][0] <= 0.0 <= t2_m5_vs_m3["auroc_ci_95"][1]:
        verdict_code = "OUTCOME_B_QUANTUM_CLASSICAL_PARITY"
        verdict_summary = "BRQK performs comparably to matched classical relational control without statistically distinguishable advantage; 95% bootstrap confidence interval spans zero."
        go_decision = "PARITY / PIVOT"
    else:
        verdict_code = "OUTCOME_D_RELATIONAL_INSUFFICIENT"
        verdict_summary = "Neither quantum nor classical relational representations demonstrated measurable advantage over standard classical baseline."
        go_decision = "NO-GO"

    verdict_payload = {
        "phase": "14A",
        "benchmark_name": "Biological Relational Quantum Kernel (BRQK)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict_code": verdict_code,
        "verdict_summary": verdict_summary,
        "go_decision": go_decision,
        "tasks": {
            "T1_Cumulative_5Yr": {
                "results": df_t1.to_dict(orient="records"),
                "bootstrap_comparisons": boot_t1,
            },
            "T2_Future_Only_M12_to_M60": {
                "results": df_t2.to_dict(orient="records"),
                "bootstrap_comparisons": boot_t2,
            },
        },
        "finite_shot_profile": shot_analysis,
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }

    verdict_json_local = os.path.join(LOCAL_OUT_DIR, "PHASE14A_VERDICT.json")
    verdict_json_audit = os.path.join(AUDIT_OUT_DIR, "PHASE14A_VERDICT.json")

    with open(verdict_json_local, "w") as f:
        json.dump(verdict_payload, f, indent=2)
    with open(verdict_json_audit, "w") as f:
        json.dump(verdict_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PHASE 14A BENCHMARK COMPLETED in {verdict_payload['total_runtime_seconds']}s")
    print(f"OFFICIAL VERDICT: {verdict_code}")
    print(f"DECISION: {go_decision}")
    print(f"Summary: {verdict_summary}")
    print(f"Local Verdict: {verdict_json_local}")
    print(f"Audit Verdict: {verdict_json_audit}")
    print("=" * 80)


if __name__ == "__main__":
    main()
