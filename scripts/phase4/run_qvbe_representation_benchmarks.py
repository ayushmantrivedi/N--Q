"""
Comprehensive Feature Extraction, Freezing, Geometric Profiling, and Representation Benchmarks.
Phase 4 — Nāḍī-Q.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import lightgbm as lgb
import xgboost as xgb

from nadiq.models.quantum.qvbe_v2 import (
    QVBEV2Config,
    QuantumVariationalBiomarkerEncoderV2,
)

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set unbuffered stdout encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

DATA_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\ppmi_primary_benchmark_cohort_n1465.parquet"
MANIFEST_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\PARTICIPANT_SPLIT_MANIFEST.csv"
PARTITION_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\feature_partitions.json"
OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1])
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_limits[i]) & (y_prob <= bin_limits[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    f2 = (5 * prec * sens) / (4 * prec + sens) if (4 * prec + sens) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "balanced_acc": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "f2": float(f2),
        "brier": float(brier),
        "ece": float(ece),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def compute_spectral_diagnostics(Q: np.ndarray, name: str = "") -> Dict[str, Any]:
    cov = np.cov(Q, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals_pos = evals[evals > 1e-12]
    
    if len(evals_pos) > 0:
        p = evals_pos / np.sum(evals_pos)
        entropy = -np.sum(p * np.log(p + 1e-12))
        eff_rank = float(np.exp(entropy))
        cond_num = float(np.max(evals_pos) / np.min(evals_pos))
    else:
        eff_rank = 0.0
        cond_num = np.nan

    norms = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    Q_norm = Q / norms
    sim_matrix = np.dot(Q_norm, Q_norm.T)
    off_diag = sim_matrix[~np.eye(len(Q), dtype=bool)]

    return {
        "representation": name,
        "n_samples": Q.shape[0],
        "n_features": Q.shape[1],
        "effective_rank": eff_rank,
        "condition_number": cond_num,
        "off_diagonal_similarity_mean": float(np.mean(off_diag)),
        "off_diagonal_similarity_var": float(np.var(off_diag)),
        "feature_variance_mean": float(np.mean(np.var(Q, axis=0))),
    }


def train_eval_model(model_family: str, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, seed: int) -> Dict[str, Any]:
    if model_family == "lightgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=seed,
            verbose=-1,
            n_jobs=2
        )
    elif model_family == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2
        )
    elif model_family == "logistic_l1":
        clf = LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=500,
            random_state=seed
        )
    elif model_family == "logistic_l2":
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]
    metrics = evaluate_predictions(y_te, y_prob)
    return metrics


def run_benchmark():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 4: QVBE REPRESENTATION LEARNING & BENCHMARK SUITE")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading Cohort and Split Manifest...")
    df = pd.read_parquet(DATA_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(PARTITION_PATH, "r") as f:
        partitions = json.load(f)

    core_features = partitions["core_features_249"]
    target_col = partitions["target"]
    pats_all = df["PATNO"].values
    y_all = df[target_col].values.astype(int)
    X_raw_all = df[core_features].values

    print(f"Total Cohort: N = {len(df)}, Features = {X_raw_all.shape[1]}")
    print(f"Target Distribution: Positive = {(y_all == 1).sum()}, Negative = {(y_all == 0).sum()}")

    # Map PATNO to index
    pat_to_idx = {pat: idx for idx, pat in enumerate(pats_all)}

    seeds = [42, 101, 2024, 777, 999]
    n_splits = 5
    total_folds = len(seeds) * n_splits

    results_records: List[Dict[str, Any]] = []
    diagnostics_records: List[Dict[str, Any]] = []
    ablation_records: List[Dict[str, Any]] = []
    runtime_records: List[Dict[str, Any]] = []

    results_csv = os.path.join(OUTPUT_DIR, "REPRESENTATION_BENCHMARK_RESULTS.csv")
    diag_csv = os.path.join(OUTPUT_DIR, "GEOMETRIC_DIAGNOSTICS.csv")
    rt_csv = os.path.join(OUTPUT_DIR, "RUNTIME_AND_RESOURCE_DATA.csv")

    completed_splits = set()
    if os.path.exists(results_csv):
        try:
            df_exist = pd.read_csv(results_csv)
            results_records = df_exist.to_dict(orient="records")
            for (s, f), _ in df_exist.groupby(["seed", "fold"]):
                completed_splits.add((int(s), int(f)))
            print(f"[Resume] Found {len(completed_splits)} already completed folds in {results_csv}. Resuming remaining...")
        except Exception as e:
            print(f"[Resume] Could not parse existing results ({e}), starting fresh.")

    if os.path.exists(diag_csv):
        try:
            df_diag_exist = pd.read_csv(diag_csv)
            diagnostics_records = df_diag_exist.to_dict(orient="records")
        except Exception:
            pass

    fold_idx = 0
    t_benchmark_start = time.time()

    print(f"\n[Step 2] Executing 6 Canonical Conditions Across {total_folds} Folds...")

    for seed in seeds:
        for fold in range(n_splits):
            fold_idx += 1
            if (seed, fold) in completed_splits:
                print(f"  [Fold {fold_idx}/{total_folds}] Already completed (Seed={seed}, Fold={fold}), skipping...", flush=True)
                continue

            t_fold_start = time.time()
            tracemalloc.start()

            test_pats = set(manifest_df[(manifest_df["seed"] == seed) & (manifest_df["fold"] == fold)]["PATNO"])
            test_indices = np.array([pat_to_idx[p] for p in test_pats])
            train_indices = np.array([i for i in range(len(pats_all)) if i not in set(test_indices)])

            y_tr, y_te = y_all[train_indices], y_all[test_indices]
            X_raw_tr, X_raw_te = X_raw_all[train_indices], X_raw_all[test_indices]

            # Imputation & Preprocessing (Standard Scaler strictly on train)
            imputer = SimpleImputer(strategy="mean")
            X_tr_imp = imputer.fit_transform(X_raw_tr)
            X_te_imp = imputer.transform(X_raw_te)

            scaler_std = StandardScaler()
            X_tr_std = scaler_std.fit_transform(X_tr_imp)
            X_te_std = scaler_std.transform(X_te_imp)

            scaler_minmax = MinMaxScaler()
            X_tr_mm = scaler_minmax.fit_transform(X_tr_imp)
            X_te_mm = scaler_minmax.transform(X_te_imp)

            # -------------------------------------------------------------
            # CONDITION A: Raw Classical Features (D=249)
            # -------------------------------------------------------------
            for model_name in ["lightgbm", "xgboost", "logistic_l1"]:
                t_m0 = time.time()
                m = train_eval_model(model_name, X_tr_std, y_tr, X_te_std, y_te, seed)
                t_m1 = time.time()
                results_records.append({
                    "seed": seed, "fold": fold, "condition": "A_Raw_Classical",
                    "representation": "Classical_Standardized_D249", "dim": 249,
                    "model": model_name, "train_time_sec": t_m1 - t_m0, **m
                })

            # -------------------------------------------------------------
            # CONDITION B: Pretrained / Scaled Tabular (MinMax D=249)
            # -------------------------------------------------------------
            for model_name in ["lightgbm", "xgboost", "logistic_l2"]:
                t_m0 = time.time()
                m = train_eval_model(model_name, X_tr_mm, y_tr, X_te_mm, y_te, seed)
                t_m1 = time.time()
                results_records.append({
                    "seed": seed, "fold": fold, "condition": "B_Scaled_Tabular",
                    "representation": "Classical_MinMax_D249", "dim": 249,
                    "model": model_name, "train_time_sec": t_m1 - t_m0, **m
                })

            # -------------------------------------------------------------
            # CONDITION C: Classical Linear Reduction (PCA-8 & PCA-12)
            # -------------------------------------------------------------
            for n_comp in [8, 12]:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr_std)
                X_te_pca = pca.transform(X_te_std)

                for model_name in ["lightgbm", "xgboost", "logistic_l2"]:
                    t_m0 = time.time()
                    m = train_eval_model(model_name, X_tr_pca, y_tr, X_te_pca, y_te, seed)
                    t_m1 = time.time()
                    results_records.append({
                        "seed": seed, "fold": fold, "condition": f"C_Classical_PCA_{n_comp}",
                        "representation": f"Classical_PCA_k{n_comp}", "dim": n_comp,
                        "model": model_name, "train_time_sec": t_m1 - t_m0, **m
                    })

            # -------------------------------------------------------------
            # CONDITION D: Mode A — Fixed QVBE-v2 (8-Qubit M=23 & 12-Qubit M=35)
            # -------------------------------------------------------------
            for n_q in [8, 12]:
                cfg_mode_a = QVBEV2Config(
                    input_dim=249,
                    n_qubits=n_q,
                    n_layers=2,
                    measurement_basis="full",
                    seed=seed,
                    mode="fixed"
                )
                qvbe_a = QuantumVariationalBiomarkerEncoderV2(cfg_mode_a)
                t_fit_a = time.time()
                qvbe_a.fit(X_tr_std, y_tr)
                qvbe_a.freeze()
                Q_tr_a = qvbe_a.transform(X_tr_std)
                Q_te_a = qvbe_a.transform(X_te_std)
                t_fe_a = time.time() - t_fit_a

                # Compute diagnostics
                diag_tr = compute_spectral_diagnostics(Q_tr_a, f"ModeA_{n_q}Q_train")
                diag_te = compute_spectral_diagnostics(Q_te_a, f"ModeA_{n_q}Q_test")
                diagnostics_records.append({"seed": seed, "fold": fold, "split": "train", **diag_tr})
                diagnostics_records.append({"seed": seed, "fold": fold, "split": "test", **diag_te})

                # Standalone Quantum features
                for model_name in ["lightgbm", "xgboost", "logistic_l2"]:
                    m = train_eval_model(model_name, Q_tr_a, y_tr, Q_te_a, y_te, seed)
                    results_records.append({
                        "seed": seed, "fold": fold, "condition": f"D_ModeA_Fixed_{n_q}Q",
                        "representation": f"QVBE_ModeA_{n_q}Q", "dim": Q_tr_a.shape[1],
                        "model": model_name, "train_time_sec": t_fe_a, **m
                    })

                # Fused Features [X || Q]
                X_tr_fused_a = np.hstack([X_tr_std, Q_tr_a])
                X_te_fused_a = np.hstack([X_te_std, Q_te_a])
                for model_name in ["lightgbm", "xgboost"]:
                    m = train_eval_model(model_name, X_tr_fused_a, y_tr, X_te_fused_a, y_te, seed)
                    results_records.append({
                        "seed": seed, "fold": fold, "condition": f"D+_ModeA_Fused_{n_q}Q",
                        "representation": f"Fused_X_plus_QVBE_ModeA_{n_q}Q", "dim": X_tr_fused_a.shape[1],
                        "model": model_name, "train_time_sec": t_fe_a, **m
                    })

            # -------------------------------------------------------------
            # CONDITION E: Mode B — Trainable QVBE-v2 (8-Qubit M=23 & 12-Qubit M=35)
            # -------------------------------------------------------------
            for n_q in [8, 12]:
                cfg_mode_b = QVBEV2Config(
                    input_dim=249,
                    n_qubits=n_q,
                    n_layers=2,
                    measurement_basis="full",
                    seed=seed,
                    mode="trainable",
                    learning_rate=0.05,
                    n_iter=5
                )
                qvbe_b = QuantumVariationalBiomarkerEncoderV2(cfg_mode_b)
                t_fit_b = time.time()
                qvbe_b.fit(X_tr_std, y_tr)
                qvbe_b.freeze()
                Q_tr_b = qvbe_b.transform(X_tr_std)
                Q_te_b = qvbe_b.transform(X_te_std)
                t_fe_b = time.time() - t_fit_b

                # Compute diagnostics
                diag_b_tr = compute_spectral_diagnostics(Q_tr_b, f"ModeB_{n_q}Q_train")
                diag_b_te = compute_spectral_diagnostics(Q_te_b, f"ModeB_{n_q}Q_test")
                diagnostics_records.append({"seed": seed, "fold": fold, "split": "train", **diag_b_tr})
                diagnostics_records.append({"seed": seed, "fold": fold, "split": "test", **diag_b_te})

                # Standalone Quantum features
                for model_name in ["lightgbm", "xgboost", "logistic_l2"]:
                    m = train_eval_model(model_name, Q_tr_b, y_tr, Q_te_b, y_te, seed)
                    results_records.append({
                        "seed": seed, "fold": fold, "condition": f"E_ModeB_Trainable_{n_q}Q",
                        "representation": f"QVBE_ModeB_{n_q}Q", "dim": Q_tr_b.shape[1],
                        "model": model_name, "train_time_sec": t_fe_b, **m
                    })

                # Fused Features [X || Q*]
                X_tr_fused_b = np.hstack([X_tr_std, Q_tr_b])
                X_te_fused_b = np.hstack([X_te_std, Q_te_b])
                for model_name in ["lightgbm", "xgboost"]:
                    m = train_eval_model(model_name, X_tr_fused_b, y_tr, X_te_fused_b, y_te, seed)
                    results_records.append({
                        "seed": seed, "fold": fold, "condition": f"E+_ModeB_Fused_{n_q}Q",
                        "representation": f"Fused_X_plus_QVBE_ModeB_{n_q}Q", "dim": X_tr_fused_b.shape[1],
                        "model": model_name, "train_time_sec": t_fe_b, **m
                    })

                # -------------------------------------------------------------
                # CONDITION F: QVBE-v2 + SVD/PCA Decomposition (on 8-Qubit Mode B)
                # -------------------------------------------------------------
                if n_q == 8:
                    pca_q = PCA(n_components=12, random_state=seed)
                    Q_tr_svd = pca_q.fit_transform(Q_tr_b)
                    Q_te_svd = pca_q.transform(Q_te_b)

                    for model_name in ["lightgbm", "logistic_l2"]:
                        m = train_eval_model(model_name, Q_tr_svd, y_tr, Q_te_svd, y_te, seed)
                        results_records.append({
                            "seed": seed, "fold": fold, "condition": "F_QVBE_SVD_Decomp",
                            "representation": "QVBE_8Q_ModeB_SVD12", "dim": 12,
                            "model": model_name, "train_time_sec": t_fe_b, **m
                        })

            # Track peak memory and fold time
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            t_fold_end = time.time()
            runtime_records.append({
                "seed": seed, "fold": fold,
                "elapsed_sec": t_fold_end - t_fold_start,
                "peak_mem_mb": peak_mem / (1024 * 1024)
            })
            # Save intermediate results so progress is persisted
            pd.DataFrame(results_records).to_csv(results_csv, index=False)
            if len(diagnostics_records) > 0:
                pd.DataFrame(diagnostics_records).to_csv(diag_csv, index=False)
            if len(runtime_records) > 0:
                pd.DataFrame(runtime_records).to_csv(rt_csv, index=False)
            print(f"  [Fold {fold_idx}/{total_folds}] Done in {t_fold_end - t_fold_start:.1f}s (Peak Mem: {peak_mem / (1024*1024):.1f} MB)", flush=True)

    # -------------------------------------------------------------
    # ABLATION EXPERIMENTS (Executed on Seed 42, Fold 0)
    # -------------------------------------------------------------
    print("\n[Step 3] Running Systematic Ablations on Benchmark Reference Split...")
    test_pats_abl = set(manifest_df[(manifest_df["seed"] == 42) & (manifest_df["fold"] == 0)]["PATNO"])
    test_idx_abl = np.array([pat_to_idx[p] for p in test_pats_abl])
    train_idx_abl = np.array([i for i in range(len(pats_all)) if i not in set(test_idx_abl)])

    imputer_abl = SimpleImputer(strategy="mean")
    X_abl_tr_imp = imputer_abl.fit_transform(X_raw_all[train_idx_abl])
    X_abl_te_imp = imputer_abl.transform(X_raw_all[test_idx_abl])

    scaler_abl = StandardScaler()
    X_abl_tr = scaler_abl.fit_transform(X_abl_tr_imp)
    X_abl_te = scaler_abl.transform(X_abl_te_imp)
    y_abl_tr, y_abl_te = y_all[train_idx_abl], y_all[test_idx_abl]

    ablation_configs = [
        # Depth
        {"ablation_category": "Depth", "param": "L=1", "n_layers": 1, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        {"ablation_category": "Depth", "param": "L=2 (Standard)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        {"ablation_category": "Depth", "param": "L=4", "n_layers": 4, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        # Unitary choice
        {"ablation_category": "Unitary", "param": "U3 (Standard)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        {"ablation_category": "Unitary", "param": "Ry-Rz Fallback", "n_layers": 2, "use_u3": False, "use_qaf": True, "measurement_basis": "full", "shots": None},
        # QAF Layer
        {"ablation_category": "Attention", "param": "With QAF (Standard)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        {"ablation_category": "Attention", "param": "Without QAF", "n_layers": 2, "use_u3": True, "use_qaf": False, "measurement_basis": "full", "shots": None},
        # Measurement Basis
        {"ablation_category": "Basis", "param": "Z-only (M=8)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "z_only", "shots": None},
        {"ablation_category": "Basis", "param": "Dual-basis (M=16)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "dual_basis", "shots": None},
        {"ablation_category": "Basis", "param": "Full (M=23)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "full", "shots": None},
        # Finite Shots
        {"ablation_category": "Shots", "param": "Analytic (Infinite)", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "dual_basis", "shots": None},
        {"ablation_category": "Shots", "param": "Shots=1024", "n_layers": 2, "use_u3": True, "use_qaf": True, "measurement_basis": "dual_basis", "shots": 1024},
    ]

    for abl in ablation_configs:
        t0 = time.time()
        cfg = QVBEV2Config(
            input_dim=249,
            n_qubits=8,
            n_layers=abl["n_layers"],
            use_u3=abl["use_u3"],
            use_qaf=abl["use_qaf"],
            measurement_basis=abl["measurement_basis"],
            shots=abl["shots"],
            seed=42,
            mode="fixed"
        )
        enc = QuantumVariationalBiomarkerEncoderV2(cfg)

        if abl["shots"] is not None:
            X_tr_eval, y_tr_eval = X_abl_tr[:150], y_abl_tr[:150]
            X_te_eval, y_te_eval = X_abl_te[:50], y_abl_te[:50]
        else:
            X_tr_eval, y_tr_eval = X_abl_tr, y_abl_tr
            X_te_eval, y_te_eval = X_abl_te, y_abl_te

        enc.fit(X_tr_eval, y_tr_eval)
        enc.freeze()
        Q_tr = enc.transform(X_tr_eval)
        Q_te = enc.transform(X_te_eval)
        t_enc = time.time() - t0

        m_lgb = train_eval_model("lightgbm", Q_tr, y_tr_eval, Q_te, y_te_eval, seed=42)
        m_lr = train_eval_model("logistic_l2", Q_tr, y_tr_eval, Q_te, y_te_eval, seed=42)

        ablation_records.append({
            "category": abl["ablation_category"],
            "parameter_variant": abl["param"],
            "dim": Q_tr.shape[1],
            "feature_time_sec": t_enc,
            "lightgbm_auroc": m_lgb["auroc"],
            "lightgbm_auprc": m_lgb["auprc"],
            "lightgbm_f1": m_lgb["f1"],
            "logreg_auroc": m_lr["auroc"],
            "logreg_auprc": m_lr["auprc"],
            "logreg_f1": m_lr["f1"],
        })
        print(f"  Ablation [{abl['ablation_category']} - {abl['param']}]: Dim={Q_tr.shape[1]}, LightGBM AUROC={m_lgb['auroc']:.4f}, LogReg AUROC={m_lr['auroc']:.4f}")

    # -------------------------------------------------------------
    # EXPORT SUMMARY RESULTS & REPORTS
    # -------------------------------------------------------------
    print("\n[Step 4] Materializing Benchmark Datasets and Reports...")
    df_results = pd.DataFrame(results_records)
    results_csv = os.path.join(OUTPUT_DIR, "REPRESENTATION_BENCHMARK_RESULTS.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"  -> Saved: {results_csv} ({len(df_results)} rows)")

    df_diag = pd.DataFrame(diagnostics_records)
    diag_csv = os.path.join(OUTPUT_DIR, "GEOMETRIC_DIAGNOSTICS.csv")
    df_diag.to_csv(diag_csv, index=False)
    print(f"  -> Saved: {diag_csv} ({len(df_diag)} rows)")

    df_abl = pd.DataFrame(ablation_records)
    abl_csv = os.path.join(OUTPUT_DIR, "QVBE_ABLATION_RESULTS.csv")
    df_abl.to_csv(abl_csv, index=False)
    print(f"  -> Saved: {abl_csv} ({len(df_abl)} rows)")

    df_rt = pd.DataFrame(runtime_records)
    rt_csv = os.path.join(OUTPUT_DIR, "RUNTIME_AND_RESOURCE_DATA.csv")
    df_rt.to_csv(rt_csv, index=False)
    print(f"  -> Saved: {rt_csv} ({len(df_rt)} rows)")

    t_total = time.time() - t_benchmark_start
    print(f"\n================ BENCHMARK COMPLETED IN {t_total:.1f}s ({t_total/60:.2f} mins) ================\n", flush=True)

    print("\n[Step 5] Triggering Statistical Synthesis and Report Generation...", flush=True)
    try:
        from analyze_representation_results import load_and_analyze
        load_and_analyze()
    except Exception as e:
        print(f"Warning: Could not run automated analysis directly ({e}). Run analyze_representation_results.py manually.", flush=True)


if __name__ == "__main__":
    run_benchmark()
