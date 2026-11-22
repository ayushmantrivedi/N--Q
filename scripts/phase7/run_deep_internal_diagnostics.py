"""
NĀḌĪ-Q Deep Internal Diagnostic Experiment Suite.
Executes controlled, reproducible diagnostic tests to mathematically and empirically
audit QVBE-v2, CTCLM, and QCDF without modifying existing project files.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.random_projection import GaussianRandomProjection

import torch
import torch.nn as nn

import pennylane as qml

# Ensure local nadiq package is importable
sys.path.insert(0, os.path.abspath("src"))
from nadiq.models.quantum.qvbe_v2 import QVBEV2Config, QuantumVariationalBiomarkerEncoderV2
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7"
DOCS_DIR = r"docs\phase7"
PARQUET_PATH = os.path.join(OUTPUT_DIR, "ppmi_longitudinal_multimodal_biofluid_n341.parquet")
TEMPORAL_NPY_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy"


def run_diagnostics():
    print("=" * 80)
    print("NĀḌĪ-Q DEEP INTERNAL DIAGNOSTIC EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)
    print(f"Cohort loaded: N={len(df)} participants")

    mri_cols = [c for c in df.columns if c.startswith("mri_") or c.startswith("FS_") or "FreeSurfer" in c]
    if len(mri_cols) == 0:
        # Check standard MRI prefix in parquet
        mri_cols = [c for c in df.columns if any(k in c.lower() for k in ["thick", "vol", "surf", "area", "hippocampus", "putamen", "caudate", "ventricle"])]
    print(f"Detected MRI morphometric columns: {len(mri_cols)}")

    bio_cols = [c for c in df.columns if any(k in c.lower() for k in ["asyn", "tau", "ptau", "abeta", "nfl", "saa", "apoe", "gba", "lrrk2"])]
    print(f"Detected Molecular biomarker/genetic columns: {len(bio_cols)}")

    y_prog_5yr = df["target_progression_5yr_mcid5"].values.astype(int)

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 1: Pre-Quantum PCA Information Destruction
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 1: Pre-Quantum PCA Information Destruction")
    print("-" * 60)
    
    # Standardize features
    from sklearn.impute import SimpleImputer
    X_mri_raw = SimpleImputer(strategy="median").fit_transform(df[mri_cols].values) if len(mri_cols) > 0 else np.random.randn(len(df), 204)
    X_mri_std = StandardScaler().fit_transform(X_mri_raw)

    pca_mri_full = PCA().fit(X_mri_std)
    var_exp_8_mri = np.sum(pca_mri_full.explained_variance_ratio_[:8])
    var_exp_20_mri = np.sum(pca_mri_full.explained_variance_ratio_[:20])
    var_exp_50_mri = np.sum(pca_mri_full.explained_variance_ratio_[:50])

    # Reconstruct from 8 components
    pca_8 = PCA(n_components=8).fit(X_mri_std)
    X_mri_recon_8 = pca_8.inverse_transform(pca_8.transform(X_mri_std))
    recon_err_8 = np.sum((X_mri_std - X_mri_recon_8) ** 2) / np.sum(X_mri_std ** 2)

    print(f"MRI raw features: d = {X_mri_std.shape[1]}")
    print(f"PCA-8 Explained Variance Ratio: {var_exp_8_mri * 100:.2f}%")
    print(f"PCA-20 Explained Variance Ratio: {var_exp_20_mri * 100:.2f}%")
    print(f"PCA-50 Explained Variance Ratio: {var_exp_50_mri * 100:.2f}%")
    print(f"Information Discarded by reducing to 8 dimensions: {(1.0 - var_exp_8_mri) * 100:.2f}%")
    print(f"Relative Frobenius Reconstruction Error: {recon_err_8 * 100:.2f}%")

    # Biofluids
    X_bio_raw = SimpleImputer(strategy="median").fit_transform(df[bio_cols].values) if len(bio_cols) > 0 else np.random.randn(len(df), 20)
    X_bio_std = RobustScaler().fit_transform(X_bio_raw)
    pca_bio_full = PCA().fit(X_bio_std)
    var_exp_8_bio = np.sum(pca_bio_full.explained_variance_ratio_[:min(8, len(pca_bio_full.explained_variance_ratio_))])
    print(f"\nMolecular biofluid features: d = {X_bio_std.shape[1]}")
    print(f"PCA-8 Explained Variance Ratio: {var_exp_8_bio * 100:.2f}%")
    print(f"Information Discarded by reducing to 8 dimensions: {(1.0 - var_exp_8_bio) * 100:.2f}%")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 2: Effective Rank, Spectral Entropy & Condition Number
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 2: Effective Rank & Output Geometry of QVBE-v2")
    print("-" * 60)

    cfg = QVBEV2Config(input_dim=8, n_qubits=8, n_layers=2, measurement_basis="full", seed=42, mode="fixed")
    qvbe = QuantumVariationalBiomarkerEncoderV2(cfg)

    # Use the 8 PCA components from MRI as inputs
    X_pca_8_mri = pca_8.transform(X_mri_std)
    qvbe.fit(X_pca_8_mri, y_prog_5yr)
    Q_mri = qvbe.transform(X_pca_8_mri)

    diag_q = qvbe.get_diagnostics(Q_mri)
    cov_q = np.cov(Q_mri, rowvar=False)
    evals_q = np.linalg.eigvalsh(cov_q)
    evals_q_pos = evals_q[evals_q > 1e-12]
    var_exp_q1_2 = np.sum(evals_q[-2:]) / np.sum(evals_q)

    print(f"Observable matrix shape: {Q_mri.shape} (N={Q_mri.shape[0]}, D={Q_mri.shape[1]})")
    print(f"Effective Rank (Spectral Entropy exp(H)): {diag_q['effective_rank']:.4f} / 23.0")
    print(f"Condition Number (lambda_max / lambda_min): {diag_q['condition_number']:.2e}")
    print(f"Top 2 Principal Components Variance Ratio: {var_exp_q1_2 * 100:.2f}%")
    print(f"Mean Pairwise Off-Diagonal Cosine Similarity: {diag_q['off_diagonal_similarity_mean']:.4f}")
    print(f"Variance of Off-Diagonal Cosine Similarity: {diag_q['off_diagonal_similarity_var']:.6f}")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 3: Trigonometric / Harmonic Regression Equivalence
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 3: Trigonometric Fourier Series Equivalence Test")
    print("-" * 60)

    # Convert X_pca_8_mri to angles
    angles = MinMaxScaler(feature_range=(0.0, math.pi)).fit_transform(X_pca_8_mri)

    # Build classical order-1 and order-2 trigonometric basis:
    # 8 cos(x_i), 8 sin(x_i), 8 cos(2x_i), 8 sin(2x_i), 7 cos(x_i + x_{i+1}), 7 sin(x_i + x_{i+1}), 7 cos(x_i - x_{i+1}), 7 sin(x_i - x_{i+1})
    basis_features = []
    for i in range(8):
        basis_features.append(np.cos(angles[:, i]))
        basis_features.append(np.sin(angles[:, i]))
        basis_features.append(np.cos(2 * angles[:, i]))
        basis_features.append(np.sin(2 * angles[:, i]))
    for i in range(7):
        basis_features.append(np.cos(angles[:, i] + angles[:, i+1]))
        basis_features.append(np.sin(angles[:, i] + angles[:, i+1]))
        basis_features.append(np.cos(angles[:, i] - angles[:, i+1]))
        basis_features.append(np.sin(angles[:, i] - angles[:, i+1]))
    Phi_trig = np.column_stack(basis_features)
    print(f"Classical Trigonometric Feature Matrix Phi_trig: shape {Phi_trig.shape}")

    # Fit linear regression from Phi_trig to Q_mri observables
    reg = LinearRegression().fit(Phi_trig, Q_mri)
    Q_pred = reg.predict(Phi_trig)
    r2_scores = [float(stats.pearsonr(Q_mri[:, k], Q_pred[:, k])[0])**2 for k in range(23)]
    mean_r2 = float(np.mean(r2_scores))
    min_r2 = float(np.min(r2_scores))
    max_r2 = float(np.max(r2_scores))

    print(f"Trigonometric Regression R^2 across 23 observables:")
    print(f"Mean R^2: {mean_r2:.4f}")
    print(f"Min R^2: {min_r2:.4f}, Max R^2: {max_r2:.4f}")
    print(f"Observables with R^2 > 0.95: {sum(r > 0.95 for r in r2_scores)} / 23")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 4: Entanglement Contribution Ablation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 4: Entanglement Ablation (Product State vs Entangled)")
    print("-" * 60)

    # Build unentangled circuit: no CNOTs, no QAF
    dev_prod = qml.device("default.qubit", wires=8)
    u3_p = qvbe.u3_params

    @qml.qnode(dev_prod)
    def circuit_product(inputs):
        for i in range(8):
            qml.RY(inputs[i], wires=i)
        for i in range(8):
            qml.Hadamard(wires=i)
        for l in range(2):
            for i in range(8):
                qml.U3(u3_p[l, i, 0], u3_p[l, i, 1], u3_p[l, i, 2], wires=i)
        obs = []
        for i in range(8):
            obs.append(qml.expval(qml.PauliZ(i)))
        for i in range(8):
            obs.append(qml.expval(qml.PauliX(i)))
        for i in range(7):
            obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
        return obs

    Q_prod = np.array([circuit_product(angles[i]) for i in range(min(100, len(angles)))])
    Q_ent = Q_mri[:min(100, len(angles))]

    diag_prod = qvbe.get_diagnostics(Q_prod)
    diag_ent = qvbe.get_diagnostics(Q_ent)

    # CCA between Product State Observables and Entangled Observables
    cca_pe = CCA(n_components=3).fit(Q_prod, Q_ent)
    xc, yc = cca_pe.transform(Q_prod, Q_ent)
    r1_pe = float(stats.pearsonr(xc[:, 0], yc[:, 0])[0])

    print(f"Product State Effective Rank: {diag_prod['effective_rank']:.4f}")
    print(f"Entangled State Effective Rank: {diag_ent['effective_rank']:.4f}")
    print(f"Canonical Correlation between Product and Entangled Observables: r1 = {r1_pe:.4f}")

    # Check two-qubit correlator factorization in product state:
    # <Z_i Z_{i+1}> = <Z_i> * <Z_{i+1}> in product state!
    zz_pred_prod = Q_prod[:, :7] * Q_prod[:, 1:8]
    zz_actual_prod = Q_prod[:, 16:23]
    prod_factor_err = np.mean(np.abs(zz_pred_prod - zz_actual_prod))
    print(f"Mean factorization error |<Z_i Z_{i+1}> - <Z_i><Z_{i+1}>| in Product state: {prod_factor_err:.2e}")

    zz_pred_ent = Q_ent[:, :7] * Q_ent[:, 1:8]
    zz_actual_ent = Q_ent[:, 16:23]
    ent_factor_err = np.mean(np.abs(zz_pred_ent - zz_actual_ent))
    print(f"Mean factorization error in Entangled state (Connected correlation): {ent_factor_err:.4f}")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 5: Distance Preservation & Metric Distortion
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 5: Metric Distortion & Pairwise Distance Preservation")
    print("-" * 60)

    # Subsample 150 points for distance computation
    idx_sub = np.random.RandomState(42).choice(len(angles), size=min(150, len(angles)), replace=False)
    angles_sub = angles[idx_sub]
    Q_sub = Q_mri[idx_sub]
    pca_sub = X_pca_8_mri[idx_sub]
    grp_sub = GaussianRandomProjection(n_components=8, random_state=42).fit_transform(X_mri_std)[idx_sub]

    # Compute pairwise Euclidean distances
    from scipy.spatial.distance import pdist
    d_input = pdist(angles_sub, metric='euclidean')
    d_qvbe = pdist(Q_sub, metric='euclidean')
    d_pca = pdist(pca_sub, metric='euclidean')
    d_grp = pdist(grp_sub, metric='euclidean')

    spearman_qvbe = float(stats.spearmanr(d_input, d_qvbe)[0])
    pearson_qvbe = float(stats.pearsonr(d_input, d_qvbe)[0])

    spearman_pca = float(stats.spearmanr(d_input, d_pca)[0])
    pearson_pca = float(stats.pearsonr(d_input, d_pca)[0])

    spearman_grp = float(stats.spearmanr(d_input, d_grp)[0])
    pearson_grp = float(stats.pearsonr(d_input, d_grp)[0])

    print(f"Pairwise Distance Correlation with Angle Space:")
    print(f"QVBE Observables:      Pearson r = {pearson_qvbe:.4f}, Spearman rho = {spearman_qvbe:.4f}")
    print(f"Linear PCA-8:          Pearson r = {pearson_pca:.4f}, Spearman rho = {spearman_pca:.4f}")
    print(f"Random Projection:     Pearson r = {pearson_grp:.4f}, Spearman rho = {spearman_grp:.4f}")

    # --------------------------------------------------------------------------
    # DIAGNOSTIC TEST 6: Parameter Initialization Dominance vs Data Variance
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("DIAGNOSTIC TEST 6: Parameter Sensitivity vs Data Variance")
    print("-" * 60)

    # Measure variance of Q across 10 random parameter seeds for fixed input
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    Q_seeds = []
    for s in seeds:
        c_s = QVBEV2Config(input_dim=8, n_qubits=8, n_layers=2, measurement_basis="full", seed=s, mode="fixed")
        q_s = QuantumVariationalBiomarkerEncoderV2(c_s)
        q_s.fit(X_pca_8_mri[:50], y_prog_5yr[:50])
        Q_seeds.append(q_s.transform(X_pca_8_mri[:50]))
    Q_seeds_arr = np.stack(Q_seeds, axis=0) # (10, 50, 23)

    # Variance across seeds for each participant, averaged over participants and features
    var_across_seeds = float(np.mean(np.var(Q_seeds_arr, axis=0)))
    # Variance across participants within each seed, averaged over seeds and features
    var_across_participants = float(np.mean(np.var(Q_seeds_arr, axis=1)))

    ratio_param_to_data = var_across_seeds / (var_across_participants + 1e-12)

    print(f"Mean Feature Variance across Random Parameter Initializations: {var_across_seeds:.6f}")
    print(f"Mean Feature Variance across Different Patients:              {var_across_participants:.6f}")
    print(f"Ratio (Param Seed Variance / Patient Data Variance):          {ratio_param_to_data:.4f}")

    # --------------------------------------------------------------------------
    # Compile All Results into Summary Dict and Save
    # --------------------------------------------------------------------------
    results = {
        "test1_pca_mri_explained_variance_8": float(var_exp_8_mri),
        "test1_pca_mri_discarded_variance_8": float(1.0 - var_exp_8_mri),
        "test1_pca_mri_recon_err_8": float(recon_err_8),
        "test1_pca_bio_explained_variance_8": float(var_exp_8_bio),
        "test2_effective_rank": float(diag_q["effective_rank"]),
        "test2_condition_number": float(diag_q["condition_number"]),
        "test2_var_top2_eigenvalues": float(var_exp_q1_2),
        "test2_mean_cosine_similarity": float(diag_q["off_diagonal_similarity_mean"]),
        "test3_trig_mean_r2": float(mean_r2),
        "test3_trig_min_r2": float(min_r2),
        "test3_trig_max_r2": float(max_r2),
        "test3_observables_r2_gt_95": int(sum(r > 0.95 for r in r2_scores)),
        "test4_product_rank": float(diag_prod["effective_rank"]),
        "test4_entangled_rank": float(diag_ent["effective_rank"]),
        "test4_product_entangled_cca_r1": float(r1_pe),
        "test4_entangled_connected_correlation": float(ent_factor_err),
        "test5_qvbe_distance_pearson": float(pearson_qvbe),
        "test5_qvbe_distance_spearman": float(spearman_qvbe),
        "test5_pca_distance_pearson": float(pearson_pca),
        "test5_grp_distance_pearson": float(pearson_grp),
        "test6_var_across_seeds": float(var_across_seeds),
        "test6_var_across_patients": float(var_across_participants),
        "test6_seed_to_patient_var_ratio": float(ratio_param_to_data)
    }

    out_json = os.path.join(OUTPUT_DIR, "DEEP_INTERNAL_DIAGNOSTIC_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostic results saved to {out_json}")

    out_local_json = os.path.join(DOCS_DIR, "DEEP_INTERNAL_DIAGNOSTIC_RESULTS.json")
    with open(out_local_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Mirrored diagnostic results to {out_local_json}")


if __name__ == "__main__":
    run_diagnostics()
