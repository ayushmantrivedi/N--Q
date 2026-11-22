"""
NĀḌĪ-Q Phase 9: Fair Quantum-Classical Comparison Benchmark Suite.
Compares QVBE-v3, QVBE-v2, Linear PCA, Kernel PCA, Random Projection, and Classical Trigonometric baselines
on real PPMI N=341 multimodal cohort under strictly identical evaluation conditions.
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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import torch

sys.path.insert(0, os.path.abspath("src"))
from nadiq.models.quantum.qvbe_v2 import QVBEV2Config, QuantumVariationalBiomarkerEncoderV2
from nadiq.models.qvbe_v3.config import QVBEv3Config
from nadiq.models.qvbe_v3.circuit import QVBEv3Circuit
from nadiq.models.qvbe_v3.trainer import QVBEv3Trainer
from nadiq.models.qvbe_v3.classical_controls import ClassicalTrigonometricControl
from nadiq.models.qvbe_v3.diagnostics import effective_rank
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_9"
DOCS_DIR = r"docs\phase9"
PARQUET_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
TEMPORAL_NPY_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy"


def run_benchmark():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 9: QUANTUM VS. CLASSICAL BENCHMARK SUITE")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)  # (341, 3, 35)
    print(f"Loaded multimodal cohort: N={len(df)} participants")

    y_prog_5yr = df["target_progression_5yr_mcid5"].values.astype(int)

    # Detect feature columns
    bio_cols = [c for c in df.columns if any(k in c.lower() for k in ["asyn", "tau", "ptau", "abeta", "nfl", "saa", "apoe", "gba", "lrrk2"])]
    mri_cols = [c for c in df.columns if any(k in c.lower() for k in ["thick", "vol", "surf", "area", "hippocampus", "putamen", "caudate", "ventricle"])]
    print(f"Molecular features: {len(bio_cols)}, Structural MRI features: {len(mri_cols)}")

    # Preprocess inputs
    X_bio_raw = SimpleImputer(strategy="median").fit_transform(df[bio_cols].values)
    X_bio_std = RobustScaler().fit_transform(X_bio_raw)

    X_mri_raw = SimpleImputer(strategy="median").fit_transform(df[mri_cols].values)
    X_mri_std = StandardScaler().fit_transform(X_mri_raw)

    # 2. Extract Unsupervised CTCLM Representation (No 5-year labels)
    print("\n--- Extracting Target-Leak-Free CTCLM Temporal Representations ---")
    time_offsets = torch.tensor([[0.0, 0.5, 1.0]] * len(df), dtype=torch.float32)
    temp_t = torch.tensor(temporal_all, dtype=torch.float32)
    ctclm_cfg = CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, num_layers=1, seed=42)
    ctclm = ContinualTemporalClinicalLatentModel(ctclm_cfg)
    ctclm.eval()
    with torch.no_grad():
        h_temp = ctclm(temp_t, time_offsets=time_offsets).h_final.numpy()
    print(f"Unsupervised h_temp shape: {h_temp.shape}")

    # 3. Build Representations
    print("\n--- Generating Quantum and Classical Representations ---")
    rep_dict: Dict[str, np.ndarray] = {}

    # A. QVBE-v2 (Mode A, 8-qubit frozen)
    cfg_v2 = QVBEV2Config(input_dim=X_bio_std.shape[1], n_qubits=8, n_layers=2, measurement_basis="full", seed=42, mode="fixed")
    qvbe_v2 = QuantumVariationalBiomarkerEncoderV2(cfg_v2)
    qvbe_v2.fit(X_bio_std)
    rep_dict["QVBE-v2 (Mode A Frozen)"] = qvbe_v2.transform(X_bio_std)

    # B. Linear PCA-8
    rep_dict["Linear PCA-8"] = PCA(n_components=8, random_state=42).fit_transform(X_bio_std)

    # C. Kernel PCA-8 (RBF)
    rep_dict["Kernel PCA-8 (RBF)"] = KernelPCA(n_components=8, kernel="rbf", gamma=1.0 / X_bio_std.shape[1], random_state=42).fit_transform(X_bio_std)

    # D. Gaussian Random Projection
    rep_dict["Random Projection-8"] = GaussianRandomProjection(n_components=8, random_state=42).fit_transform(X_bio_std)

    # E. Classical Trigonometric (Matched 23 dims)
    pca_8_scaled = MinMaxScaler(feature_range=(0.0, math.pi)).fit_transform(rep_dict["Linear PCA-8"])
    rep_dict["Classical Trig Expansion-23"] = ClassicalTrigonometricControl(target_dim=23).transform(pca_8_scaled)

    # F. QVBE-v3 (Data Re-uploading, InfoNCE trained)
    cfg_v3 = QVBEv3Config(
        n_qubits=8,
        n_layers=4,
        n_uploads=3,
        use_trainable_adapter=True,
        input_dim=X_bio_std.shape[1],
        seed=42
    )
    trainer_v3 = QVBEv3Trainer(cfg_v3, lr=0.01)
    x_bio_t = torch.tensor(X_bio_std, dtype=torch.float32)
    h_temp_t = torch.tensor(h_temp, dtype=torch.float32)

    # Train QVBE-v3 contrastively on molecular-temporal pairs (no progression labels!)
    trainer_v3.fit(x_bio_t, h_temp_t, epochs=15)
    trainer_v3.circuit.eval()
    with torch.no_grad():
        rep_dict["QVBE-v3 (InfoNCE Trained)"] = trainer_v3.circuit(x_bio_t).numpy()

    # 4. Benchmarking Metrics
    print("\n--- Running Comparative Evaluation ---")
    results_rows = []

    from scipy.spatial.distance import pdist

    d_input = pdist(X_bio_std, metric="euclidean")

    for name, rep in rep_dict.items():
        print(f"\nEvaluating: {name} (shape {rep.shape})")

        # Metric 1: Effective Rank
        eff_rank = effective_rank(rep)

        # Metric 2: Pairwise distance preservation (Pearson & Spearman)
        d_rep = pdist(rep, metric="euclidean")
        pearson_d = float(stats.pearsonr(d_input, d_rep)[0])
        spearman_d = float(stats.spearmanr(d_input, d_rep)[0])

        # Metric 3: Canonical Correlation with h_temp (CCA)
        n_comp = min(3, rep.shape[1], h_temp.shape[1])
        cca = CCA(n_components=n_comp, max_iter=1000)
        cca.fit(rep, h_temp)
        xc, yc = cca.transform(rep, h_temp)
        r1 = float(stats.pearsonr(xc[:, 0], yc[:, 0])[0])
        r2 = float(stats.pearsonr(xc[:, 1], yc[:, 1])[0]) if n_comp > 1 else 0.0

        # Permutation test for r1 (300 permutations)
        perm_r1s = []
        for _ in range(300):
            perm_idx = np.random.permutation(len(h_temp))
            h_perm = h_temp[perm_idx]
            cca_p = CCA(n_components=1, max_iter=500)
            cca_p.fit(rep, h_perm)
            xp, yp = cca_p.transform(rep, h_perm)
            perm_r1s.append(float(stats.pearsonr(xp[:, 0], yp[:, 0])[0]))
        p_perm = float(np.mean(np.array(perm_r1s) >= r1))

        # Metric 4: 5-Fold Cross-Validated Downstream AUROC (Progression 5yr)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for tr_idx, te_idx in cv.split(rep):
            X_tr, X_te = rep[tr_idx], rep[te_idx]
            y_tr, y_te = y_prog_5yr[tr_idx], y_prog_5yr[te_idx]
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            aurocs.append(roc_auc_score(y_te, probs))
        mean_auroc = float(np.mean(aurocs))
        std_auroc = float(np.std(aurocs))

        row = {
            "Representation": name,
            "Dimension": rep.shape[1],
            "Effective_Rank": eff_rank,
            "Distance_Pearson_r": pearson_d,
            "Distance_Spearman_rho": spearman_d,
            "CCA_r1": r1,
            "CCA_r2": r2,
            "CCA_p_perm": p_perm,
            "Downstream_5yr_AUROC_mean": mean_auroc,
            "Downstream_5yr_AUROC_std": std_auroc
        }
        results_rows.append(row)
        print(f"  Effective Rank: {eff_rank:.2f} | Distance Corr: {pearson_d:.4f} | CCA r1: {r1:.4f} (p={p_perm:.4f}) | AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

    df_res = pd.DataFrame(results_rows)
    csv_out = os.path.join(OUTPUT_DIR, "QUANTUM_CLASSICAL_COMPARISON_RESULTS.csv")
    df_res.to_csv(csv_out, index=False)
    csv_docs = os.path.join(DOCS_DIR, "QUANTUM_CLASSICAL_COMPARISON_RESULTS.csv")
    df_res.to_csv(csv_docs, index=False)
    print(f"\nResults successfully written to {csv_out} and {csv_docs}")


if __name__ == "__main__":
    run_benchmark()
