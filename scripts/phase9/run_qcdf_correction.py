"""
NĀḌĪ-Q Phase 9: QCDF Remediation and Multimodal Gating Benchmark.
Diagnoses Gate Collapse, Analyzes Modality Dominance, and Implements Corrected Temperature-Controlled QCDF.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath("src"))
from nadiq.models.quantum.qvbe_v2 import QVBEV2Config, QuantumVariationalBiomarkerEncoderV2
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_9"
DOCS_DIR = r"docs\phase9"
PARQUET_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
TEMPORAL_NPY_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy"


class OriginalAdaptiveGatedQCDF(nn.Module):
    """Legacy Phase 6 Gated QCDF Architecture."""
    def __init__(self, q_dim: int = 23, t_dim: int = 16, c_dim: int = 36, proj_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.1, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.proj_q = nn.Sequential(nn.Linear(q_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.proj_t = nn.Sequential(nn.Linear(t_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.proj_c = nn.Sequential(nn.Linear(c_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())

        gate_in_dim = (3 * proj_dim) + 3
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, t: torch.Tensor, c: torch.Tensor, avail: torch.Tensor):
        h_q = self.proj_q(q) * avail[:, 0:1]
        h_t = self.proj_t(t) * avail[:, 1:2]
        h_c = self.proj_c(c) * avail[:, 2:3]

        gate_input = torch.cat([h_q, h_t, h_c, avail], dim=-1)
        gate_logits = self.gate_net(gate_input)
        mask_penalty = (1.0 - avail) * -1e9
        alpha = torch.softmax(gate_logits + mask_penalty, dim=-1)

        h_fused = (alpha[:, 0:1] * h_q) + (alpha[:, 1:2] * h_t) + (alpha[:, 2:3] * h_c)
        logits = self.classifier(h_fused)
        return logits, alpha


class CorrectedQCDF(nn.Module):
    """
    Remediated QCDF Architecture.
    Fixes:
    - Independent LayerNorm per modality before projection to balance scales.
    - Clipped Temperature-Scaled Softmax: prevents gate collapse or uninformative branch domination.
    - Direct residual skip connection from high-signal temporal branch.
    """
    def __init__(self, q_dim: int = 23, t_dim: int = 16, c_dim: int = 36, proj_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.1, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_t = nn.LayerNorm(t_dim)
        self.norm_c = nn.LayerNorm(c_dim)

        self.proj_q = nn.Sequential(nn.Linear(q_dim, proj_dim), nn.GELU())
        self.proj_t = nn.Sequential(nn.Linear(t_dim, proj_dim), nn.GELU())
        self.proj_c = nn.Sequential(nn.Linear(c_dim, proj_dim), nn.GELU())

        self.gate_net = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

        # Residual combiner with temporal anchor
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, t: torch.Tensor, c: torch.Tensor, avail: torch.Tensor):
        h_q = self.proj_q(self.norm_q(q)) * avail[:, 0:1]
        h_t = self.proj_t(self.norm_t(t)) * avail[:, 1:2]
        h_c = self.proj_c(self.norm_c(c)) * avail[:, 2:3]

        gate_in = torch.cat([h_q, h_t, h_c], dim=-1)
        raw_logits = self.gate_net(gate_in)
        mask_penalty = (1.0 - avail) * -1e9
        
        # Temperature scaled softmax (temperature=1.5 for smoother distribution)
        alpha_raw = torch.softmax((raw_logits + mask_penalty) / 1.5, dim=-1)
        
        # Minimum clipping (10% floor per available modality)
        eps = 0.10
        num_avail = torch.clamp(avail.sum(dim=-1, keepdim=True), min=1.0)
        alpha = (1.0 - eps) * alpha_raw + (eps / num_avail) * avail

        # Fused latent vector
        h_fused = (alpha[:, 0:1] * h_q) + (alpha[:, 1:2] * h_t) + (alpha[:, 2:3] * h_c)
        # Residual connection from temporal latent
        h_total = h_fused + 0.5 * h_t
        logits = self.classifier(h_total)
        return logits, alpha


def run_qcdf_benchmark():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 9: QCDF REMEDIATION & MULTIMODAL ABLATION")
    print("=" * 80)

    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)
    y_prog = df["target_progression_5yr_mcid5"].values.astype(int)
    N = len(df)

    # 1. Feature Preprocessing
    bio_cols = [c for c in df.columns if any(k in c.lower() for k in ["asyn", "tau", "ptau", "abeta", "nfl", "saa"])]
    mri_cols = [c for c in df.columns if any(k in c.lower() for k in ["thick", "vol", "surf", "hippocampus", "ventricle"])]

    X_bio = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[bio_cols].values))
    X_mri = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[mri_cols].values))

    # Quantum representations (QVBE-v2)
    cfg_qvbe = QVBEV2Config(input_dim=X_bio.shape[1], n_qubits=8, n_layers=2, measurement_basis="full", seed=42)
    qvbe = QuantumVariationalBiomarkerEncoderV2(cfg_qvbe)
    qvbe.fit(X_bio)
    Q_bio = qvbe.transform(X_bio)

    # Temporal representations (CTCLM)
    time_offsets = torch.tensor([[0.0, 0.5, 1.0]] * N, dtype=torch.float32)
    ctclm = ContinualTemporalClinicalLatentModel(CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, seed=42))
    ctclm.eval()
    with torch.no_grad():
        h_temp = ctclm(torch.tensor(temporal_all, dtype=torch.float32), time_offsets=time_offsets).h_final.numpy()

    # Tensors
    Q_t = torch.tensor(Q_bio, dtype=torch.float32)
    T_t = torch.tensor(h_temp, dtype=torch.float32)
    C_t = torch.tensor(X_mri, dtype=torch.float32)
    avail_full = torch.ones(N, 3, dtype=torch.float32)
    y_t = torch.tensor(y_prog, dtype=torch.float32).unsqueeze(1)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    configs_to_test = [
        "CTCLM Temporal Alone (Unimodal)",
        "Quantum Biofluid Alone (Unimodal)",
        "Static MRI Alone (Unimodal)",
        "Naive Concatenation (All 3 Modalities)",
        "Original Adaptive Gated QCDF (Phase 6)",
        "Corrected QCDF (Phase 9 Remediated)",
        "Corrected QCDF (MRI Masked Ablation)"
    ]
    scores: Dict[str, List[float]] = {k: [] for k in configs_to_test}
    alphas_recorded: List[np.ndarray] = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(Q_bio)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        tr_y, te_y = y_prog[tr_idx], y_prog[te_idx]
        tr_yt = y_t[tr_idx]

        # 1. CTCLM Alone
        clf_t = LogisticRegression(C=1.0, max_iter=1000)
        clf_t.fit(h_temp[tr_idx], tr_y)
        scores["CTCLM Temporal Alone (Unimodal)"].append(roc_auc_score(te_y, clf_t.predict_proba(h_temp[te_idx])[:, 1]))

        # 2. Quantum Biofluid Alone
        clf_q = LogisticRegression(C=1.0, max_iter=1000)
        clf_q.fit(Q_bio[tr_idx], tr_y)
        scores["Quantum Biofluid Alone (Unimodal)"].append(roc_auc_score(te_y, clf_q.predict_proba(Q_bio[te_idx])[:, 1]))

        # 3. Static MRI Alone
        clf_c = LogisticRegression(C=1.0, max_iter=1000)
        clf_c.fit(X_mri[tr_idx], tr_y)
        scores["Static MRI Alone (Unimodal)"].append(roc_auc_score(te_y, clf_c.predict_proba(X_mri[te_idx])[:, 1]))

        # 4. Naive Concatenation
        X_concat = np.hstack([Q_bio, h_temp, X_mri])
        clf_cat = LogisticRegression(C=1.0, max_iter=1000)
        clf_cat.fit(X_concat[tr_idx], tr_y)
        scores["Naive Concatenation (All 3 Modalities)"].append(roc_auc_score(te_y, clf_cat.predict_proba(X_concat[te_idx])[:, 1]))

        # 5. Original Adaptive Gated QCDF
        model_orig = OriginalAdaptiveGatedQCDF(q_dim=Q_bio.shape[1], t_dim=h_temp.shape[1], c_dim=X_mri.shape[1], seed=42 + fold)
        opt_orig = optim.Adam(model_orig.parameters(), lr=0.01, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()

        model_orig.train()
        for ep in range(30):
            opt_orig.zero_grad()
            logits, alpha = model_orig(Q_t[tr_idx], T_t[tr_idx], C_t[tr_idx], avail_full[tr_idx])
            loss = crit(logits, tr_yt)
            loss.backward()
            opt_orig.step()

        model_orig.eval()
        with torch.no_grad():
            te_logits, te_alpha = model_orig(Q_t[te_idx], T_t[te_idx], C_t[te_idx], avail_full[te_idx])
            probs_orig = torch.sigmoid(te_logits).squeeze().numpy()
            alphas_recorded.append(te_alpha.numpy())
        scores["Original Adaptive Gated QCDF (Phase 6)"].append(roc_auc_score(te_y, probs_orig))

        # 6. Corrected QCDF (Full Trimodal)
        model_corr = CorrectedQCDF(q_dim=Q_bio.shape[1], t_dim=h_temp.shape[1], c_dim=X_mri.shape[1], seed=42 + fold)
        opt_corr = optim.Adam(model_corr.parameters(), lr=0.01, weight_decay=1e-4)

        model_corr.train()
        for ep in range(30):
            opt_corr.zero_grad()
            logits_c, _ = model_corr(Q_t[tr_idx], T_t[tr_idx], C_t[tr_idx], avail_full[tr_idx])
            loss = crit(logits_c, tr_yt)
            loss.backward()
            opt_corr.step()

        model_corr.eval()
        with torch.no_grad():
            te_logits_c, _ = model_corr(Q_t[te_idx], T_t[te_idx], C_t[te_idx], avail_full[te_idx])
            probs_corr = torch.sigmoid(te_logits_c).squeeze().numpy()
        scores["Corrected QCDF (Phase 9 Remediated)"].append(roc_auc_score(te_y, probs_corr))

        # 7. Corrected QCDF (MRI Masked Ablation)
        avail_masked = avail_full.clone()
        avail_masked[:, 2] = 0.0  # Mask out MRI
        with torch.no_grad():
            te_logits_m, _ = model_corr(Q_t[te_idx], T_t[te_idx], C_t[te_idx], avail_masked[te_idx])
            probs_masked = torch.sigmoid(te_logits_m).squeeze().numpy()
        scores["Corrected QCDF (MRI Masked Ablation)"].append(roc_auc_score(te_y, probs_masked))

    print("\n" + "=" * 80)
    print("QCDF BENCHMARK RESULTS (5-FOLD AUROC)")
    print("=" * 80)
    rows = []
    for k in configs_to_test:
        mean_auc = float(np.mean(scores[k]))
        std_auc = float(np.std(scores[k]))
        print(f"{k:45s}: {mean_auc:.4f} ± {std_auc:.4f}")
        rows.append({"Configuration": k, "AUROC_mean": mean_auc, "AUROC_std": std_auc})

    # Gate analysis
    all_alphas = np.vstack(alphas_recorded)  # (N, 3)
    mean_alphas = np.mean(all_alphas, axis=0)
    print(f"\nOriginal QCDF Mean Gating Weights [Quantum, Temporal, Static]: {mean_alphas}")

    df_out = pd.DataFrame(rows)
    csv_f = os.path.join(OUTPUT_DIR, "QCDF_CORRECTION_RESULTS.csv")
    df_out.to_csv(csv_f, index=False)
    csv_doc = os.path.join(DOCS_DIR, "QCDF_CORRECTION_RESULTS.csv")
    df_out.to_csv(csv_doc, index=False)
    print(f"\nResults saved to {csv_f} and {csv_doc}")


if __name__ == "__main__":
    run_qcdf_benchmark()
