"""
NĀḌĪ-Q Phase 9: CTCLM Architectural Remediation and Trajectory Benchmark.
Compares CTCLM-Discovery (Unsupervised Autoregressive), CTCLM-Predictor (Supervised),
and Handcrafted Motor Trajectory Statistics across 5-fold cross-validation.
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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath("src"))
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_9"
DOCS_DIR = r"docs\phase9"
PARQUET_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
TEMPORAL_NPY_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy"


class CTCLMAutoregressiveDiscovery(nn.Module):
    """
    Unsupervised CTCLM Discovery Engine.
    Trained via self-supervised autoregressive sequence reconstruction:
    predicting Visit 3 (Month 12) from early visits without any downstream target labels.
    """
    def __init__(self, config: CTCLMConfig):
        super().__init__()
        self.ctclm = ContinualTemporalClinicalLatentModel(config)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.input_dim)
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor):
        out = self.ctclm(x, time_offsets=dt)
        recon = self.decoder(out.h_final)
        return out.h_final, recon


def compute_handcrafted_features(temporal_seq: np.ndarray) -> np.ndarray:
    """
    Extract deterministic clinical trajectory statistics:
    - Baseline motor score (sum across 35 features at visit 0)
    - Early 12-month rate of change: (Visit 2 - Visit 0) / 1.0 year
    - 6-month acceleration: (Visit 2 - 2*Visit 1 + Visit 0)
    - Inter-visit motor variance across visits
    - Peak severity minus baseline
    """
    N, T, D = temporal_seq.shape
    features = []
    for i in range(N):
        seq = temporal_seq[i]  # (3, 35)
        # Sum motor severity per visit
        motor_sum = np.sum(seq, axis=1)  # (3,)
        
        baseline_sev = motor_sum[0]
        v12_sev = motor_sum[2]
        v06_sev = motor_sum[1]
        
        slope_12m = v12_sev - baseline_sev
        slope_06m = (v06_sev - baseline_sev) / 0.5
        accel = (v12_sev - v06_sev) - (v06_sev - baseline_sev)
        traj_var = float(np.var(motor_sum))
        peak_diff = float(np.max(motor_sum) - baseline_sev)
        
        # Mean subscore slopes for top motor features
        top_feature_slopes = seq[2, :5] - seq[0, :5]
        
        row = [baseline_sev, slope_12m, slope_06m, accel, traj_var, peak_diff] + list(top_feature_slopes)
        features.append(row)
    return np.array(features, dtype=np.float32)


def run_ctclm_benchmark():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 9: CTCLM REMEDIATION & TRAJECTORY BENCHMARK")
    print("=" * 80)

    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)  # (341, 3, 35)
    y_prog = df["target_progression_5yr_mcid5"].values.astype(int)
    N = len(df)
    print(f"Cohort loaded: N={N} participants, progression rate = {y_prog.mean() * 100:.1f}%")

    # Handcrafted trajectory features
    handcrafted_feats = compute_handcrafted_features(temporal_all)
    print(f"Handcrafted trajectory features shape: {handcrafted_feats.shape}")

    # Set up 5-fold cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    dt_all = torch.tensor([[0.0, 0.5, 1.0]] * N, dtype=torch.float32)
    temp_t = torch.tensor(temporal_all, dtype=torch.float32)
    y_t = torch.tensor(y_prog, dtype=torch.float32).unsqueeze(1)

    models_to_test = [
        "Handcrafted Trajectory Statistics",
        "CTCLM-Discovery (Unsupervised Autoencoder)",
        "CTCLM-Predictor (Supervised 5-Year Target)",
        "Combined: Handcrafted + CTCLM-Discovery"
    ]
    cv_results = {m: [] for m in models_to_test}

    for fold, (tr_idx, te_idx) in enumerate(cv.split(temporal_all)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        tr_temp, te_temp = temp_t[tr_idx], temp_t[te_idx]
        tr_dt, te_dt = dt_all[tr_idx], dt_all[te_idx]
        tr_y, te_y = y_prog[tr_idx], y_prog[te_idx]
        tr_y_t = y_t[tr_idx]

        # 1. Model A: Handcrafted Trajectory Features
        scaler_hc = StandardScaler()
        hc_tr = scaler_hc.fit_transform(handcrafted_feats[tr_idx])
        hc_te = scaler_hc.transform(handcrafted_feats[te_idx])
        
        clf_hc = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_hc.fit(hc_tr, tr_y)
        probs_hc = clf_hc.predict_proba(hc_te)[:, 1]
        cv_results["Handcrafted Trajectory Statistics"].append(roc_auc_score(te_y, probs_hc))

        # 2. Model B: CTCLM-Discovery (Unsupervised Autoregressive Reconstruction)
        cfg_disc = CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, num_layers=1, seed=42 + fold)
        model_disc = CTCLMAutoregressiveDiscovery(cfg_disc)
        opt_disc = optim.Adam(model_disc.parameters(), lr=0.01, weight_decay=1e-4)
        crit_mse = nn.MSELoss()

        model_disc.train()
        for ep in range(25):
            opt_disc.zero_grad()
            h_tr, recon = model_disc(tr_temp, tr_dt)
            # Reconstruct Visit 3 features from hidden state
            target_v3 = tr_temp[:, 2, :]
            loss = crit_mse(recon, target_v3)
            loss.backward()
            opt_disc.step()

        model_disc.eval()
        with torch.no_grad():
            h_tr_disc, _ = model_disc(tr_temp, tr_dt)
            h_te_disc, _ = model_disc(te_temp, te_dt)
            h_tr_disc_np = h_tr_disc.numpy()
            h_te_disc_np = h_te_disc.numpy()

        clf_disc = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_disc.fit(h_tr_disc_np, tr_y)
        probs_disc = clf_disc.predict_proba(h_te_disc_np)[:, 1]
        cv_results["CTCLM-Discovery (Unsupervised Autoencoder)"].append(roc_auc_score(te_y, probs_disc))

        # 3. Model C: CTCLM-Predictor (Supervised)
        cfg_pred = CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, num_layers=1, seed=42 + fold)
        model_pred = ContinualTemporalClinicalLatentModel(cfg_pred)
        head_pred = nn.Linear(cfg_pred.latent_dim, 1)
        opt_pred = optim.Adam(list(model_pred.parameters()) + list(head_pred.parameters()), lr=0.01, weight_decay=1e-4)
        crit_bce = nn.BCEWithLogitsLoss()

        model_pred.train()
        head_pred.train()
        for ep in range(25):
            opt_pred.zero_grad()
            out_p = model_pred(tr_temp, time_offsets=tr_dt)
            logits = head_pred(out_p.h_final)
            loss = crit_bce(logits, tr_y_t)
            loss.backward()
            opt_pred.step()

        model_pred.eval()
        head_pred.eval()
        with torch.no_grad():
            out_te_p = model_pred(te_temp, time_offsets=te_dt)
            probs_pred = torch.sigmoid(head_pred(out_te_p.h_final)).squeeze().numpy()
        cv_results["CTCLM-Predictor (Supervised 5-Year Target)"].append(roc_auc_score(te_y, probs_pred))

        # 4. Model D: Combined Handcrafted + CTCLM-Discovery
        comb_tr = np.hstack([hc_tr, h_tr_disc_np])
        comb_te = np.hstack([hc_te, h_te_disc_np])
        clf_comb = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_comb.fit(comb_tr, tr_y)
        probs_comb = clf_comb.predict_proba(comb_te)[:, 1]
        cv_results["Combined: Handcrafted + CTCLM-Discovery"].append(roc_auc_score(te_y, probs_comb))

    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION SUMMARY (AUROC)")
    print("=" * 80)
    summary_rows = []
    for m in models_to_test:
        mean_auc = float(np.mean(cv_results[m]))
        std_auc = float(np.std(cv_results[m]))
        print(f"{m:45s}: {mean_auc:.4f} ± {std_auc:.4f}")
        summary_rows.append({
            "Model": m,
            "AUROC_mean": mean_auc,
            "AUROC_std": std_auc,
            "Fold_1": cv_results[m][0],
            "Fold_2": cv_results[m][1],
            "Fold_3": cv_results[m][2],
            "Fold_4": cv_results[m][3],
            "Fold_5": cv_results[m][4],
        })

    df_out = pd.DataFrame(summary_rows)
    csv_f = os.path.join(OUTPUT_DIR, "CTCLM_CORRECTION_RESULTS.csv")
    df_out.to_csv(csv_f, index=False)
    csv_doc = os.path.join(DOCS_DIR, "CTCLM_CORRECTION_RESULTS.csv")
    df_out.to_csv(csv_doc, index=False)
    print(f"\nSaved CTCLM correction results to {csv_f} and {csv_doc}")


if __name__ == "__main__":
    run_ctclm_benchmark()
