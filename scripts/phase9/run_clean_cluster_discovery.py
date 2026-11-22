"""
NĀḌĪ-Q Phase 9: Clean (Target-Leak-Free) Clinical Subgroup Discovery.
Compares Unsupervised Trajectory Clustering vs. Leaky Supervised Clustering across k=2, 3, 4.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
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


def run_clean_clustering():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 9: LEAK-FREE CLINICAL SUBGROUP DISCOVERY")
    print("=" * 80)

    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)  # (341, 3, 35)
    y_prog = df["target_progression_5yr_mcid5"].values.astype(int)
    N = len(df)
    time_offsets = torch.tensor([[0.0, 0.5, 1.0]] * N, dtype=torch.float32)
    temp_t = torch.tensor(temporal_all, dtype=torch.float32)

    # 1. Unsupervised Discovery Representation (No labels)
    ctclm_disc = ContinualTemporalClinicalLatentModel(CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, seed=42))
    ctclm_disc.eval()
    with torch.no_grad():
        h_disc = ctclm_disc(temp_t, time_offsets=time_offsets).h_final.numpy()

    # 2. Supervised Representation (Phase 7 Leaky Protocol)
    ctclm_sup = ContinualTemporalClinicalLatentModel(CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, seed=42))
    head_sup = nn.Linear(16, 1)
    opt = optim.Adam(list(ctclm_sup.parameters()) + list(head_sup.parameters()), lr=0.01)
    crit = nn.BCEWithLogitsLoss()
    y_t = torch.tensor(y_prog, dtype=torch.float32).unsqueeze(1)
    for _ in range(35):
        opt.zero_grad()
        out = ctclm_sup(temp_t, time_offsets=time_offsets)
        loss = crit(head_sup(out.h_final), y_t)
        loss.backward()
        opt.step()
    ctclm_sup.eval()
    with torch.no_grad():
        h_sup = ctclm_sup(temp_t, time_offsets=time_offsets).h_final.numpy()

    # 3. Handcrafted trajectory features (Early motor slope + baseline)
    motor_bl = np.sum(temporal_all[:, 0, :], axis=1)
    motor_m12 = np.sum(temporal_all[:, 2, :], axis=1)
    motor_slope = motor_m12 - motor_bl
    h_handcrafted = np.column_stack([motor_bl, motor_slope])

    experiments = [
        ("Phase 7 Supervised h_temp (Target Leakage)", h_sup),
        ("Phase 9 Clean Unsupervised CTCLM (Leak-Free)", h_disc),
        ("Handcrafted Trajectory (Baseline + 12M Slope)", h_handcrafted)
    ]

    records = []

    for name, rep in experiments:
        print(f"\n--- Clustering: {name} ---")
        for k in [2, 3, 4]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = km.fit_predict(rep)
            sil = float(silhouette_score(rep, clusters))

            # Post-hoc evaluation of 5-year progression rates
            rates = []
            counts = []
            for c in range(k):
                mask = (clusters == c)
                n_c = int(np.sum(mask))
                r_c = float(np.mean(y_prog[mask]))
                rates.append(r_c)
                counts.append(n_c)

            max_rate = max(rates)
            min_rate = min(rates)
            spread = max_rate - min_rate
            
            # Chi-square test of progression distribution across clusters
            contingency = np.zeros((2, k))
            for c in range(k):
                mask = (clusters == c)
                contingency[0, c] = np.sum(y_prog[mask] == 1)
                contingency[1, c] = np.sum(y_prog[mask] == 0)
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)

            rates_str = ", ".join([f"C{c}:{rates[c]*100:.1f}% (N={counts[c]})" for c in range(k)])
            print(f"k={k}: Silhouette={sil:.3f} | Spread={spread*100:.1f}% (p={p_val:.4e}) | {rates_str}")

            records.append({
                "Representation": name,
                "k_clusters": k,
                "Silhouette_Score": sil,
                "Max_Progression_Rate": max_rate,
                "Min_Progression_Rate": min_rate,
                "Progression_Rate_Spread": spread,
                "Chi2_p_value": p_val,
                "Cluster_Details": rates_str
            })

    df_out = pd.DataFrame(records)
    csv_f = os.path.join(OUTPUT_DIR, "CLEAN_CLUSTER_RESULTS.csv")
    df_out.to_csv(csv_f, index=False)
    csv_doc = os.path.join(DOCS_DIR, "CLEAN_CLUSTER_RESULTS.csv")
    df_out.to_csv(csv_doc, index=False)
    print(f"\nResults saved to {csv_f} and {csv_doc}")


if __name__ == "__main__":
    run_clean_clustering()
