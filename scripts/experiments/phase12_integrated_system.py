"""
NĀḌĪ-Q Phase 12 Master Experiment:
Integrated Hybrid Intelligence System Benchmark & Scientific Verification.

Primary Objectives:
1. Integrate MeTTa biological KG, QVBE-v4, CTCLM, QCDF, and Q-XAI into one unified system.
2. Controlled 7-model comparison across 3 random seeds (42, 123, 456) and 5-fold Stratified CV:
   - M0: CTCLM Only (CTCLM temporal motor trajectory + static node features)
   - M1: Classical Temporal Graph + CTCLM (Classical relational branch + CTCLM)
   - M2: QVBE-v4 Only (Graph-conditioned quantum relational encoder alone)
   - M3: CTCLM + QVBE Simple Concatenation (Naive concatenation without dynamic gating)
   - M4: CTCLM + QVBE Adaptive QCDF (Full proposed hybrid architecture with dynamic gating)
   - M5: CTCLM + Classical Relational QCDF (Parameter-matched classical relational branch in QCDF)
   - M6: Zero-Quantum Suppressed (Full architecture with quantum branch strictly zeroed g_q=0)
3. Evaluate on both:
   - Primary: Cumulative 5-year progression (target_progression_5yr_mcid5, N=341)
   - Secondary: Future-only progression Month 12 to Month 60 (target_future_prog_mcid5, N=341)
4. Audit Q-XAI explanations & verify perturbation invariance.
5. Compute participant-clustered bootstrap 95% confidence intervals (M4 vs M5, M4 vs M6, M4 vs M0).
6. Output:
   - experiments/phase12/PHASE12_MODEL_METRICS.csv
   - experiments/phase12/PHASE12_VERDICT.json
   - Mirrored to F:\\NADI-Q_AUDIT_OUTPUT\\PHASE_12\\
"""

from __future__ import annotations

import copy
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pennylane as qml
import hyperon
from hyperon import MeTTa

from nadiq.models.integrated.config import NadiQIntegratedConfig
from nadiq.models.integrated.output import IntegratedOutput
from nadiq.models.integrated.fusion import AdaptiveQCDF
from nadiq.models.integrated.model import ClassicalRelationalBranch, NadiQIntegratedModel
from nadiq.models.integrated.explainability import NadiQXAIExplainer
from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel, CTCLMConfig

# Output Paths
LOCAL_OUT_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase12"
AUDIT_OUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_12"
COHORT_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
CACHE_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
os.makedirs(AUDIT_OUT_DIR, exist_ok=True)


# ==============================================================================
# Helper Functions: Calibration & Clustered Bootstrap
# ==============================================================================
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        b_lo, b_hi = bins[i], bins[i + 1]
        mask = (y_prob >= b_lo) & (y_prob < b_hi) if i < n_bins - 1 else (y_prob >= b_lo) & (y_prob <= b_hi)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def compute_clustered_bootstrap_ci(
    y_true: np.ndarray,
    probs_dict_a: Dict[int, np.ndarray],
    probs_dict_b: Dict[int, np.ndarray],
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes participant-clustered bootstrap confidence intervals.
    """
    rng = np.random.default_rng(seed)
    n_participants = len(y_true)
    seeds = list(probs_dict_a.keys())

    avg_a = np.mean([probs_dict_a[s] for s in seeds], axis=0)
    avg_b = np.mean([probs_dict_b[s] for s in seeds], axis=0)
    base_diff = float(metric_fn(y_true, avg_a) - metric_fn(y_true, avg_b))

    diffs = []
    for _ in range(n_boot):
        boot_idx = rng.choice(n_participants, size=n_participants, replace=True)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        try:
            m_a = metric_fn(y_true[boot_idx], avg_a[boot_idx])
            m_b = metric_fn(y_true[boot_idx], avg_b[boot_idx])
            diffs.append(m_a - m_b)
        except Exception:
            continue

    if len(diffs) == 0:
        return base_diff, base_diff, base_diff
    return base_diff, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ==============================================================================
# MeTTa Biological Knowledge Graph Reasoner
# ==============================================================================
class BiologicalKnowledgeGraphReasoner:
    def __init__(self, r_thresh: float = 0.05, p_thresh: float = 0.10, n_min: int = 30):
        self.r_thresh = r_thresh
        self.p_thresh = p_thresh
        self.n_min = n_min

    def discover_and_verify_graph(
        self,
        X_mol: np.ndarray,
        X_struct: np.ndarray,
        X_motor: np.ndarray,
        dX_mol: np.ndarray,
        dX_struct: np.ndarray,
        dX_motor: np.ndarray,
        dt_years: np.ndarray,
    ) -> Dict[str, Any]:
        corr_ms, p_ms = stats.pearsonr(dX_mol, dX_struct)
        corr_sf, p_sf = stats.pearsonr(dX_struct, dX_motor)
        corr_mf, p_mf = stats.pearsonr(dX_mol, dX_motor)

        se_ms = math.sqrt((1 - corr_ms**2) / max(len(dX_mol) - 2, 1))
        se_sf = math.sqrt((1 - corr_sf**2) / max(len(dX_struct) - 2, 1))
        se_mf = math.sqrt((1 - corr_mf**2) / max(len(dX_mol) - 2, 1))

        mean_dt = float(np.mean(dt_years))
        n_supp = len(dX_mol)

        candidate_edges = [
            {"src": "molecular", "dst": "structural", "lag": mean_dt, "r": float(corr_ms), "se": float(se_ms), "p": float(p_ms), "n": n_supp},
            {"src": "structural", "dst": "functional", "lag": mean_dt, "r": float(corr_sf), "se": float(se_sf), "p": float(p_sf), "n": n_supp},
            {"src": "molecular", "dst": "functional", "lag": mean_dt, "r": float(corr_mf), "se": float(se_mf), "p": float(p_mf), "n": n_supp},
        ]

        metta = MeTTa()
        metta.run("""
        (: Node Type)
        (: Edge Type)
        (: Status Type)

        (: Candidate Status)
        (: Provisional Status)
        (: Validated Status)

        (= (promote-edge $src $dst $lag $eff $pval $n)
           (if (and (>= (abs $eff) 0.05) (<= $pval 0.10) (>= $n 30))
               (if (<= $pval 0.05)
                   (Edge $src $dst $lag $eff $pval $n Validated)
                   (Edge $src $dst $lag $eff $pval $n Provisional))
               (Edge $src $dst $lag $eff $pval $n Candidate)))
        """)

        verified_edges = []
        for edge in candidate_edges:
            cmd = f"!(promote-edge {edge['src']} {edge['dst']} {edge['lag']:.2f} {edge['r']:.4f} {edge['p']:.4f} {edge['n']})"
            res = metta.run(cmd)
            status_str = "Candidate"
            if res and len(res) > 0 and len(res[0]) > 0:
                res_str = str(res[0][0])
                if "Validated" in res_str:
                    status_str = "Validated"
                elif "Provisional" in res_str:
                    status_str = "Provisional"

            verified_edges.append({
                **edge,
                "status": status_str,
                "provenance": "PPMI_train_correlation",
            })

        for e in verified_edges:
            metta.run(f"!(add-atom &self (Edge {e['src']} {e['dst']} {e['lag']:.2f} {e['r']:.4f} {e['p']:.4f} {e['n']} {e['status']}))")

        return {
            "edges": verified_edges,
            "weights": {
                "r_ms": float(corr_ms),
                "r_sf": float(corr_sf),
                "r_mf": float(corr_mf),
            },
            "se": {
                "se_ms": float(se_ms),
                "se_sf": float(se_sf),
                "se_mf": float(se_mf),
            },
            "lags": {
                "lag_ms": mean_dt,
                "lag_sf": mean_dt,
                "lag_mf": mean_dt,
            }
        }


# ==============================================================================
# Model Architecture for M3: Simple Concatenation Baseline
# ==============================================================================
class SimpleConcatModel(nn.Module):
    """
    M3: CTCLM + QVBE Simple Concatenation Baseline (No Adaptive Gating).
    Directly concatenates projected CTCLM, QVBE, and node features into an MLP classifier.
    """
    def __init__(self, config: NadiQIntegratedConfig):
        super().__init__()
        self.config = config
        self.qvbe = QVBEv4Encoder(config.qvbe_config)
        self.ctclm = ContinualTemporalClinicalLatentModel(config.ctclm_config)
        self.proj_qvbe = nn.Linear(config.qvbe_config.motif_embed_dim, 16)
        self.proj_ctclm = nn.Linear(config.ctclm_config.latent_dim, 16)
        self.proj_node = nn.Linear(6, 16)
        self.head = nn.Sequential(
            nn.Linear(16 * 3, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        ctclm_seq: torch.Tensor,
        ctclm_times: torch.Tensor,
        ctclm_mask: torch.Tensor,
        node_masks: torch.Tensor,
        elapsed_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = ctclm_seq.shape[0]
        # QVBE forward
        qvbe_out = self.qvbe(
            node_inputs=node_inputs,
            graph_edges=graph_edges,
            node_masks=node_masks,
            elapsed_times=elapsed_times,
        )
        # CTCLM forward
        ctclm_out = self.ctclm(ctclm_seq, ctclm_times, ctclm_mask)

        # Node states
        node_states_list = []
        for idx, name in enumerate(self.config.qvbe_config.node_names):
            x_raw, dx_raw = node_inputs[name]
            m = node_masks[:, idx : idx + 1]
            node_states_list.append(x_raw[:, :1] * m)
            node_states_list.append(dx_raw[:, :1] * m)
        node_flat = torch.cat(node_states_list, dim=-1)

        f_q = F.gelu(self.proj_qvbe(qvbe_out.motif_embedding))
        f_t = F.gelu(self.proj_ctclm(ctclm_out.h_final))
        f_n = F.gelu(self.proj_node(node_flat))

        concat_vec = torch.cat([f_q, f_t, f_n], dim=-1)
        return self.head(concat_vec).squeeze(-1)


# ==============================================================================
# Master Experiment Runner
# ==============================================================================
def main():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 12: MASTER INTEGRATED HYBRID INTELLIGENCE SYSTEM BENCHMARK")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Load Authoritative Multimodal PPMI Cohort
    print("\n[1/5] Loading authoritative longitudinal cohort and cache...")
    df_cohort = pd.read_parquet(COHORT_PARQUET)
    df_cache = pd.read_parquet(CACHE_PARQUET)

    pats = df_cohort["PATNO"].tolist()
    y_all_5yr = df_cohort["target_progression_5yr_mcid5"].values.astype(int)

    # Future-only Month 12 to Month 60 progression
    df_cohort["future_delta"] = df_cohort["delta_NP3TOT_5yr"] - df_cohort["early_trajectory_delta_1yr"]
    y_all_future = (df_cohort["future_delta"].values >= 5.0).astype(int)

    N = len(pats)
    print(f"Cohort size: N={N} participants")
    print(f"Cumulative 5-year progression events: {np.sum(y_all_5yr)} ({np.mean(y_all_5yr)*100:.1f}%)")
    print(f"Future-only Month 12-to-60 events   : {np.sum(y_all_future)} ({np.mean(y_all_future)*100:.1f}%)")

    # Align baseline and V04 records
    c341 = df_cache[df_cache["PATNO"].isin(pats)].copy()
    c341["visit_date_dt"] = pd.to_datetime(c341["visit_date"], errors="coerce")

    bl_df = c341[c341["EVENT_ID"] == "BL"].set_index("PATNO").reindex(pats)
    v04_df = c341[c341["EVENT_ID"] == "V04"].set_index("PATNO").reindex(pats)

    s_mol_bl = bl_df["asyn"].fillna(bl_df["asyn"].median())
    s_mol_v04 = v04_df["asyn"].combine_first(s_mol_bl)

    s_struct_bl = bl_df["MIA_CAUDATE_BILAT"].fillna(bl_df["MIA_CAUDATE_BILAT"].mean())
    s_struct_v04 = v04_df["MIA_CAUDATE_BILAT"].combine_first(s_struct_bl)

    s_cohort_np3 = df_cohort.set_index("PATNO")["NP3TOT_BL"].reindex(pats)
    s_cohort_delta = df_cohort.set_index("PATNO")["early_trajectory_delta_1yr"].reindex(pats)

    s_motor_bl = bl_df["updrs3_score"].combine_first(s_cohort_np3)
    s_motor_v04 = v04_df["updrs3_score"].combine_first(s_motor_bl + s_cohort_delta)

    raw_mol_bl = s_mol_bl.values.astype(float)
    raw_mol_v04 = s_mol_v04.values.astype(float)
    raw_struct_bl = s_struct_bl.values.astype(float)
    raw_struct_v04 = s_struct_v04.values.astype(float)
    raw_motor_bl = s_motor_bl.values.astype(float)
    raw_motor_v04 = s_motor_v04.values.astype(float)

    dt_days = (v04_df["visit_date_dt"] - bl_df["visit_date_dt"]).dt.days.values
    dt_years_raw = np.where(np.isnan(dt_days) | (dt_days <= 0), 365.25, dt_days) / 365.25

    seeds = [42, 123, 456]
    model_keys = [
        "M0_CTCLM_Only",
        "M1_ClassicalGraph_CTCLM",
        "M2_QVBE4_Only",
        "M3_CTCLM_QVBE_SimpleConcat",
        "M4_Adaptive_QCDF_Hybrid",
        "M5_ClassicalRelational_QCDF",
        "M6_ZeroQuantum_Suppressed",
    ]

    all_seed_results_primary = []
    all_seed_results_secondary = []
    all_seed_oof_probs_primary: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in model_keys}
    all_seed_oof_probs_secondary: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in model_keys}
    quantum_gate_weights_all: Dict[int, np.ndarray] = {}
    temporal_gate_weights_all: Dict[int, np.ndarray] = {}
    qxai_diagnostics_records = []

    reasoner = BiologicalKnowledgeGraphReasoner()

    print("\n[2/5] Executing Controlled 7-Model Benchmark across 3 Seeds and 5-Fold Stratified CV...")
    t_bench_start = time.time()

    for seed in seeds:
        print(f"\n==================== Random Seed {seed} ====================")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_probs_p = {m: np.zeros(N, dtype=float) for m in model_keys}
        oof_probs_s = {m: np.zeros(N, dtype=float) for m in model_keys}
        seed_gate_q = np.zeros(N, dtype=float)
        seed_gate_t = np.zeros(N, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all_5yr)):
            # 1. Fit Preprocessing Scalers inside training fold
            scaler_mol = RobustScaler()
            scaler_struct = StandardScaler()
            scaler_motor = StandardScaler()

            scaler_mol.fit(raw_mol_bl[train_idx].reshape(-1, 1))
            scaler_struct.fit(raw_struct_bl[train_idx].reshape(-1, 1))
            scaler_motor.fit(raw_motor_bl[train_idx].reshape(-1, 1))

            x_mol_bl = scaler_mol.transform(raw_mol_bl.reshape(-1, 1)).flatten()
            x_mol_v04 = scaler_mol.transform(raw_mol_v04.reshape(-1, 1)).flatten()
            dx_mol = x_mol_v04 - x_mol_bl

            x_struct_bl = scaler_struct.transform(raw_struct_bl.reshape(-1, 1)).flatten()
            x_struct_v04 = scaler_struct.transform(raw_struct_v04.reshape(-1, 1)).flatten()
            dx_struct = x_struct_v04 - x_struct_bl

            x_motor_bl = scaler_motor.transform(raw_motor_bl.reshape(-1, 1)).flatten()
            x_motor_v04 = scaler_motor.transform(raw_motor_v04.reshape(-1, 1)).flatten()
            dx_motor = x_motor_v04 - x_motor_bl

            # 2. Discover and Verify Knowledge Graph via MeTTa (Train fold only)
            graph_info = reasoner.discover_and_verify_graph(
                x_mol_bl[train_idx], x_struct_bl[train_idx], x_motor_bl[train_idx],
                dx_mol[train_idx], dx_struct[train_idx], dx_motor[train_idx],
                dt_years_raw[train_idx],
            )

            # Node representations
            node_inputs_all = {
                "molecular": (torch.tensor(x_mol_bl, dtype=torch.float32).unsqueeze(-1),
                              torch.tensor(dx_mol, dtype=torch.float32).unsqueeze(-1)),
                "structural": (torch.tensor(x_struct_bl, dtype=torch.float32).unsqueeze(-1),
                               torch.tensor(dx_struct, dtype=torch.float32).unsqueeze(-1)),
                "functional": (torch.tensor(x_motor_bl, dtype=torch.float32).unsqueeze(-1),
                               torch.tensor(dx_motor, dtype=torch.float32).unsqueeze(-1)),
            }

            def slice_node_inputs(indices):
                return {
                    k: (v[0][indices], v[1][indices])
                    for k, v in node_inputs_all.items()
                }

            train_node_inputs = slice_node_inputs(train_idx)
            test_node_inputs = slice_node_inputs(test_idx)

            # Temporal CTCLM sequence: 2 visits [BL, V04]
            ctclm_seq_all = torch.tensor(np.stack([
                x_motor_bl.reshape(N, 1),
                x_motor_v04.reshape(N, 1),
            ], axis=1), dtype=torch.float32) # (N, 2, 1)

            ctclm_times_all = torch.zeros(N, 2, dtype=torch.float32)
            ctclm_times_all[:, 1] = torch.tensor(dt_years_raw, dtype=torch.float32)

            ctclm_mask_all = torch.ones(N, 2, dtype=torch.bool)
            node_masks_all = torch.ones(N, 3, dtype=torch.float32)
            elapsed_t_all = torch.tensor(dt_years_raw, dtype=torch.float32).unsqueeze(-1)

            # Slices
            train_seq = ctclm_seq_all[train_idx]
            test_seq = ctclm_seq_all[test_idx]
            train_times = ctclm_times_all[train_idx]
            test_times = ctclm_times_all[test_idx]
            train_mask = ctclm_mask_all[train_idx]
            test_mask = ctclm_mask_all[test_idx]
            train_nmasks = node_masks_all[train_idx]
            test_nmasks = node_masks_all[test_idx]
            train_elapsed = elapsed_t_all[train_idx]
            test_elapsed = elapsed_t_all[test_idx]

            y_train_p = torch.tensor(y_all_5yr[train_idx], dtype=torch.float32)
            y_train_s = torch.tensor(y_all_future[train_idx], dtype=torch.float32)

            # Edges formatted for graph compiler
            graph_edges_fmt = [
                {
                    "source": e["src"],
                    "target": e["dst"],
                    "weight": e["r"],
                    "lag_months": e["lag"] * 12.0,
                    "se": e["se"],
                    "direction": 1,
                    "status": e["status"],
                }
                for e in graph_info["edges"]
            ]

            bce = nn.BCEWithLogitsLoss()

            # -----------------------------------------------------------------
            # Common Training Helper for NadiQIntegratedModel
            # -----------------------------------------------------------------
            def train_integrated_model(cfg, mode_str="full", force_q=None, epochs=20, lr=0.015, y_tr=y_train_p):
                m = NadiQIntegratedModel(cfg)
                opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
                m.train()
                for _ in range(epochs):
                    opt.zero_grad()
                    out = m(
                        node_inputs=train_node_inputs,
                        graph_edges=graph_edges_fmt,
                        ctclm_seq=train_seq,
                        ctclm_times=train_times,
                        ctclm_mask=train_mask,
                        node_masks=train_nmasks,
                        elapsed_times=train_elapsed,
                        mode=mode_str,
                        force_quantum_suppression=force_q,
                    )
                    loss = bce(out.logits, y_tr)
                    loss.backward()
                    opt.step()
                m.eval()
                return m

            # -----------------------------------------------------------------
            # M0: CTCLM Only
            # -----------------------------------------------------------------
            cfg_m0 = NadiQIntegratedConfig(enable_quantum_branch=False, enable_classical_relational=False)
            model_m0 = train_integrated_model(cfg_m0, mode_str="ctclm_only")
            with torch.no_grad():
                out_m0_test = model_m0(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    mode="ctclm_only",
                )
                oof_probs_p["M0_CTCLM_Only"][test_idx] = out_m0_test.probabilities.cpu().numpy()

            # -----------------------------------------------------------------
            # M1: Classical Temporal Graph + CTCLM
            # -----------------------------------------------------------------
            cfg_m1 = NadiQIntegratedConfig(enable_quantum_branch=False, enable_classical_relational=True)
            model_m1 = train_integrated_model(cfg_m1, mode_str="full")
            with torch.no_grad():
                out_m1_test = model_m1(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    mode="full",
                )
                oof_probs_p["M1_ClassicalGraph_CTCLM"][test_idx] = out_m1_test.probabilities.cpu().numpy()

            # -----------------------------------------------------------------
            # M2: QVBE-v4 Only
            # -----------------------------------------------------------------
            cfg_m2 = NadiQIntegratedConfig(enable_quantum_branch=True, enable_ctclm_branch=False, enable_classical_relational=False)
            model_m2 = train_integrated_model(cfg_m2, mode_str="qvbe_only")
            with torch.no_grad():
                out_m2_test = model_m2(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    mode="qvbe_only",
                )
                oof_probs_p["M2_QVBE4_Only"][test_idx] = out_m2_test.probabilities.cpu().numpy()

            # -----------------------------------------------------------------
            # M3: CTCLM + QVBE Simple Concatenation
            # -----------------------------------------------------------------
            cfg_m3 = NadiQIntegratedConfig()
            model_m3 = SimpleConcatModel(cfg_m3)
            opt_m3 = optim.AdamW(model_m3.parameters(), lr=0.015, weight_decay=1e-4)
            model_m3.train()
            for _ in range(20):
                opt_m3.zero_grad()
                logits_m3 = model_m3(
                    node_inputs=train_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=train_seq,
                    ctclm_times=train_times,
                    ctclm_mask=train_mask,
                    node_masks=train_nmasks,
                    elapsed_times=train_elapsed,
                )
                loss_m3 = bce(logits_m3, y_train_p)
                loss_m3.backward()
                opt_m3.step()
            model_m3.eval()
            with torch.no_grad():
                logits_m3_test = model_m3(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                )
                oof_probs_p["M3_CTCLM_QVBE_SimpleConcat"][test_idx] = torch.sigmoid(logits_m3_test).cpu().numpy()

            # -----------------------------------------------------------------
            # M4: CTCLM + QVBE Adaptive QCDF (Full Proposed Hybrid)
            # -----------------------------------------------------------------
            cfg_m4 = NadiQIntegratedConfig(
                enable_quantum_branch=True,
                enable_ctclm_branch=True,
                enable_classical_relational=False,
            )
            model_m4 = train_integrated_model(cfg_m4, mode_str="full")
            with torch.no_grad():
                out_m4_test = model_m4(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    mode="full",
                )
                oof_probs_p["M4_Adaptive_QCDF_Hybrid"][test_idx] = out_m4_test.probabilities.cpu().numpy()
                seed_gate_q[test_idx] = out_m4_test.quantum_gate_weight.view(-1).cpu().numpy()
                seed_gate_t[test_idx] = out_m4_test.temporal_gate_weight.view(-1).cpu().numpy()

            # -----------------------------------------------------------------
            # M5: CTCLM + Classical Relational QCDF
            # -----------------------------------------------------------------
            cfg_m5 = NadiQIntegratedConfig(
                enable_quantum_branch=False,
                enable_ctclm_branch=True,
                enable_classical_relational=True,
            )
            model_m5 = train_integrated_model(cfg_m5, mode_str="full")
            with torch.no_grad():
                out_m5_test = model_m5(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    mode="full",
                )
                oof_probs_p["M5_ClassicalRelational_QCDF"][test_idx] = out_m5_test.probabilities.cpu().numpy()

            # -----------------------------------------------------------------
            # M6: Zero-Quantum Suppressed (Ablation Control)
            # -----------------------------------------------------------------
            cfg_m6 = NadiQIntegratedConfig(
                enable_quantum_branch=True,
                enable_ctclm_branch=True,
                enable_classical_relational=False,
                force_quantum_suppression=True,
            )
            model_m6 = train_integrated_model(cfg_m6, mode_str="zero_quantum", force_q=True)
            with torch.no_grad():
                out_m6_test = model_m6(
                    node_inputs=test_node_inputs,
                    graph_edges=graph_edges_fmt,
                    ctclm_seq=test_seq,
                    ctclm_times=test_times,
                    ctclm_mask=test_mask,
                    node_masks=test_nmasks,
                    elapsed_times=test_elapsed,
                    force_quantum_suppression=True,
                )
                oof_probs_p["M6_ZeroQuantum_Suppressed"][test_idx] = out_m6_test.probabilities.cpu().numpy()

            # -----------------------------------------------------------------
            # Q-XAI Audit on Fold Test Set (on Model M4)
            # -----------------------------------------------------------------
            explainer_m4 = NadiQXAIExplainer(model_m4)
            exp_res = explainer_m4.explain_prediction(
                node_inputs=test_node_inputs,
                graph_edges=graph_edges_fmt,
                ctclm_seq=test_seq,
                ctclm_times=test_times,
                ctclm_mask=test_mask,
                node_masks=test_nmasks,
                elapsed_times=test_elapsed,
            )
            pert_check = explainer_m4.verify_perturbation_invariance(
                node_inputs=test_node_inputs,
                graph_edges=graph_edges_fmt,
                ctclm_seq=test_seq,
                ctclm_times=test_times,
                ctclm_mask=test_mask,
                node_masks=test_nmasks,
                elapsed_times=test_elapsed,
                target_modality_to_mask="molecular",
            )
            qxai_diagnostics_records.append({
                "seed": seed,
                "fold": fold,
                "mean_abs_delta_q": float(torch.abs(exp_res.quantum_attribution).mean().item()),
                "mean_abs_delta_t": float(torch.abs(exp_res.temporal_attribution).mean().item()),
                "mean_gate_q": float(exp_res.gate_weights["quantum"].mean().item()),
                "mean_gate_t": float(exp_res.gate_weights["temporal"].mean().item()),
                "perturbation_invariant": pert_check["is_invariant"],
                "max_pert_diff": pert_check["max_prob_difference"],
                "dominance_counts": pd.Series(exp_res.primary_dominance).value_counts().to_dict(),
            })

            # Secondary Target Optimization (M4, M5, M6)
            model_m4_sec = train_integrated_model(cfg_m4, mode_str="full", y_tr=y_train_s)
            model_m5_sec = train_integrated_model(cfg_m5, mode_str="full", y_tr=y_train_s)
            model_m6_sec = train_integrated_model(cfg_m6, mode_str="zero_quantum", force_q=True, y_tr=y_train_s)
            model_m0_sec = train_integrated_model(cfg_m0, mode_str="ctclm_only", y_tr=y_train_s)

            with torch.no_grad():
                oof_probs_s["M4_Adaptive_QCDF_Hybrid"][test_idx] = model_m4_sec(
                    test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed
                ).probabilities.cpu().numpy()
                oof_probs_s["M5_ClassicalRelational_QCDF"][test_idx] = model_m5_sec(
                    test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed
                ).probabilities.cpu().numpy()
                oof_probs_s["M6_ZeroQuantum_Suppressed"][test_idx] = model_m6_sec(
                    test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed, force_quantum_suppression=True
                ).probabilities.cpu().numpy()
                oof_probs_s["M0_CTCLM_Only"][test_idx] = model_m0_sec(
                    test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed, mode="ctclm_only"
                ).probabilities.cpu().numpy()

        # End of 5-fold CV for this seed
        quantum_gate_weights_all[seed] = seed_gate_q
        temporal_gate_weights_all[seed] = seed_gate_t

        for m in model_keys:
            all_seed_oof_probs_primary[m][seed] = oof_probs_p[m]
            all_seed_oof_probs_secondary[m][seed] = oof_probs_s[m]

            # Primary metrics
            auc_p = roc_auc_score(y_all_5yr, oof_probs_p[m])
            pr_p = average_precision_score(y_all_5yr, oof_probs_p[m])
            brier_p = brier_score_loss(y_all_5yr, oof_probs_p[m])
            ece_p = compute_ece(y_all_5yr, oof_probs_p[m])
            all_seed_results_primary.append({
                "seed": seed, "model": m, "target": "primary_5yr",
                "auroc": auc_p, "auprc": pr_p, "brier": brier_p, "ece": ece_p
            })

            # Secondary metrics (where trained)
            if np.any(oof_probs_s[m] > 0):
                auc_s = roc_auc_score(y_all_future, oof_probs_s[m])
                pr_s = average_precision_score(y_all_future, oof_probs_s[m])
                brier_s = brier_score_loss(y_all_future, oof_probs_s[m])
                ece_s = compute_ece(y_all_future, oof_probs_s[m])
                all_seed_results_secondary.append({
                    "seed": seed, "model": m, "target": "secondary_future",
                    "auroc": auc_s, "auprc": pr_s, "brier": brier_s, "ece": ece_s
                })

        print(f"Seed {seed} complete. M4 AUROC={auc_p:.4f}, M5 AUROC={roc_auc_score(y_all_5yr, oof_probs_p['M5_ClassicalRelational_QCDF']):.4f}, Mean g_quantum={np.mean(seed_gate_q):.4f}")

    print(f"\n[3/5] Benchmark completed in {time.time() - t_bench_start:.1f}s.")

    # Step 3: Aggregate Metrics
    df_res_p = pd.DataFrame(all_seed_results_primary)
    summary_primary = df_res_p.groupby("model").agg({
        "auroc": ["mean", "std"],
        "auprc": ["mean", "std"],
        "brier": ["mean", "std"],
        "ece": ["mean", "std"],
    }).reset_index()
    summary_primary.columns = [
        "model", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
        "brier_mean", "brier_std", "ece_mean", "ece_std"
    ]

    print("\n" + "=" * 80)
    print("PHASE 12 PRIMARY METRICS SUMMARY (Cumulative 5-Year Progression, N=341)")
    print("=" * 80)
    print(summary_primary.to_string(index=False))

    # Step 4: Participant-Clustered Bootstrap Analysis
    print("\n[4/5] Computing Clustered Bootstrap 95% Confidence Intervals...")
    comp_pairs = [
        ("M4_Adaptive_QCDF_Hybrid", "M5_ClassicalRelational_QCDF", "M4 vs M5 (Quantum vs Matched Classical Relational)"),
        ("M4_Adaptive_QCDF_Hybrid", "M6_ZeroQuantum_Suppressed", "M4 vs M6 (Quantum vs Quantum Suppressed)"),
        ("M4_Adaptive_QCDF_Hybrid", "M0_CTCLM_Only", "M4 vs M0 (Quantum Hybrid vs Pure CTCLM)"),
    ]

    bootstrap_results = {}
    for mod_a, mod_b, label in comp_pairs:
        diff_auc, lo_auc, hi_auc = compute_clustered_bootstrap_ci(
            y_all_5yr, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            roc_auc_score, n_boot=1000, seed=42
        )
        diff_pr, lo_pr, hi_pr = compute_clustered_bootstrap_ci(
            y_all_5yr, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            average_precision_score, n_boot=1000, seed=42
        )
        diff_brier, lo_brier, hi_brier = compute_clustered_bootstrap_ci(
            y_all_5yr, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            brier_score_loss, n_boot=1000, seed=42
        )
        bootstrap_results[label] = {
            "auroc_diff": diff_auc, "auroc_95ci": [lo_auc, hi_auc],
            "auprc_diff": diff_pr, "auprc_95ci": [lo_pr, hi_pr],
            "brier_diff": diff_brier, "brier_95ci": [lo_brier, hi_brier],
            "quantum_superior": bool(lo_auc > 0.0 and lo_pr > 0.0),
        }
        print(f"\n{label}:")
        print(f"  Delta AUROC: {diff_auc:+.4f} (95% CI: [{lo_auc:+.4f}, {hi_auc:+.4f}])")
        print(f"  Delta AUPRC: {diff_pr:+.4f} (95% CI: [{lo_pr:+.4f}, {hi_pr:+.4f}])")
        print(f"  Delta Brier: {diff_brier:+.4f} (95% CI: [{lo_brier:+.4f}, {hi_brier:+.4f}])")

    # Step 5: Save Deliverables & Verdict
    print("\n[5/5] Synthesizing Final Verdict and Saving Deliverables...")

    # Gate statistics
    all_q_gates = np.concatenate([v for v in quantum_gate_weights_all.values()])
    all_t_gates = np.concatenate([v for v in temporal_gate_weights_all.values()])
    mean_g_q = float(np.mean(all_q_gates))
    std_g_q = float(np.std(all_q_gates))
    mean_g_t = float(np.mean(all_t_gates))
    std_g_t = float(np.std(all_t_gates))

    # Determine scientific verdict
    m4_vs_m5 = bootstrap_results["M4 vs M5 (Quantum vs Matched Classical Relational)"]
    m4_vs_m6 = bootstrap_results["M4 vs M6 (Quantum vs Quantum Suppressed)"]

    quantum_advantage_found = bool(m4_vs_m5["quantum_superior"] and m4_vs_m6["quantum_superior"])
    if quantum_advantage_found:
        verdict_status = "QUANTUM RELATIONAL ENCODING DEMONSTRATES STATISTICAL ADVANTAGE"
    elif m4_vs_m5["auroc_diff"] > 0.0 and m4_vs_m6["auroc_diff"] > 0.0:
        verdict_status = "QUANTUM RELATIONAL ENCODING PARITY OR MODEST GAIN (NOT STATISTICALLY SIGNIFICANT OVER MATCHED CLASSICAL)"
    else:
        verdict_status = "QUANTUM RELATIONAL ENCODING PROVIDES NO ADVANTAGE OVER MATCHED CLASSICAL CONTROL"

    # Save CSV
    csv_local = os.path.join(LOCAL_OUT_DIR, "PHASE12_MODEL_METRICS.csv")
    csv_audit = os.path.join(AUDIT_OUT_DIR, "PHASE12_MODEL_METRICS.csv")
    summary_primary.to_csv(csv_local, index=False)
    summary_primary.to_csv(csv_audit, index=False)

    # Save Verdict JSON
    verdict_data = {
        "phase": "PHASE_12",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort_size": N,
        "n_seeds": len(seeds),
        "cv_folds": 5,
        "verdict_status": verdict_status,
        "quantum_advantage_detected": quantum_advantage_found,
        "adaptive_gating": {
            "quantum_gate_mean": mean_g_q,
            "quantum_gate_std": std_g_q,
            "temporal_gate_mean": mean_g_t,
            "temporal_gate_std": std_g_t,
            "can_suppress_quantum": True,
        },
        "model_rankings_primary_auroc": summary_primary.sort_values(by="auroc_mean", ascending=False)[["model", "auroc_mean", "auprc_mean", "brier_mean"]].to_dict(orient="records"),
        "hypothesis_tests_clustered_bootstrap": bootstrap_results,
        "qxai_summary": {
            "all_perturbation_checks_invariant": all(r["perturbation_invariant"] for r in qxai_diagnostics_records),
            "max_observed_perturbation_leakage": max(r["max_pert_diff"] for r in qxai_diagnostics_records),
            "mean_abs_quantum_attribution": float(np.mean([r["mean_abs_delta_q"] for r in qxai_diagnostics_records])),
            "mean_abs_temporal_attribution": float(np.mean([r["mean_abs_delta_t"] for r in qxai_diagnostics_records])),
        },
    }

    json_local = os.path.join(LOCAL_OUT_DIR, "PHASE12_VERDICT.json")
    json_audit = os.path.join(AUDIT_OUT_DIR, "PHASE12_VERDICT.json")
    with open(json_local, "w") as f:
        json.dump(verdict_data, f, indent=2)
    with open(json_audit, "w") as f:
        json.dump(verdict_data, f, indent=2)

    print(f"\nDeliverables written:")
    print(f"  [1] {csv_local} & {csv_audit}")
    print(f"  [2] {json_local} & {json_audit}")
    print("\n" + "=" * 80)
    print(f"OFFICIAL SCIENTIFIC VERDICT: {verdict_status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
