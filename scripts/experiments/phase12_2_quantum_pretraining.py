"""
NĀḌĪ-Q Phase 12.2 Master Experiment:
Quantum Relational Self-Supervised Pretraining & Fusion Recovery Benchmark.

Primary Research Hypothesis:
Does pretraining QVBE-v4 on evidence-grown biological relationship structure create
stable and measurable complementary information beyond an equivalently pretrained
classical relational encoder?

Evaluation Protocol:
- Authoritative PPMI Longitudinal Cohort (N=341)
- 3 Random Seeds (42, 123, 456) x 5-Fold Stratified Cross-Validation
- Pretraining fitted strictly within training participants in each fold (zero test leakage)
- Primary Evaluation: Future-Only Month-12-to-Month-60 motor progression (target_future_prog_mcid5)
- Secondary Evaluation: Cumulative baseline-to-Month-60 progression (target_progression_5yr_mcid5)
- 8 Controlled Models:
    * M0: CTCLM-only baseline
    * M1: CTCLM + QVBE random init + simple concatenation (Q0 + Simple Concat)
    * M2: CTCLM + QVBE pretrained & frozen + simple concatenation (Q1 + Simple Concat)
    * M3: CTCLM + QVBE pretrained & partially fine-tuned + simple concatenation (Q2 + Simple Concat)
    * M4: CTCLM + QVBE pretrained & partially fine-tuned + existing QCDF (Q2 + Adaptive QCDF)
    * M5: CTCLM + QVBE pretrained & partially fine-tuned + residual adaptive fusion (Q2 + Residual QCDF)
    * M6: CTCLM + matched classical relational pretraining + simple concatenation (C2 + Simple Concat)
    * M7: CTCLM + matched classical relational pretraining + residual adaptive fusion (C2 + Residual QCDF)
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

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.qvbe_v4.pretraining import (
    RelationalPretrainingConfig,
    QVBEv4PretrainingModel,
    ClassicalRelationalPretrainingModel,
)
from nadiq.models.integrated.config import NadiQIntegratedConfig
from nadiq.models.integrated.model import ClassicalRelationalBranch, NadiQIntegratedModel
from nadiq.models.integrated.fusion import AdaptiveQCDF, ResidualAdaptiveQCDF
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel, CTCLMConfig

# Output Paths
LOCAL_OUT_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase12_2"
AUDIT_OUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_12_2"
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
# Model Architectures for Benchmark (Simple Concat & Residual Adaptive)
# ==============================================================================
class DownstreamSimpleConcat(nn.Module):
    """Simple Concatenation Model supporting arbitrary relational encoder (quantum or classical)."""
    def __init__(self, relational_encoder: nn.Module, is_quantum: bool = True, freeze_strategy: str = "full"):
        super().__init__()
        self.relational_encoder = relational_encoder
        self.is_quantum = is_quantum
        self.ctclm = ContinualTemporalClinicalLatentModel(CTCLMConfig(input_dim=1, latent_dim=16, time_embedding_dim=4))

        rel_in_dim = 16 if is_quantum else 12
        self.proj_rel = nn.Linear(rel_in_dim, 16)
        self.proj_ctclm = nn.Linear(16, 16)
        self.proj_node = nn.Linear(6, 16)

        self.head = nn.Sequential(
            nn.Linear(16 * 3, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

        # Apply freeze strategy
        self._apply_freeze(freeze_strategy)

    def _apply_freeze(self, strategy: str):
        if self.is_quantum:
            if strategy == "frozen":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = False
            elif strategy == "partial":
                for p in self.relational_encoder.circuit.parameters():
                    p.requires_grad = False
                self.relational_encoder.circuit.var_rotations.requires_grad = True
                for a in self.relational_encoder.adapters.values():
                    for p in a.parameters():
                        p.requires_grad = True
            elif strategy == "full":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = True
        else:
            if strategy == "frozen":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = False
            elif strategy in ["partial", "full"]:
                for p in self.relational_encoder.parameters():
                    p.requires_grad = True

    def forward(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        graph_tensors: Dict[str, torch.Tensor],
        ctclm_seq: torch.Tensor,
        ctclm_times: torch.Tensor,
        ctclm_mask: torch.Tensor,
        node_masks: torch.Tensor,
        elapsed_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = ctclm_seq.shape[0]

        # 1. Relational encoding
        if self.is_quantum:
            q_out = self.relational_encoder(
                node_inputs=node_inputs,
                graph_edges=graph_edges,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
            )
            rel_emb = q_out.motif_embedding
        else:
            node_states_list = []
            for idx, name in enumerate(["molecular", "structural", "functional"]):
                x_raw, dx_raw = node_inputs[name]
                m = node_masks[:, idx : idx + 1]
                node_states_list.append(x_raw[:, :1] * m)
                node_states_list.append(dx_raw[:, :1] * m)
            node_flat = torch.cat(node_states_list, dim=-1)
            node_states_3d = node_flat.reshape(batch_size, 3, 2)
            rel_emb = self.relational_encoder(
                node_states_3d=node_states_3d,
                edge_weights=graph_tensors["edge_weights"],
                edge_lags=graph_tensors["edge_lags"],
                edge_se=graph_tensors["edge_uncertainties"],
                direction_signs=graph_tensors["direction_signs"],
            )

        # 2. CTCLM encoding
        ctclm_out = self.ctclm(ctclm_seq, ctclm_times, ctclm_mask)

        # 3. Node features
        node_states_list = []
        for idx, name in enumerate(["molecular", "structural", "functional"]):
            x_raw, dx_raw = node_inputs[name]
            m = node_masks[:, idx : idx + 1]
            node_states_list.append(x_raw[:, :1] * m)
            node_states_list.append(dx_raw[:, :1] * m)
        node_flat = torch.cat(node_states_list, dim=-1)

        f_rel = F.gelu(self.proj_rel(rel_emb))
        f_t = F.gelu(self.proj_ctclm(ctclm_out.h_final))
        f_n = F.gelu(self.proj_node(node_flat))

        concat_vec = torch.cat([f_rel, f_t, f_n], dim=-1)
        logits = self.head(concat_vec).squeeze(-1)
        return logits, rel_emb


class DownstreamResidualFusion(nn.Module):
    """Residual Adaptive Fusion Model supporting arbitrary relational encoder."""
    def __init__(self, relational_encoder: nn.Module, is_quantum: bool = True, freeze_strategy: str = "partial"):
        super().__init__()
        self.relational_encoder = relational_encoder
        self.is_quantum = is_quantum
        self.ctclm = ContinualTemporalClinicalLatentModel(CTCLMConfig(input_dim=1, latent_dim=16, time_embedding_dim=4))

        cfg = NadiQIntegratedConfig(
            enable_quantum_branch=is_quantum,
            enable_classical_relational=not is_quantum,
            fusion_type="residual_adaptive",
        )
        self.fusion = ResidualAdaptiveQCDF(cfg)

        # Apply freeze
        self._apply_freeze(freeze_strategy)

    def _apply_freeze(self, strategy: str):
        if self.is_quantum:
            if strategy == "frozen":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = False
            elif strategy == "partial":
                for p in self.relational_encoder.circuit.parameters():
                    p.requires_grad = False
                self.relational_encoder.circuit.var_rotations.requires_grad = True
                for a in self.relational_encoder.adapters.values():
                    for p in a.parameters():
                        p.requires_grad = True
            elif strategy == "full":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = True
        else:
            if strategy == "frozen":
                for p in self.relational_encoder.parameters():
                    p.requires_grad = False
            elif strategy in ["partial", "full"]:
                for p in self.relational_encoder.parameters():
                    p.requires_grad = True

    def forward(
        self,
        node_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        graph_edges: List[Dict[str, Any]],
        graph_tensors: Dict[str, torch.Tensor],
        ctclm_seq: torch.Tensor,
        ctclm_times: torch.Tensor,
        ctclm_mask: torch.Tensor,
        node_masks: torch.Tensor,
        elapsed_times: Optional[torch.Tensor] = None,
        force_quantum_suppression: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = ctclm_seq.shape[0]

        # Node flat
        node_states_list = []
        for idx, name in enumerate(["molecular", "structural", "functional"]):
            x_raw, dx_raw = node_inputs[name]
            m = node_masks[:, idx : idx + 1]
            node_states_list.append(x_raw[:, :1] * m)
            node_states_list.append(dx_raw[:, :1] * m)
        node_flat = torch.cat(node_states_list, dim=-1)
        node_states_3d = node_flat.reshape(batch_size, 3, 2)

        # Relational embedding
        qvbe_emb = None
        cr_emb = None
        if self.is_quantum:
            q_out = self.relational_encoder(
                node_inputs=node_inputs,
                graph_edges=graph_edges,
                node_masks=node_masks,
                elapsed_times=elapsed_times,
            )
            qvbe_emb = q_out.motif_embedding
        else:
            cr_emb = self.relational_encoder(
                node_states_3d=node_states_3d,
                edge_weights=graph_tensors["edge_weights"],
                edge_lags=graph_tensors["edge_lags"],
                edge_se=graph_tensors["edge_uncertainties"],
                direction_signs=graph_tensors["direction_signs"],
            )

        ctclm_out = self.ctclm(ctclm_seq, ctclm_times, ctclm_mask)
        ctclm_emb = ctclm_out.h_final

        logits, probs, z_fused, alpha, _, _ = self.fusion(
            node_states_flat=node_flat,
            qvbe_emb=qvbe_emb,
            ctclm_emb=ctclm_emb,
            classical_rel_emb=cr_emb,
            node_masks=node_masks,
            edge_uncertainties=graph_tensors["edge_uncertainties"],
            force_quantum_suppression=force_quantum_suppression,
        )
        active_emb = qvbe_emb if self.is_quantum else cr_emb
        return logits, probs, z_fused, alpha


# ==============================================================================
# Master Experiment Runner
# ==============================================================================
def main():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 12.2: QUANTUM RELATIONAL PRETRAINING & FUSION RECOVERY")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Load Authoritative Multimodal PPMI Cohort
    print("\n[1/6] Loading authoritative longitudinal cohort and cache...")
    df_cohort = pd.read_parquet(COHORT_PARQUET)
    df_cache = pd.read_parquet(CACHE_PARQUET)

    pats = df_cohort["PATNO"].tolist()
    y_all_5yr = df_cohort["target_progression_5yr_mcid5"].values.astype(int)

    # Future-only Month 12 to Month 60 progression (PRIMARY TARGET)
    df_cohort["future_delta"] = df_cohort["delta_NP3TOT_5yr"] - df_cohort["early_trajectory_delta_1yr"]
    y_all_future = (df_cohort["future_delta"].values >= 5.0).astype(int)

    N = len(pats)
    print(f"Cohort size: N={N} participants")
    print(f"PRIMARY Clinical Target: Future-Only Month 12-to-60: {np.sum(y_all_future)} events ({np.mean(y_all_future)*100:.1f}%)")
    print(f"SECONDARY Clinical Target: Cumulative 5-Year Progression: {np.sum(y_all_5yr)} events ({np.mean(y_all_5yr)*100:.1f}%)")

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
    reasoner = BiologicalKnowledgeGraphReasoner()

    # Benchmark models to evaluate
    benchmark_models = [
        "M0_CTCLM_Only",
        "M1_QVBE_RandomInit_SimpleConcat",
        "M2_QVBE_Pretrained_Frozen_SimpleConcat",
        "M3_QVBE_Pretrained_PartialFT_SimpleConcat",
        "M4_QVBE_Pretrained_PartialFT_AdaptiveQCDF",
        "M5_QVBE_Pretrained_PartialFT_ResidualQCDF",
        "M6_Classical_Pretrained_SimpleConcat",
        "M7_Classical_Pretrained_ResidualQCDF",
    ]

    all_seed_results_primary = []
    all_seed_results_secondary = []
    all_seed_oof_probs_primary: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in benchmark_models}
    all_seed_oof_probs_secondary: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in benchmark_models}

    pretrain_diagnostics = []
    representation_diagnostics = {
        "m1_rank": [], "m2_rank": [], "m3_rank": [], "m5_rank": [], "m6_rank": [],
        "m1_var": [], "m2_var": [], "m3_var": [], "m5_var": [], "m6_var": [],
        "m1_grad": [], "m3_grad": [], "m4_grad": [], "m5_grad": [],
    }

    best_pretrain_loss_q = float("inf")
    best_pretrain_loss_c = float("inf")

    print("\n[2/6] Executing Controlled 8-Model Benchmark across 3 Seeds & 5 Folds...")
    t_bench = time.time()

    for seed in seeds:
        print(f"\n==================== Random Seed {seed} ====================")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_p = {m: np.zeros(N, dtype=float) for m in benchmark_models}
        oof_s = {m: np.zeros(N, dtype=float) for m in benchmark_models}

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all_future)):
            # 1. Fit Preprocessing Scalers inside training fold strictly
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
                return {k: (v[0][indices], v[1][indices]) for k, v in node_inputs_all.items()}

            train_node_inputs = slice_node_inputs(train_idx)
            test_node_inputs = slice_node_inputs(test_idx)

            ctclm_seq_all = torch.tensor(np.stack([
                x_motor_bl.reshape(N, 1),
                x_motor_v04.reshape(N, 1),
            ], axis=1), dtype=torch.float32)

            ctclm_times_all = torch.zeros(N, 2, dtype=torch.float32)
            ctclm_times_all[:, 1] = torch.tensor(dt_years_raw, dtype=torch.float32)

            ctclm_mask_all = torch.ones(N, 2, dtype=torch.bool)
            node_masks_all = torch.ones(N, 3, dtype=torch.float32)
            elapsed_t_all = torch.tensor(dt_years_raw, dtype=torch.float32).unsqueeze(-1)

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

            # Primary: Future-only progression (y_all_future)
            y_train_primary = torch.tensor(y_all_future[train_idx], dtype=torch.float32)
            # Secondary: Cumulative 5-year progression (y_all_5yr)
            y_train_secondary = torch.tensor(y_all_5yr[train_idx], dtype=torch.float32)

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
            compiler_ref = QVBEv4Encoder(QVBEv4Config(node_input_dims={"molecular": 1, "structural": 1, "functional": 1})).compiler
            train_gt = compiler_ref.compile_graph_motif(graph_edges_fmt, train_nmasks, train_elapsed, batch_size=len(train_idx))
            test_gt = compiler_ref.compile_graph_motif(graph_edges_fmt, test_nmasks, test_elapsed, batch_size=len(test_idx))

            # =================================================================
            # STAGE 1: SELF-SUPERVISED RELATIONAL PRETRAINING (Train Fold Only)
            # =================================================================
            torch.manual_seed(seed + fold)
            pretrain_cfg = RelationalPretrainingConfig(learning_rate=0.015, n_epochs=15)
            qvbe_cfg = QVBEv4Config(
                n_nodes=3,
                qubits_per_node=2,
                node_names=["molecular", "structural", "functional"],
                node_input_dims={"molecular": 1, "structural": 1, "functional": 1},
                motif_embed_dim=16,
            )

            # A. Quantum Pretraining
            q_pretrainer = QVBEv4PretrainingModel(qvbe_cfg, pretrain_cfg)
            opt_q_pre = optim.AdamW(q_pretrainer.parameters(), lr=0.015, weight_decay=1e-4)
            q_pretrainer.train()
            final_loss_q = 0.0
            for ep in range(15):
                opt_q_pre.zero_grad()
                preds_q, loss_dict_q, _ = q_pretrainer.forward_pretrain(
                    node_inputs=train_node_inputs,
                    graph_edges=graph_edges_fmt,
                    node_masks=train_nmasks,
                    elapsed_times=train_elapsed,
                )
                loss_q = loss_dict_q["total_loss"]
                loss_q.backward()
                opt_q_pre.step()
                final_loss_q = float(loss_q.item())
            q_pretrainer.eval()

            # Save best quantum checkpoint
            if final_loss_q < best_pretrain_loss_q:
                best_pretrain_loss_q = final_loss_q
                torch.save(q_pretrainer.encoder.state_dict(), os.path.join(LOCAL_OUT_DIR, "best_qvbe_pretrained.pt"))
                torch.save(q_pretrainer.encoder.state_dict(), os.path.join(AUDIT_OUT_DIR, "best_qvbe_pretrained.pt"))

            # B. Matched Classical Pretraining
            c_pretrainer = ClassicalRelationalPretrainingModel(embed_dim=16, pretrain_config=pretrain_cfg)
            opt_c_pre = optim.AdamW(c_pretrainer.parameters(), lr=0.015, weight_decay=1e-4)
            c_pretrainer.train()
            final_loss_c = 0.0
            for ep in range(15):
                opt_c_pre.zero_grad()
                preds_c, loss_dict_c, _ = c_pretrainer.forward_pretrain(
                    node_inputs=train_node_inputs,
                    graph_edges=graph_edges_fmt,
                    graph_tensors=train_gt,
                    node_masks=train_nmasks,
                )
                loss_c = loss_dict_c["total_loss"]
                loss_c.backward()
                opt_c_pre.step()
                final_loss_c = float(loss_c.item())
            c_pretrainer.eval()

            # Save best classical checkpoint
            if final_loss_c < best_pretrain_loss_c:
                best_pretrain_loss_c = final_loss_c
                torch.save(c_pretrainer.branch.state_dict(), os.path.join(LOCAL_OUT_DIR, "best_classical_pretrained.pt"))
                torch.save(c_pretrainer.branch.state_dict(), os.path.join(AUDIT_OUT_DIR, "best_classical_pretrained.pt"))

            pretrain_diagnostics.append({
                "seed": seed, "fold": fold,
                "final_pretrain_loss_quantum": final_loss_q,
                "final_pretrain_loss_classical": final_loss_c,
            })

            # =================================================================
            # STAGE 2: DOWNSTREAM SUPERVISED INTEGRATION (Primary Target)
            # =================================================================

            # M0: CTCLM Only Baseline
            torch.manual_seed(seed + fold)
            cfg_m0 = NadiQIntegratedConfig(enable_quantum_branch=False, enable_classical_relational=False)
            model_m0 = NadiQIntegratedModel(cfg_m0)
            opt_m0 = optim.AdamW(model_m0.parameters(), lr=0.015, weight_decay=1e-4)
            model_m0.train()
            for _ in range(20):
                opt_m0.zero_grad()
                out_m0 = model_m0(train_node_inputs, graph_edges_fmt, train_seq, train_times, train_mask, train_nmasks, train_elapsed, mode="ctclm_only")
                loss = bce(out_m0.logits, y_train_primary)
                loss.backward()
                opt_m0.step()
            model_m0.eval()
            with torch.no_grad():
                oof_p["M0_CTCLM_Only"][test_idx] = model_m0(test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed, mode="ctclm_only").probabilities.cpu().numpy()

            # M1: CTCLM + QVBE Random Init + Simple Concat (Q0 + Concat)
            torch.manual_seed(seed + fold)
            encoder_q_rand = QVBEv4Encoder(qvbe_cfg)
            model_m1 = DownstreamSimpleConcat(encoder_q_rand, is_quantum=True, freeze_strategy="full")
            opt_m1 = optim.AdamW(model_m1.parameters(), lr=0.015, weight_decay=1e-4)
            model_m1.train()
            m1_grad = 0.0
            for ep in range(20):
                opt_m1.zero_grad()
                lg_m1, _ = model_m1(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m1, y_train_primary)
                loss.backward()
                if ep == 19:
                    m1_grad = float(torch.norm(model_m1.relational_encoder.circuit.var_rotations.grad).item())
                opt_m1.step()
            model_m1.eval()
            with torch.no_grad():
                lg_te, emb_te = model_m1(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M1_QVBE_RandomInit_SimpleConcat"][test_idx] = torch.sigmoid(lg_te).cpu().numpy()
                var_m1 = float(emb_te.var(dim=0).mean().item())
                s = torch.linalg.svdvals(emb_te)
                p_s = s / s.sum()
                rank_m1 = float(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-12))).item())
                representation_diagnostics["m1_var"].append(var_m1)
                representation_diagnostics["m1_rank"].append(rank_m1)
                representation_diagnostics["m1_grad"].append(m1_grad)

            # M2: CTCLM + QVBE Pretrained & Frozen + Simple Concat (Q1 + Concat)
            torch.manual_seed(seed + fold)
            encoder_q_frozen = copy.deepcopy(q_pretrainer.encoder)
            model_m2 = DownstreamSimpleConcat(encoder_q_frozen, is_quantum=True, freeze_strategy="frozen")
            opt_m2 = optim.AdamW(model_m2.parameters(), lr=0.015, weight_decay=1e-4)
            model_m2.train()
            for _ in range(20):
                opt_m2.zero_grad()
                lg_m2, _ = model_m2(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m2, y_train_primary)
                loss.backward()
                opt_m2.step()
            model_m2.eval()
            with torch.no_grad():
                lg_te, emb_te = model_m2(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M2_QVBE_Pretrained_Frozen_SimpleConcat"][test_idx] = torch.sigmoid(lg_te).cpu().numpy()
                var_m2 = float(emb_te.var(dim=0).mean().item())
                s = torch.linalg.svdvals(emb_te)
                p_s = s / s.sum()
                rank_m2 = float(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-12))).item())
                representation_diagnostics["m2_var"].append(var_m2)
                representation_diagnostics["m2_rank"].append(rank_m2)

            # M3: CTCLM + QVBE Pretrained & Partially Fine-Tuned + Simple Concat (Q2 + Concat)
            torch.manual_seed(seed + fold)
            encoder_q_partial = copy.deepcopy(q_pretrainer.encoder)
            model_m3 = DownstreamSimpleConcat(encoder_q_partial, is_quantum=True, freeze_strategy="partial")
            opt_m3 = optim.AdamW(filter(lambda p: p.requires_grad, model_m3.parameters()), lr=0.015, weight_decay=1e-4)
            model_m3.train()
            m3_grad = 0.0
            for ep in range(20):
                opt_m3.zero_grad()
                lg_m3, _ = model_m3(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m3, y_train_primary)
                loss.backward()
                if ep == 19:
                    m3_grad = float(torch.norm(model_m3.relational_encoder.circuit.var_rotations.grad).item())
                opt_m3.step()
            model_m3.eval()
            with torch.no_grad():
                lg_te, emb_te = model_m3(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M3_QVBE_Pretrained_PartialFT_SimpleConcat"][test_idx] = torch.sigmoid(lg_te).cpu().numpy()
                var_m3 = float(emb_te.var(dim=0).mean().item())
                s = torch.linalg.svdvals(emb_te)
                p_s = s / s.sum()
                rank_m3 = float(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-12))).item())
                representation_diagnostics["m3_var"].append(var_m3)
                representation_diagnostics["m3_rank"].append(rank_m3)
                representation_diagnostics["m3_grad"].append(m3_grad)

            # M4: CTCLM + QVBE Pretrained & Partially Fine-Tuned + Existing QCDF (Q2 + Adaptive QCDF)
            torch.manual_seed(seed + fold)
            cfg_m4 = NadiQIntegratedConfig(enable_quantum_branch=True, enable_ctclm_branch=True, enable_classical_relational=False)
            model_m4 = NadiQIntegratedModel(cfg_m4)
            model_m4.qvbe.load_state_dict(q_pretrainer.encoder.state_dict())
            # Freeze coupling angles
            for p in model_m4.qvbe.circuit.parameters():
                p.requires_grad = False
            model_m4.qvbe.circuit.var_rotations.requires_grad = True
            opt_m4 = optim.AdamW(filter(lambda p: p.requires_grad, model_m4.parameters()), lr=0.015, weight_decay=1e-4)
            model_m4.train()
            m4_grad = 0.0
            for ep in range(20):
                opt_m4.zero_grad()
                out_m4 = model_m4(train_node_inputs, graph_edges_fmt, train_seq, train_times, train_mask, train_nmasks, train_elapsed, mode="full")
                loss = bce(out_m4.logits, y_train_primary)
                loss.backward()
                if ep == 19:
                    m4_grad = float(torch.norm(model_m4.qvbe.circuit.var_rotations.grad).item())
                opt_m4.step()
            model_m4.eval()
            with torch.no_grad():
                out_m4_te = model_m4(test_node_inputs, graph_edges_fmt, test_seq, test_times, test_mask, test_nmasks, test_elapsed, mode="full")
                oof_p["M4_QVBE_Pretrained_PartialFT_AdaptiveQCDF"][test_idx] = out_m4_te.probabilities.cpu().numpy()
                representation_diagnostics["m4_grad"].append(m4_grad)

            # M5: CTCLM + QVBE Pretrained & Partially Fine-Tuned + Residual Adaptive QCDF (Q2 + Residual QCDF)
            torch.manual_seed(seed + fold)
            encoder_q_res = copy.deepcopy(q_pretrainer.encoder)
            model_m5 = DownstreamResidualFusion(encoder_q_res, is_quantum=True, freeze_strategy="partial")
            opt_m5 = optim.AdamW(filter(lambda p: p.requires_grad, model_m5.parameters()), lr=0.015, weight_decay=1e-4)
            model_m5.train()
            m5_grad = 0.0
            for ep in range(20):
                opt_m5.zero_grad()
                lg_m5, _, _, _ = model_m5(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m5, y_train_primary)
                loss.backward()
                if ep == 19:
                    m5_grad = float(torch.norm(model_m5.relational_encoder.circuit.var_rotations.grad).item())
                opt_m5.step()
            model_m5.eval()
            with torch.no_grad():
                lg_te, probs_te, z_te, alpha_te = model_m5(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M5_QVBE_Pretrained_PartialFT_ResidualQCDF"][test_idx] = probs_te.cpu().numpy()
                var_m5 = float(z_te.var(dim=0).mean().item())
                s = torch.linalg.svdvals(z_te)
                p_s = s / s.sum()
                rank_m5 = float(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-12))).item())
                representation_diagnostics["m5_var"].append(var_m5)
                representation_diagnostics["m5_rank"].append(rank_m5)
                representation_diagnostics["m5_grad"].append(m5_grad)

            # M6: CTCLM + Matched Classical Relational Pretraining + Simple Concat (C2 + Simple Concat)
            torch.manual_seed(seed + fold)
            branch_c_pre = copy.deepcopy(c_pretrainer.branch)
            model_m6 = DownstreamSimpleConcat(branch_c_pre, is_quantum=False, freeze_strategy="partial")
            opt_m6 = optim.AdamW(model_m6.parameters(), lr=0.015, weight_decay=1e-4)
            model_m6.train()
            for _ in range(20):
                opt_m6.zero_grad()
                lg_m6, _ = model_m6(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m6, y_train_primary)
                loss.backward()
                opt_m6.step()
            model_m6.eval()
            with torch.no_grad():
                lg_te, emb_te = model_m6(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M6_Classical_Pretrained_SimpleConcat"][test_idx] = torch.sigmoid(lg_te).cpu().numpy()
                var_m6 = float(emb_te.var(dim=0).mean().item())
                s = torch.linalg.svdvals(emb_te)
                p_s = s / s.sum()
                rank_m6 = float(torch.exp(-torch.sum(p_s * torch.log(p_s + 1e-12))).item())
                representation_diagnostics["m6_var"].append(var_m6)
                representation_diagnostics["m6_rank"].append(rank_m6)

            # M7: CTCLM + Matched Classical Relational Pretraining + Residual Adaptive Fusion (C2 + Residual QCDF)
            torch.manual_seed(seed + fold)
            branch_c_res = copy.deepcopy(c_pretrainer.branch)
            model_m7 = DownstreamResidualFusion(branch_c_res, is_quantum=False, freeze_strategy="partial")
            opt_m7 = optim.AdamW(model_m7.parameters(), lr=0.015, weight_decay=1e-4)
            model_m7.train()
            for _ in range(20):
                opt_m7.zero_grad()
                lg_m7, _, _, _ = model_m7(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)
                loss = bce(lg_m7, y_train_primary)
                loss.backward()
                opt_m7.step()
            model_m7.eval()
            with torch.no_grad():
                lg_te, probs_te, _, _ = model_m7(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)
                oof_p["M7_Classical_Pretrained_ResidualQCDF"][test_idx] = probs_te.cpu().numpy()

            # Secondary Target Optimization (M3, M5, M6, M7 on 5-Year Cumulative Progression)
            # Re-train models M3, M5, M6 on secondary target y_train_secondary
            m3_sec = DownstreamSimpleConcat(copy.deepcopy(q_pretrainer.encoder), is_quantum=True, freeze_strategy="partial")
            opt_m3_sec = optim.AdamW(filter(lambda p: p.requires_grad, m3_sec.parameters()), lr=0.015, weight_decay=1e-4)
            m3_sec.train()
            for _ in range(20):
                opt_m3_sec.zero_grad()
                loss = bce(m3_sec(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)[0], y_train_secondary)
                loss.backward()
                opt_m3_sec.step()
            m3_sec.eval()

            m5_sec = DownstreamResidualFusion(copy.deepcopy(q_pretrainer.encoder), is_quantum=True, freeze_strategy="partial")
            opt_m5_sec = optim.AdamW(filter(lambda p: p.requires_grad, m5_sec.parameters()), lr=0.015, weight_decay=1e-4)
            m5_sec.train()
            for _ in range(20):
                opt_m5_sec.zero_grad()
                loss = bce(m5_sec(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)[0], y_train_secondary)
                loss.backward()
                opt_m5_sec.step()
            m5_sec.eval()

            m6_sec = DownstreamSimpleConcat(copy.deepcopy(c_pretrainer.branch), is_quantum=False, freeze_strategy="partial")
            opt_m6_sec = optim.AdamW(m6_sec.parameters(), lr=0.015, weight_decay=1e-4)
            m6_sec.train()
            for _ in range(20):
                opt_m6_sec.zero_grad()
                loss = bce(m6_sec(train_node_inputs, graph_edges_fmt, train_gt, train_seq, train_times, train_mask, train_nmasks, train_elapsed)[0], y_train_secondary)
                loss.backward()
                opt_m6_sec.step()
            m6_sec.eval()

            with torch.no_grad():
                oof_s["M3_QVBE_Pretrained_PartialFT_SimpleConcat"][test_idx] = torch.sigmoid(m3_sec(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)[0]).cpu().numpy()
                oof_s["M5_QVBE_Pretrained_PartialFT_ResidualQCDF"][test_idx] = m5_sec(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)[1].cpu().numpy()
                oof_s["M6_Classical_Pretrained_SimpleConcat"][test_idx] = torch.sigmoid(m6_sec(test_node_inputs, graph_edges_fmt, test_gt, test_seq, test_times, test_mask, test_nmasks, test_elapsed)[0]).cpu().numpy()

        for m in benchmark_models:
            all_seed_oof_probs_primary[m][seed] = oof_p[m]
            all_seed_oof_probs_secondary[m][seed] = oof_s[m]

            auc_p = roc_auc_score(y_all_future, oof_p[m])
            pr_p = average_precision_score(y_all_future, oof_p[m])
            brier_p = brier_score_loss(y_all_future, oof_p[m])
            ece_p = compute_ece(y_all_future, oof_p[m])
            all_seed_results_primary.append({
                "seed": seed, "model": m, "target": "future_only_m12_m60",
                "auroc": auc_p, "auprc": pr_p, "brier": brier_p, "ece": ece_p
            })

            if np.any(oof_s[m] > 0):
                auc_s = roc_auc_score(y_all_5yr, oof_s[m])
                pr_s = average_precision_score(y_all_5yr, oof_s[m])
                brier_s = brier_score_loss(y_all_5yr, oof_s[m])
                ece_s = compute_ece(y_all_5yr, oof_s[m])
                all_seed_results_secondary.append({
                    "seed": seed, "model": m, "target": "cumulative_5yr",
                    "auroc": auc_s, "auprc": pr_s, "brier": brier_s, "ece": ece_s
                })

        print(f"Seed {seed} complete. M3 (Pretrained Q Concat) AUROC={roc_auc_score(y_all_future, oof_p['M3_QVBE_Pretrained_PartialFT_SimpleConcat']):.4f}, M6 (Pretrained C Concat) AUROC={roc_auc_score(y_all_future, oof_p['M6_Classical_Pretrained_SimpleConcat']):.4f}, M5 (Pretrained Q Residual) AUROC={roc_auc_score(y_all_future, oof_p['M5_QVBE_Pretrained_PartialFT_ResidualQCDF']):.4f}")

    print(f"\n[3/6] Benchmark execution completed in {time.time() - t_bench:.1f}s.")

    # Step 3: Aggregate Primary Metrics
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
    print("PHASE 12.2 PRIMARY BENCHMARK RESULTS (Future-Only Month 12-to-60 Motor Progression, N=341)")
    print("=" * 80)
    print(summary_primary.sort_values(by="auroc_mean", ascending=False).to_string(index=False))

    # Secondary Metrics
    df_res_s = pd.DataFrame(all_seed_results_secondary)
    summary_secondary = df_res_s.groupby("model").agg({
        "auroc": ["mean", "std"],
        "auprc": ["mean", "std"],
        "brier": ["mean", "std"],
        "ece": ["mean", "std"],
    }).reset_index()
    summary_secondary.columns = [
        "model", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
        "brier_mean", "brier_std", "ece_mean", "ece_std"
    ]

    print("\n" + "=" * 80)
    print("PHASE 12.2 SECONDARY BENCHMARK RESULTS (Cumulative 5-Year Progression, N=341)")
    print("=" * 80)
    print(summary_secondary.sort_values(by="auroc_mean", ascending=False).to_string(index=False))

    # Step 4: Paired Clustered Bootstrap Hypothesis Testing (Primary Target)
    print("\n[4/6] Computing Clustered Bootstrap 95% Confidence Intervals...")
    comp_pairs = [
        ("M3_QVBE_Pretrained_PartialFT_SimpleConcat", "M6_Classical_Pretrained_SimpleConcat", "M3 vs M6 (Pretrained Quantum Concat vs Pretrained Classical Concat)"),
        ("M3_QVBE_Pretrained_PartialFT_SimpleConcat", "M1_QVBE_RandomInit_SimpleConcat", "M3 vs M1 (Pretrained Quantum Concat vs Random Init Quantum Concat)"),
        ("M5_QVBE_Pretrained_PartialFT_ResidualQCDF", "M4_QVBE_Pretrained_PartialFT_AdaptiveQCDF", "M5 vs M4 (Residual Adaptive Fusion vs Standard Adaptive QCDF)"),
        ("M5_QVBE_Pretrained_PartialFT_ResidualQCDF", "M7_Classical_Pretrained_ResidualQCDF", "M5 vs M7 (Quantum Residual Fusion vs Classical Residual Fusion)"),
        ("M3_QVBE_Pretrained_PartialFT_SimpleConcat", "M0_CTCLM_Only", "M3 vs M0 (Pretrained Quantum Hybrid vs Pure CTCLM Baseline)"),
    ]

    bootstrap_results = {}
    for mod_a, mod_b, label in comp_pairs:
        diff_auc, lo_auc, hi_auc = compute_clustered_bootstrap_ci(
            y_all_future, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            roc_auc_score, n_boot=1000, seed=42
        )
        diff_pr, lo_pr, hi_pr = compute_clustered_bootstrap_ci(
            y_all_future, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            average_precision_score, n_boot=1000, seed=42
        )
        diff_brier, lo_brier, hi_brier = compute_clustered_bootstrap_ci(
            y_all_future, all_seed_oof_probs_primary[mod_a], all_seed_oof_probs_primary[mod_b],
            brier_score_loss, n_boot=1000, seed=42
        )
        bootstrap_results[label] = {
            "auroc_diff": diff_auc, "auroc_95ci": [lo_auc, hi_auc],
            "auprc_diff": diff_pr, "auprc_95ci": [lo_pr, hi_pr],
            "brier_diff": diff_brier, "brier_95ci": [lo_brier, hi_brier],
        }
        print(f"\n{label}:")
        print(f"  Delta AUROC: {diff_auc:+.4f} (95% CI: [{lo_auc:+.4f}, {hi_auc:+.4f}])")
        print(f"  Delta AUPRC: {diff_pr:+.4f} (95% CI: [{lo_pr:+.4f}, {hi_pr:+.4f}])")
        print(f"  Delta Brier: {diff_brier:+.4f} (95% CI: [{lo_brier:+.4f}, {hi_brier:+.4f}])")

    # Step 5: Representation & Gradient Diagnostics
    print("\n[5/6] Synthesizing Representation & Gradient Flow Analysis...")
    diag_summary = {
        "mean_rank_m1_random": float(np.mean(representation_diagnostics["m1_rank"])),
        "mean_rank_m2_frozen": float(np.mean(representation_diagnostics["m2_rank"])),
        "mean_rank_m3_partial": float(np.mean(representation_diagnostics["m3_rank"])),
        "mean_rank_m5_residual": float(np.mean(representation_diagnostics["m5_rank"])),
        "mean_rank_m6_classical": float(np.mean(representation_diagnostics["m6_rank"])),
        "mean_grad_m1_random": float(np.mean(representation_diagnostics["m1_grad"])),
        "mean_grad_m3_partial": float(np.mean(representation_diagnostics["m3_grad"])),
        "mean_grad_m4_adaptive_qcdf": float(np.mean(representation_diagnostics["m4_grad"])),
        "mean_grad_m5_residual_qcdf": float(np.mean(representation_diagnostics["m5_grad"])),
    }
    print(f"  Effective Rank: Random={diag_summary['mean_rank_m1_random']:.2f}, Frozen={diag_summary['mean_rank_m2_frozen']:.2f}, Pretrained+PartialFT={diag_summary['mean_rank_m3_partial']:.2f}, ResidualFusion={diag_summary['mean_rank_m5_residual']:.2f}, Classical={diag_summary['mean_rank_m6_classical']:.2f}")
    print(f"  Circuit Gradient Norm: Random={diag_summary['mean_grad_m1_random']:.4e}, PartialFT={diag_summary['mean_grad_m3_partial']:.4e}, AdaptiveQCDF={diag_summary['mean_grad_m4_adaptive_qcdf']:.4e}, ResidualQCDF={diag_summary['mean_grad_m5_residual_qcdf']:.4e}")

    # Determine Scientific Verdict
    # Key hypothesis: Pretrained Quantum (M3) vs Pretrained Classical (M6)
    m3_vs_m6 = bootstrap_results["M3 vs M6 (Pretrained Quantum Concat vs Pretrained Classical Concat)"]
    m3_vs_m1 = bootstrap_results["M3 vs M1 (Pretrained Quantum Concat vs Random Init Quantum Concat)"]
    m5_vs_m4 = bootstrap_results["M5 vs M4 (Residual Adaptive Fusion vs Standard Adaptive QCDF)"]

    quantum_advantage = bool(m3_vs_m6["auroc_95ci"][0] > 0.0)
    pretraining_helps = bool(m3_vs_m1["auroc_95ci"][0] > 0.0 or m3_vs_m1["auroc_diff"] > 0.01)
    qcdf_was_bottleneck = bool(m5_vs_m4["auroc_diff"] > 0.01 and m5_vs_m4["auroc_95ci"][0] > 0.0)

    if quantum_advantage:
        verdict_status = "QUANTUM PRETRAINING ADDS COMPLEMENTARY VALUE"
    elif pretraining_helps and not quantum_advantage:
        verdict_status = "PRETRAINING IMPROVES REPRESENTATION BUT NOT CLINICAL PREDICTION"
    elif qcdf_was_bottleneck:
        verdict_status = "QCDF WAS THE MAIN BOTTLENECK"
    else:
        verdict_status = "NO QUANTUM-SPECIFIC VALUE DETECTED"

    # Step 6: Save Deliverables
    print("\n[6/6] Writing Deliverables and Saving Checkpoints...")

    csv_local = os.path.join(LOCAL_OUT_DIR, "PHASE12_2_MODEL_METRICS.csv")
    csv_audit = os.path.join(AUDIT_OUT_DIR, "PHASE12_2_MODEL_METRICS.csv")
    summary_primary.to_csv(csv_local, index=False)
    summary_primary.to_csv(csv_audit, index=False)

    verdict_payload = {
        "phase": "PHASE_12_2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort_size": N,
        "n_seeds": len(seeds),
        "cv_folds": 5,
        "primary_target": "target_future_prog_mcid5 (Month 12-to-60 Motor Progression)",
        "secondary_target": "target_progression_5yr_mcid5 (Cumulative Baseline-to-Month-60 Progression)",
        "verdict_status": verdict_status,
        "quantum_complementarity_detected": quantum_advantage,
        "pretraining_convergence": {
            "best_pretrain_loss_quantum": best_pretrain_loss_q,
            "best_pretrain_loss_classical": best_pretrain_loss_c,
            "mean_pretrain_loss_quantum": float(np.mean([d["final_pretrain_loss_quantum"] for d in pretrain_diagnostics])),
            "mean_pretrain_loss_classical": float(np.mean([d["final_pretrain_loss_classical"] for d in pretrain_diagnostics])),
        },
        "representation_diagnostics": diag_summary,
        "primary_model_rankings": summary_primary.sort_values(by="auroc_mean", ascending=False).to_dict(orient="records"),
        "secondary_model_rankings": summary_secondary.sort_values(by="auroc_mean", ascending=False).to_dict(orient="records"),
        "hypothesis_tests_clustered_bootstrap": bootstrap_results,
    }

    json_local = os.path.join(LOCAL_OUT_DIR, "PHASE12_2_VERDICT.json")
    json_audit = os.path.join(AUDIT_OUT_DIR, "PHASE12_2_VERDICT.json")
    with open(json_local, "w") as f:
        json.dump(verdict_payload, f, indent=2)
    with open(json_audit, "w") as f:
        json.dump(verdict_payload, f, indent=2)

    print(f"\nDeliverables written:")
    print(f"  [1] {csv_local} & {csv_audit}")
    print(f"  [2] {json_local} & {json_audit}")
    print(f"  [3] {os.path.join(LOCAL_OUT_DIR, 'best_qvbe_pretrained.pt')} & {os.path.join(AUDIT_OUT_DIR, 'best_qvbe_pretrained.pt')}")
    print(f"  [4] {os.path.join(LOCAL_OUT_DIR, 'best_classical_pretrained.pt')} & {os.path.join(AUDIT_OUT_DIR, 'best_classical_pretrained.pt')}")
    print("\n" + "=" * 80)
    print(f"OFFICIAL SCIENTIFIC VERDICT: {verdict_status}")
    print("=" * 80)


if __name__ == "__main__":
    main()

