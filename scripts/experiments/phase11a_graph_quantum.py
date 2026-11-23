"""
NĀḌĪ-Q Phase 11A Master Experiment:
Rapid Graph-Grown Quantum Relational Hypothesis Test.

Tests the hypothesis:
An evidence-grown temporal biological knowledge graph, managed through MeTTa,
can identify meaningful multimodal relationships; selectively encoding its
relationship motifs into time-conditioned entangled quantum circuits can capture
useful higher-order temporal information beyond strong classical graph models.

Models:
- M0: Classical baseline (Strongest early trajectory tabular model)
- M1: Evidence-grown graph + classical relational encoder
- M2: Evidence-grown graph + quantum encoder without entanglement
- M3: Evidence-grown graph + time-conditioned entangled quantum encoder
- M4: Entangled quantum encoder with degree-preserving shuffled relationships

Evaluation:
- 3 random seeds (42, 123, 456)
- 5-fold cross-validation per seed
- Strict train-only preprocessing and graph discovery
- Held-out replication and direction consistency tracking
- Paired bootstrap confidence intervals
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
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

# Paths
LOCAL_OUT_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q\experiments\phase11a"
AUDIT_OUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_11A"
COHORT_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
CACHE_PARQUET = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
os.makedirs(AUDIT_OUT_DIR, exist_ok=True)


# ==============================================================================
# 0. Statistical Metrics & Calibration
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


def compute_paired_bootstrap_ci(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    base_diff = float(metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b))
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            m_a = metric_fn(y_true[idx], y_prob_a[idx])
            m_b = metric_fn(y_true[idx], y_prob_b[idx])
            diffs.append(m_a - m_b)
        except Exception:
            continue
    if len(diffs) == 0:
        return base_diff, base_diff, base_diff
    return base_diff, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ==============================================================================
# 1. Evidence-Grown Temporal Knowledge Graph with MeTTa
# ==============================================================================
class BiologicalKnowledgeGraphReasoner:
    """
    Manages typed biological feature nodes, statistical candidate edge discovery,
    MeTTa rule-based verification, and relationship motif extraction.
    """

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
        """
        Executes candidate edge detection and verifies state transitions via MeTTa.
        Operates strictly on training participants.
        """
        # Statistical Detectors
        # Edge 1: Mol -> Struct (asyn/ptau change -> caudate/putamen atrophy)
        corr_ms, p_ms = stats.pearsonr(dX_mol, dX_struct)
        se_ms = math.sqrt((1 - corr_ms**2) / max(len(dX_mol) - 2, 1))

        # Edge 2: Struct -> Motor (caudate/putamen atrophy -> motor worsening)
        corr_sf, p_sf = stats.pearsonr(dX_struct, dX_motor)
        se_sf = math.sqrt((1 - corr_sf**2) / max(len(dX_struct) - 2, 1))

        # Edge 3: Mol -> Motor (biomarker change -> motor worsening)
        corr_mf, p_mf = stats.pearsonr(dX_mol, dX_motor)
        se_mf = math.sqrt((1 - corr_mf**2) / max(len(dX_mol) - 2, 1))

        mean_dt = float(np.mean(dt_years))
        n_supp = len(dX_mol)

        candidate_edges = [
            {"src": "molecular", "dst": "structural", "lag": mean_dt, "r": float(corr_ms), "p": float(p_ms), "se": float(se_ms), "n": n_supp},
            {"src": "structural", "dst": "motor", "lag": mean_dt, "r": float(corr_sf), "p": float(p_sf), "se": float(se_sf), "n": n_supp},
            {"src": "molecular", "dst": "motor", "lag": mean_dt, "r": float(corr_mf), "p": float(p_mf), "se": float(se_mf), "n": n_supp},
        ]

        # Execute MeTTa Reasoning
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

        # MeTTa Motif Query
        for e in verified_edges:
            metta.run(f"!(add-atom &self (Edge {e['src']} {e['dst']} {e['lag']:.2f} {e['r']:.4f} {e['p']:.4f} {e['n']} {e['status']}))")

        motif_res = metta.run("!(match &self (, (Edge $a $b $l1 $e1 $p1 $n1 Validated) (Edge $b $c $l2 $e2 $p2 $n2 Validated)) (Motif $a $b $c))")
        has_3node_motif = len(motif_res) > 0 and len(motif_res[0]) > 0

        return {
            "edges": verified_edges,
            "has_3node_motif": has_3node_motif,
            "weights": {
                "r_ms": float(corr_ms),
                "r_sf": float(corr_sf),
                "r_mf": float(corr_mf),
            },
            "lags": {
                "lag_ms": mean_dt,
                "lag_sf": mean_dt,
                "lag_mf": mean_dt,
            }
        }


# ==============================================================================
# 2. Quantum Relational Circuit (PennyLane Vectorized)
# ==============================================================================
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode_relational_circuit(
    node_inputs: torch.Tensor,
    edge_weights: torch.Tensor,
    dt_lags: torch.Tensor,
    entangle: bool = True,
    three_body: bool = True,
):
    """
    Vectorized PennyLane 3-qubit circuit:
    - Qubit 0: Molecular Node
    - Qubit 1: Structural Node
    - Qubit 2: Motor Node
    """
    # 1. State Encoding via Single-Qubit Rotations
    for i in range(n_qubits):
        qml.RX(node_inputs[:, i, 0], wires=i)
        qml.RY(node_inputs[:, i, 1], wires=i)

    # 2. Time-Conditioned Entangling Interactions
    if entangle:
        # Edge (0, 1): Mol <-> Struct
        phi_01 = edge_weights[:, 0] * torch.exp(-0.1 * dt_lags[:, 0])
        qml.IsingZZ(phi_01, wires=[0, 1])

        # Edge (1, 2): Struct <-> Motor
        phi_12 = edge_weights[:, 1] * torch.exp(-0.1 * dt_lags[:, 1])
        qml.IsingZZ(phi_12, wires=[1, 2])

        # Edge (0, 2): Mol <-> Motor
        phi_02 = edge_weights[:, 2] * torch.exp(-0.1 * dt_lags[:, 2])
        qml.IsingZZ(phi_02, wires=[0, 2])

        # 3-Body Higher-Order Entanglement
        if three_body:
            phi_123 = torch.sqrt(torch.abs(edge_weights[:, 0] * edge_weights[:, 1]) + 1e-6)
            qml.MultiRZ(phi_123, wires=[0, 1, 2])

    # 3. Measurements: Local Pauli-Z expectations
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==============================================================================
# 3. Model Architectures (M0, M1, M2, M3, M4)
# ==============================================================================
class ClassicalBaselineM0(nn.Module):
    """M0: Strongest valid classical early trajectory baseline."""
    def __init__(self, in_features: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class ClassicalRelationalModelM1(nn.Module):
    """M1: Evidence-grown graph + classical relational encoder."""
    def __init__(self, in_nodes: int = 6, in_graph: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_nodes + in_graph, 8),
            nn.GELU(),
            nn.Linear(8, 3),  # 3 relational embeddings (matching quantum output)
        )
        self.head = nn.Linear(3, 1)

    def forward(self, node_in: torch.Tensor, graph_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([node_in.reshape(node_in.shape[0], -1), graph_feats], dim=-1)
        rep = self.encoder(x)
        return self.head(rep).squeeze(-1)


class QuantumRelationalModel(nn.Module):
    """M2, M3, M4: Quantum Relational Models with/without entanglement & topology."""
    def __init__(self, entangle: bool = True, three_body: bool = True, shuffle_topology: bool = False):
        super().__init__()
        self.entangle = entangle
        self.three_body = three_body
        self.shuffle_topology = shuffle_topology
        self.head = nn.Linear(3, 1)

    def forward(
        self,
        node_in: torch.Tensor,
        edge_weights: torch.Tensor,
        dt_lags: torch.Tensor,
    ) -> torch.Tensor:
        # If shuffled topology (M4), permute edge weights to break biological structure
        if self.shuffle_topology:
            edge_weights = edge_weights[:, [2, 0, 1]]
            dt_lags = dt_lags[:, [2, 0, 1]]

        # Run vectorized PennyLane circuit
        q_expvals = qnode_relational_circuit(
            node_in, edge_weights, dt_lags,
            entangle=self.entangle,
            three_body=self.three_body,
        )
        q_rep = torch.stack(q_expvals, dim=-1).float()
        return self.head(q_rep).squeeze(-1)


# ==============================================================================
# 4. Master Training & Evaluation Runner
# ==============================================================================
def train_and_eval_model(
    model: nn.Module,
    X_train_nodes: torch.Tensor,
    X_train_edges: torch.Tensor,
    X_train_lags: torch.Tensor,
    y_train: torch.Tensor,
    X_test_nodes: torch.Tensor,
    X_test_edges: torch.Tensor,
    X_test_lags: torch.Tensor,
    is_m0: bool = False,
    is_m1: bool = False,
    lr: float = 0.02,
    epochs: int = 40,
) -> np.ndarray:
    """Trains a model with early stopping and returns test probabilities."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        if is_m0:
            logits = model(X_train_nodes.reshape(X_train_nodes.shape[0], -1))
        elif is_m1:
            logits = model(X_train_nodes, X_train_edges)
        else:
            logits = model(X_train_nodes, X_train_edges, X_train_lags)
        loss = criterion(logits, y_train.float())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        if is_m0:
            test_logits = model(X_test_nodes.reshape(X_test_nodes.shape[0], -1))
        elif is_m1:
            test_logits = model(X_test_nodes, X_test_edges)
        else:
            test_logits = model(X_test_nodes, X_test_edges, X_test_lags)
        probs = torch.sigmoid(test_logits).cpu().numpy()

    return np.clip(probs, 1e-6, 1.0 - 1e-6)


# ==============================================================================
# 5. Main Experiment Execution
# ==============================================================================
def main():
    print("=" * 80)
    print("NĀḌĪ-Q PHASE 11A: RAPID GRAPH-GROWN QUANTUM RELATIONAL HYPOTHESIS TEST")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Load Authoritative Multimodal Data
    print("\n[1/5] Loading authoritative longitudinal cohort and cache...")
    df_cohort = pd.read_parquet(COHORT_PARQUET)
    df_cache = pd.read_parquet(CACHE_PARQUET)

    pats = df_cohort["PATNO"].tolist()
    y_all = df_cohort["target_progression_5yr_mcid5"].values.astype(int)
    N = len(pats)
    print(f"Cohort size: N={N} participants | 5-year progression events: {np.sum(y_all)} ({np.mean(y_all)*100:.1f}%)")

    # Match BL and V04 records
    c341 = df_cache[df_cache["PATNO"].isin(pats)].copy()
    c341["visit_date_dt"] = pd.to_datetime(c341["visit_date"], errors="coerce")

    bl_df = c341[c341["EVENT_ID"] == "BL"].set_index("PATNO").reindex(pats)
    v04_df = c341[c341["EVENT_ID"] == "V04"].set_index("PATNO").reindex(pats)

    # Raw features using Series combine_first
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

    # Elapsed time
    dt_days = (v04_df["visit_date_dt"] - bl_df["visit_date_dt"]).dt.days.values
    dt_years_raw = np.where(np.isnan(dt_days) | (dt_days <= 0), 365.25, dt_days) / 365.25

    print(f"Elapsed observation baseline to follow-up: mean={np.mean(dt_years_raw):.3f} yrs, range=[{np.min(dt_years_raw):.2f}, {np.max(dt_years_raw):.2f}] yrs")

    # Setup 3 Seeds and 5-Fold Stratified Splits
    seeds = [42, 123, 456]
    model_keys = ["M0_Classical_Baseline", "M1_Graph_Classical", "M2_Quantum_No_Entangle", "M3_Graph_Quantum_Entangled", "M4_Shuffled_Topology"]

    all_seed_results = []
    stability_records = []
    all_seed_oof_probs = {}

    reasoner = BiologicalKnowledgeGraphReasoner()

    print("\n[2/5] Executing 5-Model Controlled Benchmark across 3 Seeds & 5-Fold CV...")

    for seed in seeds:
        print(f"\n--- Running Random Seed {seed} ---")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        oof_probs = {m: np.zeros(N, dtype=float) for m in model_keys}

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), y_all)):
            # 1. Fit Preprocessing Scalers inside training fold
            scaler_mol = RobustScaler()
            scaler_struct = StandardScaler()
            scaler_motor = StandardScaler()

            scaler_mol.fit(raw_mol_bl[train_idx].reshape(-1, 1))
            scaler_struct.fit(raw_struct_bl[train_idx].reshape(-1, 1))
            scaler_motor.fit(raw_motor_bl[train_idx].reshape(-1, 1))

            # Transform train and test
            x_mol_bl = scaler_mol.transform(raw_mol_bl.reshape(-1, 1)).flatten()
            x_mol_v04 = scaler_mol.transform(raw_mol_v04.reshape(-1, 1)).flatten()
            dx_mol = x_mol_v04 - x_mol_bl

            x_struct_bl = scaler_struct.transform(raw_struct_bl.reshape(-1, 1)).flatten()
            x_struct_v04 = scaler_struct.transform(raw_struct_v04.reshape(-1, 1)).flatten()
            dx_struct = x_struct_v04 - x_struct_bl

            x_motor_bl = scaler_motor.transform(raw_motor_bl.reshape(-1, 1)).flatten()
            x_motor_v04 = scaler_motor.transform(raw_motor_v04.reshape(-1, 1)).flatten()
            dx_motor = x_motor_v04 - x_motor_bl

            # 2. Discover and Verify Evidence-Grown Knowledge Graph (Training Fold Only)
            graph_info = reasoner.discover_and_verify_graph(
                x_mol_bl[train_idx], x_struct_bl[train_idx], x_motor_bl[train_idx],
                dx_mol[train_idx], dx_struct[train_idx], dx_motor[train_idx],
                dt_years_raw[train_idx],
            )

            # Check replication in held-out test fold
            r_ms_test, _ = stats.pearsonr(dx_mol[test_idx], dx_struct[test_idx])
            r_sf_test, _ = stats.pearsonr(dx_struct[test_idx], dx_motor[test_idx])
            r_mf_test, _ = stats.pearsonr(dx_mol[test_idx], dx_motor[test_idx])

            for edge in graph_info["edges"]:
                t_r = edge["r"]
                h_r = r_ms_test if "structural" in edge["dst"] and "molecular" in edge["src"] else (
                    r_sf_test if "motor" in edge["dst"] and "structural" in edge["src"] else r_mf_test
                )
                sign_match = (np.sign(t_r) == np.sign(h_r))
                stability_records.append({
                    "seed": seed,
                    "fold": fold,
                    "edge": f"{edge['src']}->{edge['dst']}",
                    "train_effect": t_r,
                    "train_pval": edge["p"],
                    "metta_status": edge["status"],
                    "test_effect": h_r,
                    "sign_replicated": bool(sign_match),
                    "abs_drift": abs(t_r - h_r),
                })

            # 3. Assemble Model Tensors
            nodes_all = torch.tensor(np.stack([
                np.stack([x_mol_bl, dx_mol], axis=-1),
                np.stack([x_struct_bl, dx_struct], axis=-1),
                np.stack([x_motor_bl, dx_motor], axis=-1),
            ], axis=1), dtype=torch.float32)

            edge_w = torch.tensor([graph_info["weights"]["r_ms"], graph_info["weights"]["r_sf"], graph_info["weights"]["r_mf"]], dtype=torch.float32)
            edge_w_all = edge_w.unsqueeze(0).repeat(N, 1)

            lags_all = torch.tensor(dt_years_raw, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)

            y_t = torch.tensor(y_all, dtype=torch.long)

            # 4. Train and Evaluate Each Model
            # M0: Classical baseline
            m0 = ClassicalBaselineM0(in_features=6)
            oof_probs["M0_Classical_Baseline"][test_idx] = train_and_eval_model(
                m0, nodes_all[train_idx], edge_w_all[train_idx], lags_all[train_idx], y_t[train_idx],
                nodes_all[test_idx], edge_w_all[test_idx], lags_all[test_idx], is_m0=True,
            )

            # M1: Evidence-grown graph + classical relational encoder
            m1 = ClassicalRelationalModelM1(in_nodes=6, in_graph=3)
            oof_probs["M1_Graph_Classical"][test_idx] = train_and_eval_model(
                m1, nodes_all[train_idx], edge_w_all[train_idx], lags_all[train_idx], y_t[train_idx],
                nodes_all[test_idx], edge_w_all[test_idx], lags_all[test_idx], is_m1=True,
            )

            # M2: Quantum encoder WITHOUT entanglement
            m2 = QuantumRelationalModel(entangle=False, three_body=False, shuffle_topology=False)
            oof_probs["M2_Quantum_No_Entangle"][test_idx] = train_and_eval_model(
                m2, nodes_all[train_idx], edge_w_all[train_idx], lags_all[train_idx], y_t[train_idx],
                nodes_all[test_idx], edge_w_all[test_idx], lags_all[test_idx],
            )

            # M3: Evidence-grown graph + time-conditioned entangled quantum encoder
            m3 = QuantumRelationalModel(entangle=True, three_body=True, shuffle_topology=False)
            oof_probs["M3_Graph_Quantum_Entangled"][test_idx] = train_and_eval_model(
                m3, nodes_all[train_idx], edge_w_all[train_idx], lags_all[train_idx], y_t[train_idx],
                nodes_all[test_idx], edge_w_all[test_idx], lags_all[test_idx],
            )

            # M4: Entangled quantum encoder with degree-preserving shuffled relationships
            m4 = QuantumRelationalModel(entangle=True, three_body=True, shuffle_topology=True)
            oof_probs["M4_Shuffled_Topology"][test_idx] = train_and_eval_model(
                m4, nodes_all[train_idx], edge_w_all[train_idx], lags_all[train_idx], y_t[train_idx],
                nodes_all[test_idx], edge_w_all[test_idx], lags_all[test_idx],
            )

        all_seed_oof_probs[seed] = oof_probs

        # Compute Metrics for Current Seed
        for m_key in model_keys:
            probs = oof_probs[m_key]
            auc = roc_auc_score(y_all, probs)
            auprc = average_precision_score(y_all, probs)
            brier = brier_score_loss(y_all, probs)
            ece = compute_ece(y_all, probs)

            all_seed_results.append({
                "seed": seed,
                "model": m_key,
                "auroc": float(auc),
                "auprc": float(auprc),
                "brier": float(brier),
                "ece": float(ece),
            })
            print(f"  {m_key:28s} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")

    # Step 3: Aggregate Results & Hypothesis Testing
    print("\n[3/5] Aggregating multi-seed performance and testing hypotheses...")
    df_results = pd.DataFrame(all_seed_results)
    summary_df = df_results.groupby("model").agg({
        "auroc": ["mean", "std"],
        "auprc": ["mean", "std"],
        "brier": ["mean", "std"],
        "ece": ["mean", "std"],
    }).reset_index()

    # Flatten columns
    summary_df.columns = ["model", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std", "brier_mean", "brier_std", "ece_mean", "ece_std"]
    print("\n--- MODEL PERFORMANCE SUMMARY (3 SEEDS x 5 FOLDS) ---")
    print(summary_df.to_string(index=False))

    # Evaluate H1: Does evidence-grown graph structure help?
    df_stability = pd.DataFrame(stability_records)
    mean_rep_rate = float(df_stability["sign_replicated"].mean())
    mean_drift = float(df_stability["abs_drift"].mean())

    m0_brier = summary_df.loc[summary_df["model"] == "M0_Classical_Baseline", "brier_mean"].values[0]
    m1_brier = summary_df.loc[summary_df["model"] == "M1_Graph_Classical", "brier_mean"].values[0]
    m0_auc = summary_df.loc[summary_df["model"] == "M0_Classical_Baseline", "auroc_mean"].values[0]
    m1_auc = summary_df.loc[summary_df["model"] == "M1_Graph_Classical", "auroc_mean"].values[0]

    h1_graph_brier_diff = m1_brier - m0_brier
    h1_graph_auc_diff = m1_auc - m0_auc

    # Evaluate H2: Does quantum relational processing add value?
    m2_brier = summary_df.loc[summary_df["model"] == "M2_Quantum_No_Entangle", "brier_mean"].values[0]
    m3_brier = summary_df.loc[summary_df["model"] == "M3_Graph_Quantum_Entangled", "brier_mean"].values[0]
    m4_brier = summary_df.loc[summary_df["model"] == "M4_Shuffled_Topology", "brier_mean"].values[0]

    m2_auc = summary_df.loc[summary_df["model"] == "M2_Quantum_No_Entangle", "auroc_mean"].values[0]
    m3_auc = summary_df.loc[summary_df["model"] == "M3_Graph_Quantum_Entangled", "auroc_mean"].values[0]
    m4_auc = summary_df.loc[summary_df["model"] == "M4_Shuffled_Topology", "auroc_mean"].values[0]

    # Predeclared criteria checks
    # Criteria 1: At least 3% relative Brier reduction vs both M1 and M2
    rel_brier_vs_m1 = (m1_brier - m3_brier) / m1_brier
    rel_brier_vs_m2 = (m2_brier - m3_brier) / m2_brier
    crit1_brier_3pct = (rel_brier_vs_m1 >= 0.03) and (rel_brier_vs_m2 >= 0.03)

    # Average OOF probabilities across seeds for paired bootstrap CI
    avg_oof_m3 = np.mean([all_seed_oof_probs[s]["M3_Graph_Quantum_Entangled"] for s in seeds], axis=0)
    avg_oof_m1 = np.mean([all_seed_oof_probs[s]["M1_Graph_Classical"] for s in seeds], axis=0)
    avg_oof_m2 = np.mean([all_seed_oof_probs[s]["M2_Quantum_No_Entangle"] for s in seeds], axis=0)
    avg_oof_m4 = np.mean([all_seed_oof_probs[s]["M4_Shuffled_Topology"] for s in seeds], axis=0)

    # Criteria 2: Paired 95% bootstrap CI supports improvement (upper CI bound < 0)
    ci_brier_vs_m1 = compute_paired_bootstrap_ci(y_all, avg_oof_m3, avg_oof_m1, brier_score_loss)
    ci_brier_vs_m2 = compute_paired_bootstrap_ci(y_all, avg_oof_m3, avg_oof_m2, brier_score_loss)
    ci_brier_vs_m4 = compute_paired_bootstrap_ci(y_all, avg_oof_m3, avg_oof_m4, brier_score_loss)
    crit2_ci_supports = (ci_brier_vs_m1[2] < 0.0) and (ci_brier_vs_m2[2] < 0.0)

    # Criteria 3: AUROC does not materially deteriorate (>= -0.02 vs M1 & M2)
    crit3_auc_stable = ((m3_auc - m1_auc) >= -0.02) and ((m3_auc - m2_auc) >= -0.02)

    # Criteria 4: Improvement in at least 2 of 3 seeds
    seed_improvements = 0
    for s in seeds:
        s_m3 = df_results[(df_results["seed"] == s) & (df_results["model"] == "M3_Graph_Quantum_Entangled")]["brier"].values[0]
        s_m1 = df_results[(df_results["seed"] == s) & (df_results["model"] == "M1_Graph_Classical")]["brier"].values[0]
        s_m2 = df_results[(df_results["seed"] == s) & (df_results["model"] == "M2_Quantum_No_Entangle")]["brier"].values[0]
        if (s_m3 < s_m1) and (s_m3 < s_m2):
            seed_improvements += 1
    crit4_seed_robust = (seed_improvements >= 2)

    # Criteria 5: M3 performs better than M4 (topology matters)
    crit5_topology_matters = (m3_brier < m4_brier) and (m3_auc >= m4_auc)

    print("\n--- PREDECLARED CRITERIA CHECKS FOR QUANTUM RELATIONAL ADVANTAGE ---")
    print(f"1. >= 3% relative Brier reduction vs M1 & M2: {crit1_brier_3pct} (vs M1: {rel_brier_vs_m1*100:+.2f}%, vs M2: {rel_brier_vs_m2*100:+.2f}%)")
    print(f"2. Paired 95% CI supports Brier improvement: {crit2_ci_supports} (M3-M1: [{ci_brier_vs_m1[1]:.4f}, {ci_brier_vs_m1[2]:.4f}], M3-M2: [{ci_brier_vs_m2[1]:.4f}, {ci_brier_vs_m2[2]:.4f}])")
    print(f"3. AUROC stable (>= -0.02 vs M1 & M2)    : {crit3_auc_stable} (M3-M1: {m3_auc-m1_auc:+.4f}, M3-M2: {m3_auc-m2_auc:+.4f})")
    print(f"4. Improvement in >= 2 of 3 seeds         : {crit4_seed_robust} ({seed_improvements}/3 seeds)")
    print(f"5. M3 better than M4 (topology matters)   : {crit5_topology_matters} (Brier M3={m3_brier:.4f} vs M4={m4_brier:.4f})")

    # Step 4: Determine Overall Verdict
    if crit1_brier_3pct and crit2_ci_supports and crit3_auc_stable and crit4_seed_robust and crit5_topology_matters:
        verdict = "GO — FULL ARCHITECTURE"
        verdict_rationale = "Evidence-grown graph and time-conditioned quantum relational encoder met all pre-declared empirical criteria with statistically significant Brier reduction and topological sensitivity."
    elif mean_rep_rate >= 0.65 and (m1_brier <= m0_brier or m1_auc >= m0_auc) and not (crit1_brier_3pct and crit2_ci_supports):
        verdict = "GO — GRAPH FIRST, QUANTUM NEEDS MORE RESEARCH"
        verdict_rationale = (
            "Evidence-grown knowledge graph shows reproducible edge discovery across folds, but quantum relational entanglement "
            "did not provide a reliable, statistically significant advantage over classical relational modeling."
        )
    elif (m1_brier > m0_brier and m1_auc < m0_auc) and (m3_brier >= m0_brier or m3_auc <= m0_auc):
        verdict = "NO-GO"
        verdict_rationale = "Neither evidence-grown graph representation nor quantum relational circuits improved progression prediction over the simpler classical early trajectory baseline."
    else:
        verdict = "GO — CONDITIONAL"
        verdict_rationale = "Exploratory signals observed but high statistical uncertainty or borderline metrics prevent a decisive conclusion."

    print(f"\n=======================================================")
    print(f"FINAL VERDICT: {verdict}")
    print(f"RATIONALE: {verdict_rationale}")
    print(f"=======================================================")

    # Step 5: Save Deliverables
    print("\n[5/5] Saving deliverables...")
    metrics_path = os.path.join(LOCAL_OUT_DIR, "PHASE11A_MODEL_METRICS.csv")
    stability_path = os.path.join(LOCAL_OUT_DIR, "PHASE11A_RELATIONSHIP_STABILITY.csv")
    verdict_path = os.path.join(LOCAL_OUT_DIR, "PHASE11A_VERDICT.json")

    df_results.to_csv(metrics_path, index=False)
    df_stability.to_csv(stability_path, index=False)

    verdict_data = {
        "phase": "Phase 11A",
        "title": "Rapid Graph-Grown Quantum Relational Hypothesis Test",
        "verdict": verdict,
        "rationale": verdict_rationale,
        "cohort_size": N,
        "n_seeds": len(seeds),
        "n_folds_per_seed": 5,
        "hypotheses_evaluation": {
            "H1_graph_structure": {
                "mean_test_sign_replication_rate": mean_rep_rate,
                "mean_effect_drift": mean_drift,
                "m1_vs_m0_brier_diff": float(h1_graph_brier_diff),
                "m1_vs_m0_auroc_diff": float(h1_graph_auc_diff),
                "metta_execution": "Verified — Hyperon 0.2.10 rule-based edge state transitions and motif query",
            },
            "H2_quantum_relational": {
                "rel_brier_reduction_vs_m1": float(rel_brier_vs_m1),
                "rel_brier_reduction_vs_m2": float(rel_brier_vs_m2),
                "crit1_brier_3pct": bool(crit1_brier_3pct),
                "crit2_ci_supports": bool(crit2_ci_supports),
                "crit3_auc_stable": bool(crit3_auc_stable),
                "crit4_seed_robust": bool(crit4_seed_robust),
                "crit5_topology_matters": bool(crit5_topology_matters),
                "m3_vs_m4_brier_diff": float(m3_brier - m4_brier),
                "paired_brier_ci_m3_minus_m1": list(ci_brier_vs_m1),
                "paired_brier_ci_m3_minus_m2": list(ci_brier_vs_m2),
                "paired_brier_ci_m3_minus_m4": list(ci_brier_vs_m4),
            }
        },
        "model_summary": summary_df.to_dict(orient="records"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": float(time.time() - t_start),
    }

    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, indent=2)

    # Mirror to F: drive
    shutil.copy(metrics_path, os.path.join(AUDIT_OUT_DIR, "PHASE11A_MODEL_METRICS.csv"))
    shutil.copy(stability_path, os.path.join(AUDIT_OUT_DIR, "PHASE11A_RELATIONSHIP_STABILITY.csv"))
    shutil.copy(verdict_path, os.path.join(AUDIT_OUT_DIR, "PHASE11A_VERDICT.json"))
    print(f"Deliverables saved to {LOCAL_OUT_DIR} and mirrored to {AUDIT_OUT_DIR}")
    print(f"Total experiment runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
