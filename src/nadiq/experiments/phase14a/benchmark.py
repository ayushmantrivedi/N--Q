"""
NĀḌĪ-Q Phase 14A: Benchmark Engine for M0-M9 and Ablations.
Trains and evaluates all 11 model configurations strictly inside cross-validation folds:
- M0: Logistic Regression (linear baseline)
- M1: RBF-SVM (non-linear classical baseline)
- M2: Poly-SVM deg 2 (degree-2 polynomial baseline)
- M3: Classical Relational Kernel SVM (matched classical relational control)
- M4: Classical Relational Linear/MLP (explicit feature baseline)
- M5: Biological Relational Quantum Kernel SVM (BRQK)
- M6: CTCLM Baseline (temporal motor model)
- M7: CTCLM + Matched Classical Relational Kernel (classical hybrid)
- M8: CTCLM + Quantum Relational Kernel (quantum hybrid)
- M9: Quantum Kernel with Shuffled Topology (negative control)
- M5_NoEntangle: Quantum Kernel without Entanglement (ablation control)
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from nadiq.models.quantum.brqk.config import BRQKConfig
from nadiq.models.quantum.brqk.quantum_kernel import BiologicalRelationalQuantumKernel
from nadiq.models.quantum.brqk.kernel_validation import validate_kernel_matrix, regularize_kernel_matrix
from nadiq.models.classical.relational_kernel import ClassicalRelationalKernel
from nadiq.models.quantum.brqk.feature_map import FeatureStandardizer
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel, CTCLMConfig


class CTCLMClassifier(nn.Module):
    """Temporal motor model using CTCLM latent representation."""

    def __init__(self, latent_dim: int = 16, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.ctclm = ContinualTemporalClinicalLatentModel(
            CTCLMConfig(input_dim=1, latent_dim=latent_dim, time_embedding_dim=4, seed=seed)
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        seq: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.ctclm(seq, times, mask)
        logits = self.head(out.h_final).squeeze(-1)
        return logits


class Phase14ABenchmarkRunner:
    """
    Orchestrates training, kernel Gram matrix generation, and inference across all models for a single fold.
    """

    def __init__(
        self,
        biological_pairs: Optional[List[Tuple[int, int]]] = None,
        shuffled_pairs: Optional[List[Tuple[int, int]]] = None,
        svm_c: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.biological_pairs = biological_pairs or [
            (0, 2), (1, 2), (0, 3), (1, 3),
            (2, 4), (3, 4), (3, 5), (0, 4),
        ]
        # Shuffled / scrambled pairing negative control
        self.shuffled_pairs = shuffled_pairs or [
            (0, 1), (2, 3), (4, 5), (0, 5),
            (1, 4), (2, 5), (1, 5), (0, 2),
        ]
        self.svm_c = svm_c
        self.random_state = random_state

    def run_fold(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
        ctclm_inputs: Optional[Dict[str, np.ndarray]] = None,
        train_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute all models on one cross-validation fold.
        Returns dictionary mapping model_name to {'prob': np.ndarray, 'extra': dict}.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # -------------------------------------------------------------
        # Feature standardization for classical non-kernel models
        # -------------------------------------------------------------
        scaler = FeatureStandardizer()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # =============================================================
        # M0: Logistic Regression (Linear Baseline)
        # =============================================================
        lr = LogisticRegression(C=self.svm_c, max_iter=1000, random_state=seed)
        lr.fit(X_tr_scaled, y_tr)
        p_m0 = lr.predict_proba(X_te_scaled)[:, 1]
        results["M0_Logistic_Regression"] = {"prob": p_m0, "decision": lr.decision_function(X_te_scaled)}

        # =============================================================
        # M1: RBF-SVM (Classical Kernel Baseline)
        # =============================================================
        svm_rbf = SVC(kernel="rbf", C=self.svm_c, probability=True, random_state=seed)
        svm_rbf.fit(X_tr_scaled, y_tr)
        p_m1 = svm_rbf.predict_proba(X_te_scaled)[:, 1]
        results["M1_RBF_SVM"] = {"prob": p_m1, "decision": svm_rbf.decision_function(X_te_scaled)}

        # =============================================================
        # M2: Poly-SVM (Degree 2 Polynomial Baseline)
        # =============================================================
        svm_poly = SVC(kernel="poly", degree=2, C=self.svm_c, coef0=1.0, probability=True, random_state=seed)
        svm_poly.fit(X_tr_scaled, y_tr)
        p_m2 = svm_poly.predict_proba(X_te_scaled)[:, 1]
        results["M2_Poly_SVM"] = {"prob": p_m2, "decision": svm_poly.decision_function(X_te_scaled)}

        # =============================================================
        # M3: Classical Relational-Feature Kernel SVM
        # =============================================================
        crk = ClassicalRelationalKernel(interaction_pairs=self.biological_pairs, kernel_type="rbf")
        crk.fit(X_tr)
        K_crk_tr = crk.compute_gram_matrix(X_tr)
        K_crk_te = crk.compute_gram_matrix(X_tr, X_te)

        svm_m3 = SVC(kernel="precomputed", C=self.svm_c, probability=True, random_state=seed)
        svm_m3.fit(K_crk_tr, y_tr)
        p_m3 = svm_m3.predict_proba(K_crk_te)[:, 1]
        dec_m3_tr = svm_m3.decision_function(K_crk_tr)
        dec_m3_te = svm_m3.decision_function(K_crk_te)
        results["M3_Classical_Relational_Kernel"] = {"prob": p_m3, "decision": dec_m3_te, "train_decision": dec_m3_tr}

        # =============================================================
        # M4: Classical Relational Feature Classifier (Explicit phi)
        # =============================================================
        phi_tr = crk.compute_relational_features(X_tr)
        phi_te = crk.compute_relational_features(X_te)
        lr_m4 = LogisticRegression(C=self.svm_c, max_iter=1000, random_state=seed)
        lr_m4.fit(phi_tr, y_tr)
        p_m4 = lr_m4.predict_proba(phi_te)[:, 1]
        results["M4_Classical_Relational_Classifier"] = {"prob": p_m4, "decision": lr_m4.decision_function(phi_te)}

        # =============================================================
        # M5: Quantum Relational Kernel SVM (BRQK)
        # =============================================================
        brqk_cfg = BRQKConfig(
            n_qubits=6,
            feature_dim=6,
            interaction_pairs=self.biological_pairs,
            entanglement_type="biological",
            simulation_mode="statevector",
            seed=seed,
        )
        brqk = BiologicalRelationalQuantumKernel(brqk_cfg)
        brqk.fit(X_tr)
        K_brqk_tr = brqk.compute_gram_matrix(X_tr)
        K_brqk_te = brqk.compute_gram_matrix(X_tr, X_te)

        # Validate Gram matrix
        val_m5 = validate_kernel_matrix(K_brqk_tr)
        if not val_m5["is_psd"]:
            K_brqk_tr = regularize_kernel_matrix(K_brqk_tr, ridge=1e-5)

        svm_m5 = SVC(kernel="precomputed", C=self.svm_c, probability=True, random_state=seed)
        svm_m5.fit(K_brqk_tr, y_tr)
        p_m5 = svm_m5.predict_proba(K_brqk_te)[:, 1]
        dec_m5_tr = svm_m5.decision_function(K_brqk_tr)
        dec_m5_te = svm_m5.decision_function(K_brqk_te)
        results["M5_Quantum_Relational_Kernel"] = {
            "prob": p_m5,
            "decision": dec_m5_te,
            "train_decision": dec_m5_tr,
            "validation": val_m5,
        }

        # =============================================================
        # M9: Quantum Kernel with Shuffled Topology (Negative Control)
        # =============================================================
        shuff_cfg = BRQKConfig(
            n_qubits=6,
            feature_dim=6,
            interaction_pairs=self.shuffled_pairs,
            entanglement_type="shuffled",
            simulation_mode="statevector",
            seed=seed,
        )
        brqk_shuff = BiologicalRelationalQuantumKernel(shuff_cfg)
        brqk_shuff.fit(X_tr)
        K_shuff_tr = brqk_shuff.compute_gram_matrix(X_tr)
        K_shuff_te = brqk_shuff.compute_gram_matrix(X_tr, X_te)

        svm_m9 = SVC(kernel="precomputed", C=self.svm_c, probability=True, random_state=seed)
        svm_m9.fit(K_shuff_tr, y_tr)
        p_m9 = svm_m9.predict_proba(K_shuff_te)[:, 1]
        results["M9_Quantum_Shuffled_Topology"] = {"prob": p_m9, "decision": svm_m9.decision_function(K_shuff_te)}

        # =============================================================
        # Ablation: Quantum Kernel No Entanglement
        # =============================================================
        no_ent_cfg = BRQKConfig(
            n_qubits=6,
            feature_dim=6,
            interaction_pairs=[],
            entanglement_type="none",
            simulation_mode="statevector",
            seed=seed,
        )
        brqk_no_ent = BiologicalRelationalQuantumKernel(no_ent_cfg)
        brqk_no_ent.fit(X_tr)
        K_no_ent_tr = brqk_no_ent.compute_gram_matrix(X_tr)
        K_no_ent_te = brqk_no_ent.compute_gram_matrix(X_tr, X_te)

        svm_no_ent = SVC(kernel="precomputed", C=self.svm_c, probability=True, random_state=seed)
        svm_no_ent.fit(K_no_ent_tr, y_tr)
        p_no_ent = svm_no_ent.predict_proba(K_no_ent_te)[:, 1]
        results["M5_Ablation_No_Entanglement"] = {"prob": p_no_ent, "decision": svm_no_ent.decision_function(K_no_ent_te)}

        # =============================================================
        # M6: CTCLM Baseline (Temporal Motor Model)
        # =============================================================
        dec_m6_tr = np.zeros(len(X_tr))
        dec_m6_te = np.zeros(len(X_te))
        p_m6 = np.full(len(X_te), float(np.mean(y_tr)))

        if ctclm_inputs is not None and train_idx is not None and test_idx is not None:
            seq_t = torch.tensor(ctclm_inputs["seq"], dtype=torch.float32)
            times_t = torch.tensor(ctclm_inputs["times"], dtype=torch.float32)
            mask_t = torch.tensor(ctclm_inputs["mask"], dtype=torch.float32)
            y_t = torch.tensor(y_tr, dtype=torch.float32)

            seq_tr, times_tr, mask_tr = seq_t[train_idx], times_t[train_idx], mask_t[train_idx]
            seq_te, times_te, mask_te = seq_t[test_idx], times_t[test_idx], mask_t[test_idx]

            ctclm_model = CTCLMClassifier(latent_dim=16, seed=seed)
            optimizer = optim.Adam(ctclm_model.parameters(), lr=0.01)
            criterion = nn.BCEWithLogitsLoss()

            ctclm_model.train()
            for _ in range(25):
                optimizer.zero_grad()
                lg_tr = ctclm_model(seq_tr, times_tr, mask_tr)
                loss = criterion(lg_tr, y_t)
                loss.backward()
                optimizer.step()

            ctclm_model.eval()
            with torch.no_grad():
                dec_m6_tr = ctclm_model(seq_tr, times_tr, mask_tr).numpy()
                dec_m6_te = ctclm_model(seq_te, times_te, mask_te).numpy()
                p_m6 = torch.sigmoid(torch.tensor(dec_m6_te)).numpy()

        results["M6_CTCLM_Baseline"] = {"prob": p_m6, "decision": dec_m6_te, "train_decision": dec_m6_tr}

        # =============================================================
        # M7: CTCLM + Matched Classical Relational Kernel (Late Fusion)
        # =============================================================
        meta_features_tr_c = np.column_stack([dec_m6_tr, dec_m3_tr])
        meta_features_te_c = np.column_stack([dec_m6_te, dec_m3_te])
        meta_c = LogisticRegression(C=1.0, random_state=seed)
        meta_c.fit(meta_features_tr_c, y_tr)
        p_m7 = meta_c.predict_proba(meta_features_te_c)[:, 1]
        results["M7_CTCLM_Classical_Relational_Hybrid"] = {"prob": p_m7, "decision": meta_c.decision_function(meta_features_te_c)}

        # =============================================================
        # M8: CTCLM + Quantum Relational Kernel (Late Fusion)
        # =============================================================
        meta_features_tr_q = np.column_stack([dec_m6_tr, dec_m5_tr])
        meta_features_te_q = np.column_stack([dec_m6_te, dec_m5_te])
        meta_q = LogisticRegression(C=1.0, random_state=seed)
        meta_q.fit(meta_features_tr_q, y_tr)
        p_m8 = meta_q.predict_proba(meta_features_te_q)[:, 1]
        results["M8_CTCLM_Quantum_Relational_Hybrid"] = {"prob": p_m8, "decision": meta_q.decision_function(meta_features_te_q)}

        return results
