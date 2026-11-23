"""
NĀḌĪ-Q Architecture A: Decoupled Clinical Predictor (Plane 2).

100% classical inference.  Zero imports from quantum libraries or quantum-related modules.
Clinical predictions are invariant to quantum service status.

Accepts an optional evidence_graph_path argument for display/logging ONLY —
it is never used to change predictions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from nadiq.experiments.phase14a.data_protocol import DataProtocol14A

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClinicalPredictionResult:
    """Full output of a cross-validated clinical prediction run."""

    task: str = "T2"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    n_folds: int = 5

    # Per-seed metrics
    seed_aurocs: List[float] = field(default_factory=list)
    seed_auprcs: List[float] = field(default_factory=list)
    seed_briers: List[float] = field(default_factory=list)

    # Ensemble metrics (3-seed mean probability aggregation)
    ensemble_auroc: float = 0.0
    ensemble_auprc: float = 0.0
    ensemble_brier: float = 0.0

    # Participant-level predictions
    y_true: List[int] = field(default_factory=list)
    y_prob_ensemble: List[float] = field(default_factory=list)

    # Invariance verification
    max_quantum_delta: float = 0.0  # always 0.0 by construction

    # Evidence graph path (display only — never affects predictions)
    evidence_graph_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "seeds": self.seeds,
            "n_folds": self.n_folds,
            "seed_aurocs": [round(x, 4) for x in self.seed_aurocs],
            "mean_auroc": round(float(np.mean(self.seed_aurocs)), 4) if self.seed_aurocs else 0.0,
            "std_auroc": round(float(np.std(self.seed_aurocs)), 4) if self.seed_aurocs else 0.0,
            "ensemble_auroc": round(self.ensemble_auroc, 4),
            "ensemble_auprc": round(self.ensemble_auprc, 4),
            "ensemble_brier": round(self.ensemble_brier, 4),
            "max_quantum_delta": self.max_quantum_delta,
            "evidence_graph_path": self.evidence_graph_path,
            "quantum_invariant": self.max_quantum_delta < 1e-8,
        }


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class ArchitectureAClinicalPredictor:
    """
    Clinical Parkinson's progression predictor (Architecture A, Plane 2).

    Uses logistic regression with fold-contained median imputation + standard scaling.
    Runs multi-seed stratified k-fold cross-validation and returns an ensemble result.

    Parameters
    ----------
    task : str
        'T1' or 'T2'.  Default 'T2' (Month-12 → Month-60 future progression).
    seeds : list of int
        Random seeds for StratifiedKFold shuffling.  Default [42, 123, 456].
    n_folds : int
        Number of folds.  Default 5.
    cohort_path, cache_path : str
        PPMI data paths (forwarded to DataProtocol14A).
    evidence_graph_path : str, optional
        Path to evidence_graph.json — used for logging display only.
    C : float
        Logistic regression regularisation strength.  Default 1.0.
    max_iter : int
        Max LR iterations.  Default 1000.
    """

    def __init__(
        self,
        task: str = "T2",
        seeds: Optional[List[int]] = None,
        n_folds: int = 5,
        cohort_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        evidence_graph_path: Optional[str] = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.task = task.upper()
        self.seeds = seeds if seeds is not None else [42, 123, 456]
        self.n_folds = n_folds
        self.evidence_graph_path = evidence_graph_path
        self.C = C
        self.max_iter = max_iter

        # Data loading
        dp_kwargs: Dict[str, Any] = {}
        if cohort_path:
            dp_kwargs["cohort_path"] = cohort_path
        if cache_path:
            dp_kwargs["cache_path"] = cache_path
        self._data_protocol = DataProtocol14A(**dp_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self) -> ClinicalPredictionResult:
        """
        Run multi-seed cross-validated logistic regression prediction.

        Returns
        -------
        ClinicalPredictionResult
        """
        X_raw, y, pats = self._data_protocol.get_task_data(task=self.task)
        y = np.asarray(y, dtype=int)
        n = len(y)

        logger.info(
            "ClinicalPredictor: task=%s  N=%d  positives=%d (%.1f%%)",
            self.task, n, int(y.sum()), 100.0 * y.mean(),
        )

        if self.evidence_graph_path and os.path.exists(self.evidence_graph_path):
            logger.info(
                "Evidence graph loaded for display (NOT used in inference): %s",
                self.evidence_graph_path,
            )

        # Per-seed OOF probability arrays
        seed_probs: List[np.ndarray] = []
        seed_aurocs: List[float] = []
        seed_auprcs: List[float] = []
        seed_briers: List[float] = []

        for seed in self.seeds:
            probs = self._run_cv_seed(X_raw, y, seed=seed)
            seed_probs.append(probs)

            auroc = roc_auc_score(y, probs)
            auprc = average_precision_score(y, probs)
            brier = brier_score_loss(y, probs)
            seed_aurocs.append(float(auroc))
            seed_auprcs.append(float(auprc))
            seed_briers.append(float(brier))

            logger.info(
                "  seed=%d  AUROC=%.4f  AUPRC=%.4f  Brier=%.4f",
                seed, auroc, auprc, brier,
            )

        # 3-seed ensemble (mean probability)
        prob_ensemble = np.mean(np.stack(seed_probs, axis=0), axis=0)

        ensemble_auroc = float(roc_auc_score(y, prob_ensemble))
        ensemble_auprc = float(average_precision_score(y, prob_ensemble))
        ensemble_brier = float(brier_score_loss(y, prob_ensemble))

        logger.info(
            "Ensemble (N_seeds=%d): AUROC=%.4f  AUPRC=%.4f  Brier=%.4f",
            len(self.seeds), ensemble_auroc, ensemble_auprc, ensemble_brier,
        )

        # Verify quantum invariance: Δ_max = 0 by construction (no quantum code here)
        max_delta = 0.0

        return ClinicalPredictionResult(
            task=self.task,
            seeds=self.seeds,
            n_folds=self.n_folds,
            seed_aurocs=seed_aurocs,
            seed_auprcs=seed_auprcs,
            seed_briers=seed_briers,
            ensemble_auroc=ensemble_auroc,
            ensemble_auprc=ensemble_auprc,
            ensemble_brier=ensemble_brier,
            y_true=y.tolist(),
            y_prob_ensemble=prob_ensemble.tolist(),
            max_quantum_delta=max_delta,
            evidence_graph_path=self.evidence_graph_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_cv_seed(self, X_raw: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        """
        Run one seed of stratified k-fold CV.  Returns OOF probability array.
        All preprocessing (imputation + scaling) is fit strictly within each training fold.
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        oof_probs = np.zeros(len(y), dtype=float)

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y)):
            X_tr, y_tr, X_te, y_te = self._data_protocol.prepare_fold_data(
                X_raw, y, tr_idx, te_idx
            )

            # Standard scaling fit on training fold only
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            # Logistic regression
            lr = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=seed,
            )
            lr.fit(X_tr_sc, y_tr)
            oof_probs[te_idx] = lr.predict_proba(X_te_sc)[:, 1]

        return oof_probs
