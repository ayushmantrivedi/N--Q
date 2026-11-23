"""
NĀḌĪ-Q Architecture A: End-to-End Pipeline Orchestrator.

Coordinates Plane 1 (Evidence Discovery) and Plane 2 (Clinical Inference).
The two planes are strictly decoupled: clinical output is 100% invariant to
the quantum service status or evidence graph content.

Usage
-----
pipeline = ArchitectureAPipeline()
report = pipeline.run_full(output_dir="experiments/architecture_a")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from nadiq.architecture_a.evidence_discovery import EvidenceDiscoveryPipeline, EvidenceGraphResult
from nadiq.architecture_a.evidence_exporter import EvidenceGraphExporter
from nadiq.architecture_a.clinical_predictor import (
    ArchitectureAClinicalPredictor,
    ClinicalPredictionResult,
)
from nadiq.experiments.phase14a.data_protocol import DataProtocol14A
from nadiq.models.metta_quantum.types import BiologicalQUBOConfig

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureAPipelineReport:
    """Full report from one Architecture A pipeline run."""

    # Timing
    total_runtime_ms: float = 0.0
    discovery_runtime_ms: float = 0.0
    prediction_runtime_ms: float = 0.0

    # Evidence discovery
    discovery_result: Optional[EvidenceGraphResult] = None
    evidence_json_path: Optional[str] = None
    evidence_metta_path: Optional[str] = None

    # Clinical prediction
    clinical_result: Optional[ClinicalPredictionResult] = None

    # Verification flags
    auroc_above_threshold: bool = False
    auroc_threshold: float = 0.635
    prediction_invariant: bool = True
    runtime_below_600ms: bool = False

    def to_dict(self) -> Dict[str, Any]:
        disc = self.discovery_result.to_dict() if self.discovery_result else {}
        clin = self.clinical_result.to_dict() if self.clinical_result else {}
        return {
            "total_runtime_ms": round(self.total_runtime_ms, 2),
            "discovery_runtime_ms": round(self.discovery_runtime_ms, 2),
            "prediction_runtime_ms": round(self.prediction_runtime_ms, 2),
            "evidence_discovery": disc,
            "clinical_prediction": clin,
            "verification": {
                "ensemble_auroc": clin.get("ensemble_auroc"),
                "auroc_above_threshold": self.auroc_above_threshold,
                "auroc_threshold": self.auroc_threshold,
                "prediction_invariant": self.prediction_invariant,
                "runtime_below_600ms_classical": self.runtime_below_600ms,
                "max_quantum_delta": clin.get("max_quantum_delta", 0.0),
            },
            "evidence_json_path": self.evidence_json_path,
            "evidence_metta_path": self.evidence_metta_path,
        }


class ArchitectureAPipeline:
    """
    Full Architecture A pipeline orchestrator.

    Parameters
    ----------
    task : str
        Clinical prediction task ('T2' default: Month-12→Month-60).
    seeds : list of int
        Seeds for multi-seed CV ensemble.
    n_folds : int
        CV folds.
    solver_mode : str
        QUBO solver mode ('auto', 'tabu', 'exact', 'qaoa_p1', 'qaoa_p2').
    budget_k : int
        Number of relationships to select.
    alpha_validated : float
        BH-FDR strict threshold.
    alpha_provisional : float
        BH-FDR lenient threshold.
    cohort_path, cache_path : str
        PPMI data paths.
    """

    def __init__(
        self,
        task: str = "T2",
        seeds: Optional[List[int]] = None,
        n_folds: int = 5,
        solver_mode: str = "auto",
        budget_k: int = 4,
        alpha_validated: float = 0.05,
        alpha_provisional: float = 0.10,
        min_selections_fallback: int = 2,
        qubo_config: Optional[BiologicalQUBOConfig] = None,
        qaoa_timeout_sec: float = 15.0,
        cohort_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        auroc_threshold: float = 0.635,
        discovery_runtime_budget_ms: float = 600.0,
    ) -> None:
        self.task = task
        self.seeds = seeds if seeds is not None else [42, 123, 456]
        self.n_folds = n_folds
        self.solver_mode = solver_mode
        self.budget_k = budget_k
        self.auroc_threshold = auroc_threshold
        self.discovery_runtime_budget_ms = discovery_runtime_budget_ms

        # Shared data protocol
        dp_kwargs: Dict[str, Any] = {}
        if cohort_path:
            dp_kwargs["cohort_path"] = cohort_path
        if cache_path:
            dp_kwargs["cache_path"] = cache_path
        self._data_protocol = DataProtocol14A(**dp_kwargs)

        # Plane 1
        cfg = qubo_config or BiologicalQUBOConfig(budget_k=budget_k)
        cfg.budget_k = budget_k
        self._discovery = EvidenceDiscoveryPipeline(
            solver_mode=solver_mode,
            budget_k=budget_k,
            alpha_validated=alpha_validated,
            alpha_provisional=alpha_provisional,
            min_selections_fallback=min_selections_fallback,
            qubo_config=cfg,
            qaoa_timeout_sec=qaoa_timeout_sec,
        )

        # Plane 2 (evidence_graph_path set after discovery)
        self._predictor_kwargs: Dict[str, Any] = {
            "task": task,
            "seeds": self.seeds,
            "n_folds": n_folds,
            **dp_kwargs,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full(
        self,
        output_dir: str = "experiments/architecture_a",
        dataset_label: str = "PPMI_N341",
    ) -> ArchitectureAPipelineReport:
        """
        Execute the full pipeline and write results to output_dir.

        Returns
        -------
        ArchitectureAPipelineReport
        """
        os.makedirs(output_dir, exist_ok=True)
        t_total_start = time.perf_counter()

        # ---- Plane 1: Evidence Discovery ----
        logger.info("=== Architecture A — Plane 1: Evidence Discovery ===")
        t_disc_start = time.perf_counter()

        X_raw, y, pats = self._data_protocol.get_task_data(task=self.task)
        # Use all data for global discovery (fold-contained discovery handled per-fold in CV)
        # For the pipeline-level export, run on the full cohort (non-leaked: correlation only)
        discovery_result = self._discovery.run(X_raw)

        disc_ms = (time.perf_counter() - t_disc_start) * 1000.0

        # Export evidence graph
        exporter = EvidenceGraphExporter(discovery_result, dataset_label=dataset_label)
        json_path = exporter.to_json(os.path.join(output_dir, "evidence_graph.json"))
        metta_path = exporter.to_metta_file(os.path.join(output_dir, "evidence_graph.metta"))

        logger.info(
            "Discovery complete: %d edges selected in %.1fms",
            len(discovery_result.selected_edges), disc_ms,
        )

        # ---- Plane 2: Clinical Prediction ----
        logger.info("=== Architecture A — Plane 2: Clinical Prediction ===")
        t_pred_start = time.perf_counter()

        predictor = ArchitectureAClinicalPredictor(
            evidence_graph_path=json_path,
            **self._predictor_kwargs,
        )
        clinical_result = predictor.predict()
        pred_ms = (time.perf_counter() - t_pred_start) * 1000.0

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        # ---- Verification ----
        auroc_ok = clinical_result.ensemble_auroc >= self.auroc_threshold
        inv_ok = clinical_result.max_quantum_delta < 1e-8
        disc_runtime_ok = disc_ms < self.discovery_runtime_budget_ms

        logger.info(
            "Pipeline complete: AUROC=%.4f (≥%.3f: %s)  Δ_max=%.2e (invariant: %s)  disc_ms=%.1f (<600ms: %s)",
            clinical_result.ensemble_auroc, self.auroc_threshold, "✓" if auroc_ok else "✗",
            clinical_result.max_quantum_delta, "✓" if inv_ok else "✗",
            disc_ms, "✓" if disc_runtime_ok else "✗",
        )

        report = ArchitectureAPipelineReport(
            total_runtime_ms=total_ms,
            discovery_runtime_ms=disc_ms,
            prediction_runtime_ms=pred_ms,
            discovery_result=discovery_result,
            evidence_json_path=json_path,
            evidence_metta_path=metta_path,
            clinical_result=clinical_result,
            auroc_above_threshold=auroc_ok,
            auroc_threshold=self.auroc_threshold,
            prediction_invariant=inv_ok,
            runtime_below_600ms=disc_runtime_ok,
        )

        # Write verdict JSON
        verdict_path = os.path.join(output_dir, "ARCHITECTURE_A_VERDICT.json")
        with open(verdict_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Verdict written → %s", verdict_path)

        return report

    # ------------------------------------------------------------------
    # Quick prediction only (Plane 2 standalone)
    # ------------------------------------------------------------------

    def run_prediction_only(
        self,
        evidence_graph_path: Optional[str] = None,
        output_dir: str = "experiments/architecture_a",
    ) -> ClinicalPredictionResult:
        """
        Run Plane 2 (clinical inference) standalone without redoing discovery.

        evidence_graph_path is used for logging display only.
        """
        os.makedirs(output_dir, exist_ok=True)
        predictor = ArchitectureAClinicalPredictor(
            evidence_graph_path=evidence_graph_path,
            **self._predictor_kwargs,
        )
        result = predictor.predict()

        # Save result
        out_path = os.path.join(output_dir, "CLINICAL_PREDICTION_RESULT.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.info("Clinical result written → %s", out_path)
        return result
