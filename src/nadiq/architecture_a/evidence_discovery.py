"""
NĀḌĪ-Q Architecture A: Biological Evidence Discovery Pipeline.

Staged pipeline:
  Statistical FDR (BH q≤0.05/0.10)
  → MeTTa Compiler (symbolic constraints, redundancy, contradiction detection)
  → QUBO Builder
  → DualSolverEngine (Tabu primary / QAOA async sidecar)
  → EvidenceGraphResult

All preprocessing (correlation estimation) must be called with fold-contained data
externally; this module only orchestrates the already-computed candidates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

from nadiq.architecture_a.fdr import (
    adaptive_fdr_tier,
    screen_candidates_fdr,
    power_analysis_summary,
)
from nadiq.architecture_a.optimization_engine import DualSolverEngine
from nadiq.models.metta_quantum.compiler import MeTTaGraphCompiler
from nadiq.models.metta_quantum.types import (
    BiologicalQUBOConfig,
    CandidateRelationship,
    QuantumSelectionOutput,
)

logger = logging.getLogger(__name__)

# Feature column names (matches DataProtocol14A)
FEATURE_NAMES = [
    "asyn",
    "ptau",
    "caudate",
    "putamen",
    "motor_total",
    "motor_subscore",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvidenceGraphResult:
    """Full output of one evidence-discovery run."""

    # Selected relationships
    selected_edges: List[CandidateRelationship] = field(default_factory=list)

    # All candidates before QUBO (post-FDR)
    all_candidates: List[CandidateRelationship] = field(default_factory=list)

    # Solver result
    solver_output: Optional[QuantumSelectionOutput] = None

    # Provenance
    solver_mode_used: str = "none"
    n_candidates_total: int = 0
    n_candidates_validated: int = 0
    n_candidates_provisional: int = 0
    n_candidates_fallback: int = 0
    fdr_fallback_triggered: bool = False
    power_analysis: Dict[str, float] = field(default_factory=dict)

    # Statistical quality labels
    statistical_quality: str = "Provisional/Explanatory"
    replication_note: str = (
        "Evidence replication rate on 1-year PPMI deltas is ~6–7%. "
        "Relationships are hypothesis-generating, not definitive causal biomarkers."
    )

    # Timing
    discovery_runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_edges": [e.to_dict() for e in self.selected_edges],
            "n_selected": len(self.selected_edges),
            "solver_mode_used": self.solver_mode_used,
            "n_candidates_total": self.n_candidates_total,
            "n_candidates_validated": self.n_candidates_validated,
            "n_candidates_provisional": self.n_candidates_provisional,
            "n_candidates_fallback": self.n_candidates_fallback,
            "fdr_fallback_triggered": self.fdr_fallback_triggered,
            "power_analysis": self.power_analysis,
            "statistical_quality": self.statistical_quality,
            "replication_note": self.replication_note,
            "discovery_runtime_ms": self.discovery_runtime_ms,
            "solver_output": self.solver_output.to_dict() if self.solver_output else None,
        }


# ---------------------------------------------------------------------------
# Candidate discovery from raw multimodal data
# ---------------------------------------------------------------------------

def _compute_pairwise_correlations(
    X: np.ndarray,
    feature_names: List[str],
    min_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise Pearson correlations from a (N, F) feature matrix.

    Uses only complete cases per pair. Returns raw candidate list.
    """
    n_feat = X.shape[1]
    candidates = []
    idx = 0
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            xi = X[:, i]
            xj = X[:, j]
            # Complete cases
            mask = ~(np.isnan(xi) | np.isnan(xj))
            n_cc = int(np.sum(mask))
            if n_cc < min_n:
                continue
            r, p = scipy_stats.pearsonr(xi[mask], xj[mask])
            se = float(1.0 / np.sqrt(max(n_cc - 3, 1)))
            candidates.append({
                "id": idx,
                "src": feature_names[i],
                "dst": feature_names[j],
                "r": float(r),
                "p": float(p),
                "se": se,
                "n": n_cc,
                "lag": 1.0,
                "rel_type": f"{feature_names[i]}_to_{feature_names[j]}",
                "direction": 1 if r >= 0 else -1,
            })
            idx += 1
    return candidates


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class EvidenceDiscoveryPipeline:
    """
    End-to-end biological relationship discovery pipeline (Plane 1 of Architecture A).

    Usage
    -----
    pipeline = EvidenceDiscoveryPipeline(solver_mode='tabu', budget_k=4)
    result = pipeline.run(X_train, feature_names=FEATURE_NAMES)
    """

    def __init__(
        self,
        solver_mode: str = "auto",
        budget_k: int = 4,
        alpha_validated: float = 0.05,
        alpha_provisional: float = 0.10,
        min_selections_fallback: int = 2,
        qubo_config: Optional[BiologicalQUBOConfig] = None,
        qaoa_timeout_sec: float = 15.0,
        min_n_per_pair: int = 20,
    ) -> None:
        self.solver_mode = solver_mode
        self.budget_k = budget_k
        self.alpha_validated = alpha_validated
        self.alpha_provisional = alpha_provisional
        self.min_selections_fallback = min_selections_fallback
        self.min_n_per_pair = min_n_per_pair
        self.qaoa_timeout_sec = qaoa_timeout_sec

        cfg = qubo_config or BiologicalQUBOConfig(budget_k=budget_k)
        cfg.budget_k = budget_k
        self.qubo_config = cfg

        self._compiler = MeTTaGraphCompiler(r_thresh=0.05, p_thresh=alpha_provisional)
        self._engine = DualSolverEngine(
            qubo_config=self.qubo_config,
            solver_mode=solver_mode,
            qaoa_timeout_sec=qaoa_timeout_sec,
        )

    def run(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        collinear_redundancies: Optional[List[Tuple[int, int, float]]] = None,
        incompatibility_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> EvidenceGraphResult:
        """
        Run the full discovery pipeline on a training-fold feature matrix.

        Parameters
        ----------
        X_train : np.ndarray of shape (N_train, F)
            Training features (may contain NaNs; complete-case pairs used).
        feature_names : list of str, optional
            Column names; defaults to FEATURE_NAMES.
        collinear_redundancies, incompatibility_pairs :
            Optional expert constraints forwarded to MeTTaGraphCompiler.

        Returns
        -------
        EvidenceGraphResult
        """
        t0 = time.perf_counter()
        feat_names = feature_names or FEATURE_NAMES
        n_train = X_train.shape[0]

        # --- Stage 1: Compute pairwise correlations ---
        raw_candidates = _compute_pairwise_correlations(
            X_train, feat_names, min_n=self.min_n_per_pair
        )
        logger.info("Discovery: %d raw candidate pairs from N_train=%d", len(raw_candidates), n_train)

        # --- Stage 2: FDR screening ---
        screened, tier_map = screen_candidates_fdr(
            raw_candidates,
            alpha_validated=self.alpha_validated,
            alpha_provisional=self.alpha_provisional,
            min_selections_fallback=self.min_selections_fallback,
        )
        power_info = {}
        try:
            power_info = power_analysis_summary(n_train)
        except Exception:
            pass

        fdr_fallback = any(t == "Candidate (Fallback)" for t in tier_map.values())
        n_validated = sum(1 for t in tier_map.values() if t == "Validated")
        n_provisional = sum(1 for t in tier_map.values() if t == "Provisional")
        n_fallback = sum(1 for t in tier_map.values() if t == "Candidate (Fallback)")

        # Enrich raw candidates with FDR tier before compiler
        enriched = []
        for c in screened:
            c2 = dict(c)
            # Map fdr_tier into status field for compiler
            tier = c2.get("fdr_tier", "Candidate")
            if tier in {"Validated", "Provisional"}:
                c2["status_hint"] = tier
            else:
                c2["status_hint"] = "Candidate"
            enriched.append(c2)

        # --- Stage 3: MeTTa compiler ---
        try:
            compiled_candidates, redundancy_map, contradiction_pairs_set = (
                self._compiler.compile_candidates(
                    enriched,
                    collinear_redundancies=collinear_redundancies,
                    incompatibility_pairs=incompatibility_pairs,
                )
            )
        except Exception as exc:
            logger.error("MeTTaGraphCompiler failed: %s. Using raw candidates.", exc)
            compiled_candidates = [
                CandidateRelationship(
                    id=c["id"],
                    src=c["src"],
                    dst=c["dst"],
                    rel_type=c.get("rel_type", f"{c['src']}_{c['dst']}"),
                    direction=c.get("direction", 1),
                    effect_r=c["r"],
                    p_val=c["p"],
                    se=c.get("se", 0.1),
                    lag_years=c.get("lag", 1.0),
                    evidence_n=c.get("n", n_train),
                    status=c.get("status_hint", "Candidate"),
                    utility=abs(c["r"]),
                )
                for c in enriched
            ]
            redundancy_map = {}
            contradiction_pairs_set = set()

        logger.info(
            "MeTTa compiler: %d candidates, %d redundancy pairs, %d contradiction pairs",
            len(compiled_candidates),
            len(redundancy_map),
            len(contradiction_pairs_set),
        )

        # --- Stage 4: QUBO + Solver ---
        if compiled_candidates:
            solver_out = self._engine.solve(
                compiled_candidates, redundancy_map, contradiction_pairs_set
            )
            selected_edges = solver_out.selected_edges
            solver_mode_used = self._engine.last_solver_used
        else:
            solver_out = None
            selected_edges = []
            solver_mode_used = "none (no candidates)"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = EvidenceGraphResult(
            selected_edges=selected_edges,
            all_candidates=compiled_candidates,
            solver_output=solver_out,
            solver_mode_used=solver_mode_used,
            n_candidates_total=len(compiled_candidates),
            n_candidates_validated=n_validated,
            n_candidates_provisional=n_provisional,
            n_candidates_fallback=n_fallback,
            fdr_fallback_triggered=fdr_fallback,
            power_analysis=power_info,
            discovery_runtime_ms=elapsed_ms,
        )

        logger.info(
            "EvidenceDiscoveryPipeline complete: %d selected, solver=%s, %.1fms",
            len(selected_edges),
            solver_mode_used,
            elapsed_ms,
        )
        return result
