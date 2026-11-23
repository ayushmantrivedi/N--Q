"""
NĀḌĪ-Q Architecture A: Evidence Graph Exporter.

Serializes the selected biological relationship graph to:
  1. JSON  — machine-readable with full provenance metadata.
  2. MeTTa AtomSpace — Hyperon-compatible triplet syntax.
  3. Summary dict — lightweight Python representation.

All exported graphs carry explicit 'Provisional/Explanatory' labels.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nadiq.architecture_a.evidence_discovery import EvidenceGraphResult

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "PROVISIONAL/EXPLANATORY ONLY. These relationships are discovered from "
    "PPMI 1-year multimodal correlations with an estimated replication rate of "
    "~6–7% across 15 folds. They are hypothesis-generating candidate relationships, "
    "NOT definitive causal biomarkers. Do not use for clinical decision-making."
)


class EvidenceGraphExporter:
    """
    Serializes an EvidenceGraphResult into portable, labeled formats.

    Parameters
    ----------
    result : EvidenceGraphResult
        The pipeline output from EvidenceDiscoveryPipeline.run().
    dataset_label : str
        Human-readable dataset label (e.g. 'PPMI_N341').
    """

    def __init__(
        self,
        result: EvidenceGraphResult,
        dataset_label: str = "PPMI",
    ) -> None:
        self.result = result
        self.dataset_label = dataset_label

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_json(self, output_path: str) -> str:
        """
        Write full evidence graph to a JSON file.

        Parameters
        ----------
        output_path : str
            Target file path (e.g. 'results/evidence_graph.json').

        Returns
        -------
        str
            Absolute path of the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        edges = []
        for e in self.result.selected_edges:
            edges.append({
                "id": e.id,
                "src": e.src,
                "dst": e.dst,
                "rel_type": e.rel_type,
                "direction": e.direction,
                "effect_r": round(e.effect_r, 6),
                "p_val": round(e.p_val, 6),
                "se": round(e.se, 6),
                "lag_years": e.lag_years,
                "evidence_n": e.evidence_n,
                "status": e.status,
                "utility": round(e.utility, 6),
                "statistical_quality": "Provisional/Explanatory",
            })

        solver_meta: Dict[str, Any] = {}
        if self.result.solver_output is not None:
            so = self.result.solver_output
            solver_meta = {
                "method": so.method,
                "energy": round(so.energy, 6),
                "utility_score": round(so.utility_score, 6),
                "is_feasible": so.is_feasible,
                "optimality_gap": round(so.optimality_gap, 6),
                "runtime_ms": round(so.runtime_sec * 1000, 2),
                "n_evaluations": so.n_evaluations,
            }

        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset_label,
            "disclaimer": _DISCLAIMER,
            "statistical_quality": "Provisional/Explanatory",
            "replication_note": self.result.replication_note,
            "discovery_summary": {
                "n_selected": len(edges),
                "n_candidates_total": self.result.n_candidates_total,
                "n_validated": self.result.n_candidates_validated,
                "n_provisional": self.result.n_candidates_provisional,
                "n_fallback": self.result.n_candidates_fallback,
                "fdr_fallback_triggered": self.result.fdr_fallback_triggered,
                "solver_mode_used": self.result.solver_mode_used,
                "discovery_runtime_ms": round(self.result.discovery_runtime_ms, 2),
            },
            "power_analysis": self.result.power_analysis,
            "solver_provenance": solver_meta,
            "edges": edges,
            "metta_triplets": self._metta_triplets(),
        }

        abs_path = os.path.abspath(output_path)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("EvidenceGraphExporter: wrote JSON → %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # MeTTa AtomSpace export
    # ------------------------------------------------------------------

    def _metta_triplets(self) -> List[str]:
        """Generate MeTTa-compatible relation triplets for selected edges."""
        triplets = []
        for e in self.result.selected_edges:
            direction_str = "positive" if e.direction >= 0 else "negative"
            triplets.append(
                f"(Relationship {e.src} {e.dst} "
                f"(EffectR {e.effect_r:.4f}) "
                f"(PVal {e.p_val:.4f}) "
                f"(Status {e.status}) "
                f"(Direction {direction_str}) "
                f"(Quality Provisional/Explanatory))"
            )
        return triplets

    def to_metta_atomspace(self, metta_instance: Any) -> int:
        """
        Register selected edges in a live Hyperon MeTTa AtomSpace.

        Parameters
        ----------
        metta_instance : hyperon.MeTTa
            An active MeTTa runtime instance.

        Returns
        -------
        int
            Number of atoms registered.
        """
        n_atoms = 0
        for triplet in self._metta_triplets():
            try:
                metta_instance.run(f"!(add-atom &self {triplet})")
                n_atoms += 1
            except Exception as exc:
                logger.warning("Failed to add MeTTa atom: %s — %s", triplet[:60], exc)
        logger.info("EvidenceGraphExporter: registered %d MeTTa atoms", n_atoms)
        return n_atoms

    def to_metta_file(self, output_path: str) -> str:
        """Write MeTTa triplets to a .metta text file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        abs_path = os.path.abspath(output_path)
        lines = [
            "; NĀḌĪ-Q Architecture A — Evidence Graph (Provisional/Explanatory)",
            f"; Generated: {datetime.now(timezone.utc).isoformat()}",
            f"; Dataset: {self.dataset_label}",
            f"; Disclaimer: {_DISCLAIMER}",
            "",
        ] + self._metta_triplets()
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.info("EvidenceGraphExporter: wrote MeTTa file → %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a compact machine-readable summary."""
        return {
            "n_selected": len(self.result.selected_edges),
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "effect_r": round(e.effect_r, 4),
                    "p_val": round(e.p_val, 4),
                    "status": e.status,
                    "utility": round(e.utility, 4),
                }
                for e in self.result.selected_edges
            ],
            "solver": self.result.solver_mode_used,
            "fdr_fallback": self.result.fdr_fallback_triggered,
            "statistical_quality": "Provisional/Explanatory",
            "runtime_ms": round(self.result.discovery_runtime_ms, 2),
        }
