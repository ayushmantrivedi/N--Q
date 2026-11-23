"""
NĀḌĪ-Q Architecture A: Quantum-Assisted Evidence Discovery.

Plane 1 — Biological Evidence Discovery (MeTTa + QUBO + Classical Optimization).
Plane 2 — Clinical Inference (100% classical, zero quantum coupling).

Both planes are strictly decoupled: clinical predictions are invariant to quantum service status.
"""

from nadiq.architecture_a.fdr import (
    benjamini_hochberg_fdr,
    adaptive_fdr_tier,
    minimum_detectable_effect,
)
from nadiq.architecture_a.optimization_engine import DualSolverEngine
from nadiq.architecture_a.evidence_discovery import EvidenceDiscoveryPipeline, EvidenceGraphResult
from nadiq.architecture_a.evidence_exporter import EvidenceGraphExporter
from nadiq.architecture_a.clinical_predictor import ArchitectureAClinicalPredictor
from nadiq.architecture_a.pipeline import ArchitectureAPipeline

__all__ = [
    "benjamini_hochberg_fdr",
    "adaptive_fdr_tier",
    "minimum_detectable_effect",
    "DualSolverEngine",
    "EvidenceDiscoveryPipeline",
    "EvidenceGraphResult",
    "EvidenceGraphExporter",
    "ArchitectureAClinicalPredictor",
    "ArchitectureAPipeline",
]
