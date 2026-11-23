"""
NĀḌĪ-Q Phase 13: MeTTa-Q Types and Data Contracts.
Defines candidate relationships, QUBO configurations, and optimizer output records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateRelationship:
    """Represents a discovered candidate biological relationship or temporal motif."""
    id: int
    src: str
    dst: str
    rel_type: str
    direction: int
    effect_r: float
    p_val: float
    se: float
    lag_years: float
    evidence_n: int
    status: str = "Candidate"  # "Validated", "Provisional", "Candidate"
    utility: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "rel_type": self.rel_type,
            "direction": self.direction,
            "effect_r": self.effect_r,
            "p_val": self.p_val,
            "se": self.se,
            "lag_years": self.lag_years,
            "evidence_n": self.evidence_n,
            "status": self.status,
            "utility": self.utility,
        }


@dataclass
class BiologicalQUBOConfig:
    """Configuration for QUBO penalization terms and selection budget."""
    lambda_redundancy: float = 0.50
    mu_contradiction: float = 2.00
    gamma_budget: float = 1.50
    budget_k: int = 4
    min_utility: float = 0.01


@dataclass
class QuantumSelectionOutput:
    """Standardized output across QAOA and classical control solvers."""
    method: str
    bitstring: str
    selected_indices: List[int]
    selected_edges: List[CandidateRelationship]
    energy: float  # QUBO minimization energy: H(x) = -Obj_max(x)
    utility_score: float  # Original maximization objective: Obj_max(x)
    is_feasible: bool  # Matches budget K and violates 0 hard contradictions
    budget_error: int  # |sum(x) - K|
    num_contradictions: int
    optimality_gap: float  # Energy difference from exact global minimum
    probability: float  # State probability (for QAOA) or 1.0 (for deterministic)
    circuit_depth: int
    gate_count: Dict[str, int]
    n_evaluations: int
    runtime_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "bitstring": self.bitstring,
            "selected_indices": self.selected_indices,
            "num_selected": len(self.selected_indices),
            "energy": self.energy,
            "utility_score": self.utility_score,
            "is_feasible": self.is_feasible,
            "budget_error": self.budget_error,
            "num_contradictions": self.num_contradictions,
            "optimality_gap": self.optimality_gap,
            "probability": self.probability,
            "circuit_depth": self.circuit_depth,
            "gate_count": self.gate_count,
            "n_evaluations": self.n_evaluations,
            "runtime_sec": self.runtime_sec,
        }
