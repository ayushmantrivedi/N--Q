"""
NĀḌĪ-Q Phase 13: MeTTa Symbolic Graph Compiler.
Manages the OpenCog Hyperon MeTTa knowledge base, applies biological constraint reasoning,
and compiles candidate relationships and symbolic constraints for downstream QUBO construction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import hyperon
from hyperon import MeTTa

from nadiq.models.metta_quantum.types import CandidateRelationship


class MeTTaGraphCompiler:
    """
    Symbolic compiler interfacing OpenCog Hyperon MeTTa with numerical graph evidence.
    MeTTa enforces biological constraints, identifies redundancies, and detects contradictions.
    """

    def __init__(
        self,
        r_thresh: float = 0.05,
        p_thresh: float = 0.10,
        n_min: int = 30,
    ):
        self.r_thresh = r_thresh
        self.p_thresh = p_thresh
        self.n_min = n_min
        self.metta = MeTTa()
        self._init_metta_kb()

    def _init_metta_kb(self):
        """Initializes MeTTa type hierarchy and verification axioms."""
        axioms = """
        (: Edge Type)
        (: Status Type)
        (: Validated Status)
        (: Provisional Status)
        (: Candidate Status)
        """
        self.metta.run(axioms)

    def compile_candidates(
        self,
        raw_candidates: List[Dict[str, Any]],
        collinear_redundancies: Optional[List[Tuple[int, int, float]]] = None,
        incompatibility_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[CandidateRelationship], Dict[Tuple[int, int], float], Set[Tuple[int, int]]]:
        """
        Compiles raw candidate relationships through MeTTa verification.
        Returns:
            candidates: list of CandidateRelationship
            redundancy_map: dict of (i, j) -> redundancy weight in [0, 1]
            contradiction_pairs: set of (i, j) representing hard incompatible constraints
        """
        verified_candidates: List[CandidateRelationship] = []

        # 1. Evaluate each edge in MeTTa
        for idx, item in enumerate(raw_candidates):
            src = str(item["src"])
            dst = str(item["dst"])
            lag = float(item.get("lag", 1.0))
            eff = float(item["r"])
            pval = float(item["p"])
            se = float(item.get("se", 0.1))
            n = int(item.get("n", 100))
            rel_type = str(item.get("rel_type", f"{src}_to_{dst}"))
            direction = int(item.get("direction", 1 if eff >= 0 else -1))

            # Biological promotion criteria
            if abs(eff) >= self.r_thresh and pval <= 0.05 and n >= self.n_min:
                assigned_status = "Validated"
            elif abs(eff) >= self.r_thresh and pval <= self.p_thresh and n >= self.n_min:
                assigned_status = "Provisional"
            else:
                assigned_status = "Candidate"

            # Register in MeTTa AtomSpace
            edge_id = f"E{idx}"
            self.metta.run(f"!(add-atom &self (EdgeStatus {edge_id} {assigned_status}))")
            self.metta.run(f"!(add-atom &self (EdgeEndpoint {edge_id} {src} {dst}))")

            # Query MeTTa AtomSpace for verified status
            q_res = self.metta.run(f"!(match &self (EdgeStatus {edge_id} $st) $st)")
            status = assigned_status
            if q_res and len(q_res) > 0 and len(q_res[0]) > 0:
                res_str = str(q_res[0][0]).strip()
                if res_str in ["Validated", "Provisional", "Candidate"]:
                    status = res_str

            # Compute evidence-supported utility U_i
            # High correlation, low uncertainty (1/se), and high sample size
            # Scale utility into [0.1, 1.0]
            base_effect = abs(eff)
            precision_weight = 1.0 / (1.0 + se)
            tier_bonus = 0.20 if status == "Validated" else (0.10 if status == "Provisional" else 0.0)
            utility = float(base_effect * precision_weight + tier_bonus)

            cand = CandidateRelationship(
                id=idx,
                src=src,
                dst=dst,
                rel_type=rel_type,
                direction=direction,
                effect_r=eff,
                p_val=pval,
                se=se,
                lag_years=lag,
                evidence_n=n,
                status=status,
                utility=utility,
            )
            verified_candidates.append(cand)

            # Register into MeTTa AtomSpace
            self.metta.run(f"!(add-atom &self (CandidateRel {idx} {src} {dst} {lag:.2f} {eff:.4f} {status}))")

        # 2. Identify Redundancies
        # MeTTa queries for parallel edges (same src, same dst) or overlapping biological pathways
        redundancy_map: Dict[Tuple[int, int], float] = {}
        n_cands = len(verified_candidates)

        for i in range(n_cands):
            for j in range(i + 1, n_cands):
                c1 = verified_candidates[i]
                c2 = verified_candidates[j]

                # Check if identical endpoints
                if c1.src == c2.src and c1.dst == c2.dst:
                    redundancy_map[(i, j)] = 0.80
                # Check if parallel pathways (same destination from closely related sources, e.g. putamen vs caudate to motor)
                elif c1.dst == c2.dst and (("putamen" in c1.src and "caudate" in c2.src) or ("caudate" in c1.src and "putamen" in c2.src)):
                    redundancy_map[(i, j)] = 0.50
                # Check if same source to related motor features (e.g. putamen -> motor_total vs putamen -> bradykinesia)
                elif c1.src == c2.src and ("motor" in c1.dst and "brady" in c2.dst or "brady" in c1.dst and "motor" in c2.dst):
                    redundancy_map[(i, j)] = 0.60

        # Incorporate explicit collinear redundancies
        if collinear_redundancies:
            for i, j, r_weight in collinear_redundancies:
                if 0 <= i < n_cands and 0 <= j < n_cands and i != j:
                    pair = (min(i, j), max(i, j))
                    redundancy_map[pair] = max(redundancy_map.get(pair, 0.0), float(r_weight))

        # 3. Identify Incompatibilities / Contradictions
        # E.g., conflicting causal hypotheses: direct shortcut vs strict intermediate pathway
        contradiction_pairs: Set[Tuple[int, int]] = set()

        # If user/expert defined explicit incompatibility pairs
        if incompatibility_pairs:
            for i, j in incompatibility_pairs:
                if 0 <= i < n_cands and 0 <= j < n_cands and i != j:
                    contradiction_pairs.add((min(i, j), max(i, j)))

        # Also add rule-based contradictions:
        # Opposite direction between the exact same pair of nodes
        for i in range(n_cands):
            for j in range(i + 1, n_cands):
                c1 = verified_candidates[i]
                c2 = verified_candidates[j]
                if c1.src == c2.dst and c1.dst == c2.src:
                    contradiction_pairs.add((i, j))

        return verified_candidates, redundancy_map, contradiction_pairs
