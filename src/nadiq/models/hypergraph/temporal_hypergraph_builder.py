"""
NĀḌĪ-Q Phase 10A: Temporal Hypergraph Builder.
Constructs time-stamped, multimodal hypergraph states for longitudinal patient trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class HypergraphNode:
    """
    Represents an individual modality node within a patient's visit hypergraph.
    """
    node_id: str
    patient_id: int
    visit: str
    modality: str  # 'molecular', 'structural', 'functional', 'temporal'
    features: torch.Tensor  # Representation vector
    timestamp_months: float  # Elapsed time in months from baseline
    observation_age: float   # Patient chronological age at visit
    available: bool = True
    quality_score: float = 1.0


@dataclass
class Hyperedge:
    """
    Represents a multi-node hyperedge connecting 2 or more related nodes.
    """
    edge_id: str
    edge_type: str  # 'intra_molecular', 'intra_structural', 'intra_functional', 'cross_modal_visit', 'temporal_transition'
    patient_id: int
    visit: str
    node_ids: List[str]
    elapsed_time: float = 0.0
    representation: Optional[torch.Tensor] = None
    quantum_conditioned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisitHypergraph:
    """
    Hypergraph snapshot for a single patient visit.
    """
    patient_id: int
    visit: str
    timestamp_months: float
    observation_age: float
    nodes: Dict[str, HypergraphNode] = field(default_factory=dict)
    hyperedges: Dict[str, Hyperedge] = field(default_factory=dict)
    availability: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PatientLongitudinalHypergraph:
    """
    Complete temporal hypergraph sequence for an individual patient across all visits.
    """
    patient_id: int
    visits: List[str]
    snapshots: Dict[str, VisitHypergraph] = field(default_factory=dict)
    temporal_edges: List[Hyperedge] = field(default_factory=list)
    progression_target_5yr: Optional[int] = None
    delta_np3tot_5yr: Optional[float] = None


class TemporalHypergraphBuilder:
    """
    Factory for building verified, patient-isolated temporal hypergraphs from multimodal data.
    """

    def __init__(
        self,
        dim_mol: int = 15,
        dim_struct: int = 36,
        dim_func: int = 35,
        alignment_tolerance_months: float = 3.0
    ):
        self.dim_mol = dim_mol
        self.dim_struct = dim_struct
        self.dim_func = dim_func
        self.alignment_tolerance_months = alignment_tolerance_months

    def build_patient_hypergraph(
        self,
        patient_id: int,
        visit_records: List[Dict[str, Any]],
        progression_5yr: Optional[int] = None,
        delta_5yr: Optional[float] = None
    ) -> PatientLongitudinalHypergraph:
        """
        Builds a multi-visit temporal hypergraph for a single patient.
        Ensures strict participant isolation (no cross-patient edges).
        """
        # Sort visits by timestamp / month
        sorted_records = sorted(visit_records, key=lambda r: r.get("timestamp_months", 0.0))
        
        visits = [r["visit"] for r in sorted_records]
        patient_hg = PatientLongitudinalHypergraph(
            patient_id=patient_id,
            visits=visits,
            progression_target_5yr=progression_5yr,
            delta_np3tot_5yr=delta_5yr
        )

        prev_snapshot: Optional[VisitHypergraph] = None

        for rec in sorted_records:
            v_name = rec["visit"]
            t_months = float(rec.get("timestamp_months", 0.0))
            age = float(rec.get("age_at_visit", 60.0))

            snap = VisitHypergraph(
                patient_id=patient_id,
                visit=v_name,
                timestamp_months=t_months,
                observation_age=age
            )

            # 1. Construct Nodes
            # Molecular Node
            has_mol = rec.get("has_mol", False)
            feat_mol = torch.tensor(rec.get("feat_mol", np.zeros(self.dim_mol)), dtype=torch.float32)
            id_mol = f"P{patient_id}_{v_name}_mol"
            node_mol = HypergraphNode(
                node_id=id_mol, patient_id=patient_id, visit=v_name,
                modality="molecular", features=feat_mol,
                timestamp_months=t_months, observation_age=age, available=has_mol
            )
            snap.nodes[id_mol] = node_mol
            snap.availability["molecular"] = has_mol

            # Structural Node (MRI)
            has_struct = rec.get("has_struct", False)
            feat_struct = torch.tensor(rec.get("feat_struct", np.zeros(self.dim_struct)), dtype=torch.float32)
            id_struct = f"P{patient_id}_{v_name}_struct"
            node_struct = HypergraphNode(
                node_id=id_struct, patient_id=patient_id, visit=v_name,
                modality="structural", features=feat_struct,
                timestamp_months=t_months, observation_age=age, available=has_struct
            )
            snap.nodes[id_struct] = node_struct
            snap.availability["structural"] = has_struct

            # Functional Node (Motor UPDRS)
            has_func = rec.get("has_func", False)
            feat_func = torch.tensor(rec.get("feat_func", np.zeros(self.dim_func)), dtype=torch.float32)
            id_func = f"P{patient_id}_{v_name}_func"
            node_func = HypergraphNode(
                node_id=id_func, patient_id=patient_id, visit=v_name,
                modality="functional", features=feat_func,
                timestamp_months=t_months, observation_age=age, available=has_func
            )
            snap.nodes[id_func] = node_func
            snap.availability["functional"] = has_func

            # Temporal Context Node
            id_temp = f"P{patient_id}_{v_name}_temp"
            t_feat = torch.tensor([t_months / 12.0, age / 100.0, float(has_mol), float(has_struct), float(has_func)], dtype=torch.float32)
            node_temp = HypergraphNode(
                node_id=id_temp, patient_id=patient_id, visit=v_name,
                modality="temporal", features=t_feat,
                timestamp_months=t_months, observation_age=age, available=True
            )
            snap.nodes[id_temp] = node_temp
            snap.availability["temporal"] = True

            # 2. Construct Hyperedges for this Visit
            # A. Intra-modality Hyperedges
            e_mol_id = f"E_{patient_id}_{v_name}_intra_mol"
            snap.hyperedges[e_mol_id] = Hyperedge(
                edge_id=e_mol_id, edge_type="intra_molecular",
                patient_id=patient_id, visit=v_name, node_ids=[id_mol]
            )

            e_struct_id = f"E_{patient_id}_{v_name}_intra_struct"
            snap.hyperedges[e_struct_id] = Hyperedge(
                edge_id=e_struct_id, edge_type="intra_structural",
                patient_id=patient_id, visit=v_name, node_ids=[id_struct]
            )

            e_func_id = f"E_{patient_id}_{v_name}_intra_func"
            snap.hyperedges[e_func_id] = Hyperedge(
                edge_id=e_func_id, edge_type="intra_functional",
                patient_id=patient_id, visit=v_name, node_ids=[id_func]
            )

            # B. Cross-Modal Visit Hyperedge (bridges Mol, Struct, and Func)
            cross_nodes = [id_mol, id_struct, id_func]
            e_cross_id = f"E_{patient_id}_{v_name}_cross_modal"
            snap.hyperedges[e_cross_id] = Hyperedge(
                edge_id=e_cross_id, edge_type="cross_modal_visit",
                patient_id=patient_id, visit=v_name, node_ids=cross_nodes,
                metadata={"available_modalities": [m for m, av in snap.availability.items() if av]}
            )

            # C. Temporal Transition Hyperedge (from previous visit)
            if prev_snapshot is not None:
                dt_months = t_months - prev_snapshot.timestamp_months
                e_trans_id = f"E_{patient_id}_{prev_snapshot.visit}_to_{v_name}_temporal"
                prev_temp_id = f"P{patient_id}_{prev_snapshot.visit}_temp"
                trans_edge = Hyperedge(
                    edge_id=e_trans_id, edge_type="temporal_transition",
                    patient_id=patient_id, visit=v_name,
                    node_ids=[prev_temp_id, id_temp, id_func],
                    elapsed_time=dt_months / 12.0  # in years
                )
                patient_hg.temporal_edges.append(trans_edge)
                snap.hyperedges[e_trans_id] = trans_edge

            patient_hg.snapshots[v_name] = snap
            prev_snapshot = snap

        return patient_hg
