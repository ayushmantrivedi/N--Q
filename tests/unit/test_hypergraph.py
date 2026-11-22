"""
Unit tests for NĀḌĪ-Q Temporal Hypergraph and TQ-RHNet modules.

Verifies:
- HypergraphNode, Hyperedge, VisitHypergraph data contracts.
- TimeConditionedQuantumHyperedgeEncoder forward pass and output shapes.
- QuantumRelationalDriftCalculator fidelity drift and classical distance metrics.
- IncidenceHypergraphMessagePassingLayer message-passing contracts.
- ClinicalEvent and EventChainExtractor functionality.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from nadiq.models.hypergraph.temporal_hypergraph_builder import (
    HypergraphNode,
    Hyperedge,
)
from nadiq.models.hypergraph.time_conditioned_quantum_hyperedge import (
    TimeConditionedQuantumHyperedgeEncoder,
)
from nadiq.models.hypergraph.quantum_relational_drift import (
    QuantumRelationalDriftCalculator,
)
from nadiq.models.hypergraph.incidence_hypergraph_message_passing import (
    IncidenceHypergraphMessagePassingLayer,
)
from nadiq.models.hypergraph.event_chain_extractor import (
    ClinicalEvent,
    EventChainExtractor,
)


def test_hypergraph_node_and_edge_contract():
    feat = torch.tensor([0.1, 0.2, 0.3, 0.4])
    node = HypergraphNode(
        node_id="bio_bl",
        patient_id=3001,
        visit="BL",
        modality="molecular",
        features=feat,
        timestamp_months=0.0,
        observation_age=0.0,
        available=True,
    )
    assert node.node_id == "bio_bl"
    assert node.patient_id == 3001
    assert node.modality == "molecular"
    assert node.features.shape == (4,)

    edge = Hyperedge(
        edge_id="e_pair_01",
        edge_type="pairwise_bio_motor",
        patient_id=3001,
        visit="BL",
        node_ids=["bio_bl", "motor_bl"],
        elapsed_time=0.0,
    )
    assert "bio_bl" in edge.node_ids
    assert edge.patient_id == 3001


def test_time_conditioned_quantum_hyperedge():
    encoder = TimeConditionedQuantumHyperedgeEncoder(
        dim_mol=15,
        dim_struct=36,
        dim_func=35,
        n_qubits=8,
        n_layers=1,
        entangle=True,
        enable_time_conditioning=True,
        enable_three_body=True,
        seed=42,
    )
    batch_size = 2
    h_mol = torch.randn(batch_size, 15)
    h_struct = torch.randn(batch_size, 36)
    h_func = torch.randn(batch_size, 35)
    dt_years = torch.tensor([[1.0], [2.0]])
    avail = torch.ones(batch_size, 3)

    rep, aux = encoder.forward(
        h_mol, h_struct, h_func,
        dt_years=dt_years,
        avail_mask=avail,
    )
    assert rep.shape[0] == batch_size
    assert rep.shape[1] > 0
    assert "C_mol_struct" in aux
    assert "Cumulant_3body" in aux


def test_quantum_relational_drift():
    calc = QuantumRelationalDriftCalculator(rbf_gamma=0.05)
    psi1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    psi2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.complex64)

    # Orthogonal quantum states: fid=0.0, drift=1.0
    fid_orth, drift_orth = calc.compute_quantum_drift(psi1, psi2)
    assert pytest.approx(fid_orth, abs=1e-4) == 0.0
    assert pytest.approx(drift_orth, abs=1e-4) == 1.0

    # Identical quantum states: fid=1.0, drift=0.0
    fid_ident, drift_ident = calc.compute_quantum_drift(psi1, psi1)
    assert pytest.approx(fid_ident, abs=1e-4) == 1.0
    assert pytest.approx(drift_ident, abs=1e-4) == 0.0

    # Baseline visit (no prior state)
    fid_bl, drift_bl = calc.compute_quantum_drift(None, psi1)
    assert fid_bl == 1.0
    assert drift_bl == 0.0

    # Classical distances
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([1.0, 2.0, 3.0])
    c_dist = calc.compute_classical_distances(x1, x2)
    assert "euclidean_distance" in c_dist
    assert pytest.approx(c_dist["euclidean_distance"], abs=1e-4) == 0.0


def test_incidence_message_passing_layer():
    layer = IncidenceHypergraphMessagePassingLayer(
        dim_mol=15,
        dim_struct=36,
        dim_func=35,
        dim_cross_edge=24,
        dim_hidden=32,
    )
    batch_size = 2
    h_mol = torch.randn(batch_size, 15)
    h_struct = torch.randn(batch_size, 36)
    h_func = torch.randn(batch_size, 35)
    e_cross = torch.randn(batch_size, 24)
    drift = torch.tensor([[0.1], [0.2]])
    avail = torch.ones(batch_size, 3)

    nodes_out, e_out, aux = layer.forward(
        h_mol, h_struct, h_func, e_cross,
        drift_dq=drift,
        avail_mask=avail,
    )
    assert "molecular" in nodes_out
    assert "structural" in nodes_out
    assert "functional" in nodes_out
    assert nodes_out["molecular"].shape == (batch_size, 32)
    assert e_out.shape == (batch_size, 32)


def test_event_chain_extractor():
    extractor = EventChainExtractor(
        asyn_elevation_threshold=0.5,
        caudate_decline_threshold=-0.5,
        motor_worsening_threshold=3.0,
    )
    df = pd.DataFrame([
        {
            "PATNO": 3001,
            "EVENT_ID": "BL",
            "timestamp_months": 0.0,
            "updrs3_score": 15.0,
            "asyn": 1000.0,
            "MIA_CAUDATE_BILAT": 3000.0,
        },
        {
            "PATNO": 3001,
            "EVENT_ID": "V04",
            "timestamp_months": 12.0,
            "updrs3_score": 25.0,
            "asyn": 2000.0,
            "MIA_CAUDATE_BILAT": 1000.0,
        },
    ])
    events = extractor.extract_patient_events(patient_id=3001, longitudinal_df=df)
    assert isinstance(events, list)
    assert len(events) == 3
    for ev in events:
        assert isinstance(ev, ClinicalEvent)
        assert ev.patient_id == 3001
        assert ev.timestamp_months == 12.0
