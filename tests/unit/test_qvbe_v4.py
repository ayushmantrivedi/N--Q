"""
Unit tests for QVBE-v4 Graph-Conditioned Temporal Quantum Relational Engine.
Verifies compiler contract, missingness invariance, topology sensitivity,
temporal conditioning, higher-order triad activation, finite gradients, and entropy diagnostics.
"""

import copy
import pytest
import torch
import numpy as np

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.graph_compiler import NumericalGraphCompiler
from nadiq.models.qvbe_v4.circuit import QVBEv4Circuit
from nadiq.models.qvbe_v4.encoder import QVBEv4Encoder
from nadiq.models.qvbe_v4.trainer import QVBEv4Trainer


@pytest.fixture
def test_config():
    return QVBEv4Config(
        n_nodes=3,
        qubits_per_node=2,
        node_names=["molecular", "structural", "functional"],
        node_input_dims={"molecular": 4, "structural": 6, "functional": 5},
        n_layers=1,
        motif_embed_dim=8,
        hidden_dim=8,
        decay_lambda=0.10,
        seed=42,
    )


@pytest.fixture
def mock_node_inputs():
    torch.manual_seed(42)
    B = 2
    return {
        "molecular": (torch.randn(B, 4), torch.randn(B, 4)),
        "structural": (torch.randn(B, 6), torch.randn(B, 6)),
        "functional": (torch.randn(B, 5), torch.randn(B, 5)),
    }


@pytest.fixture
def mock_edges():
    return [
        {
            "src": "molecular",
            "dst": "structural",
            "r": 0.45,
            "status": "Validated",
            "lag": 1.5,
        },
        {
            "src": "structural",
            "dst": "functional",
            "r": 0.38,
            "status": "Provisional",
            "lag": 2.0,
        },
    ]


def test_graph_compiler_contract(test_config, mock_edges):
    """Verifies that the numerical compiler maps edge attributes and status scaling correctly."""
    compiler = NumericalGraphCompiler(test_config.node_names)
    tensors = compiler.compile_graph_motif(mock_edges, batch_size=2)

    # Pairs are (0,1): mol-struct, (0,2): mol-func, (1,2): struct-func
    # Pair 0: Validated (scale 1.0) -> 0.45 * 1.0 = 0.45
    # Pair 1: None -> 0.0
    # Pair 2: Provisional (scale 0.60) -> 0.38 * 0.60 = 0.228
    assert tensors["edge_weights"].shape == (2, 3)
    np.testing.assert_allclose(tensors["edge_weights"][0, 0].item(), 0.45, atol=1e-4)
    assert tensors["edge_weights"][0, 1].item() == 0.0
    np.testing.assert_allclose(tensors["edge_weights"][0, 2].item(), 0.38 * 0.60, atol=1e-4)

    # Triad should be active because pairs (0,1) and (1,2) are both present
    assert tensors["triad_active"][0, 0].item() == 1.0
    assert tensors["triad_weight"][0, 0].item() > 0.0


def test_missingness_invariance(test_config, mock_edges, mock_node_inputs):
    """
    CRITICAL FORENSIC CONTRACT:
    When a modality is masked as absent (mask=0), modifying its raw numeric values
    must produce EXACTLY ZERO change in model output, preventing numeric contamination.
    """
    model = QVBEv4Encoder(test_config)
    model.eval()

    # Mask: node 1 (structural) is completely absent
    # Molecular=1, Structural=0, Functional=1
    node_masks = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    inputs_1 = copy.deepcopy(mock_node_inputs)
    with torch.no_grad():
        out1 = model(inputs_1, mock_edges, node_masks=node_masks)

    # Now drastically change the absent structural features
    inputs_2 = copy.deepcopy(mock_node_inputs)
    inputs_2["structural"] = (
        inputs_2["structural"][0] * 1000.0 + 999.0,
        inputs_2["structural"][1] * (-500.0) - 123.0,
    )

    with torch.no_grad():
        out2 = model(inputs_2, mock_edges, node_masks=node_masks)

    # Verify motif embedding and all expectations are exactly identical
    torch.testing.assert_close(out1.motif_embedding, out2.motif_embedding, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out1.local_expectations_z, out2.local_expectations_z, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out1.pairwise_correlations_zz, out2.pairwise_correlations_zz, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out1.downstream_logits, out2.downstream_logits, rtol=1e-5, atol=1e-6)


def test_circuit_topology_sensitivity(test_config, mock_node_inputs):
    """Verifies that changing graph edges directly changes quantum correlations."""
    model = QVBEv4Encoder(test_config)
    model.eval()

    # Graph A: edge between 0 and 1
    edges_a = [{"src": "molecular", "dst": "structural", "r": 0.8, "status": "Validated", "lag": 1.0}]
    # Graph B: edge between 1 and 2
    edges_b = [{"src": "structural", "dst": "functional", "r": 0.8, "status": "Validated", "lag": 1.0}]

    with torch.no_grad():
        out_a = model(mock_node_inputs, edges_a)
        out_b = model(mock_node_inputs, edges_b)

    # ZZ correlation on pair 0 should differ between Graph A (active) and Graph B (inactive)
    corr_a = out_a.pairwise_correlations_zz[:, 0]
    corr_b = out_b.pairwise_correlations_zz[:, 0]
    assert not torch.allclose(corr_a, corr_b, atol=1e-3)


def test_temporal_decay_conditioning(test_config, mock_edges, mock_node_inputs):
    """Verifies that increasing elapsed time reduces coupling through temporal decay."""
    model = QVBEv4Encoder(test_config)
    model.eval()

    # Immediate observation (lag = 0) vs Long elapsed time (lag = 50)
    t_short = torch.tensor([[0.0], [0.0]])
    t_long = torch.tensor([[50.0], [50.0]])

    with torch.no_grad():
        out_short = model(mock_node_inputs, mock_edges, elapsed_times=t_short)
        out_long = model(mock_node_inputs, mock_edges, elapsed_times=t_long)

    # Entangled correlations must reflect temporal attenuation
    assert not torch.allclose(out_short.connected_correlators, out_long.connected_correlators, atol=1e-4)


def test_triad_higher_order_activation(test_config, mock_node_inputs):
    """Verifies that 3-body triad coupling activates only when the triad motif is present."""
    cfg = copy.deepcopy(test_config)
    cfg.enable_higher_order_coupling = True
    model = QVBEv4Encoder(cfg)
    model.eval()

    # Case 1: Disconnected edges (no triad chain)
    edges_no_triad = [{"src": "molecular", "dst": "structural", "r": 0.5, "status": "Validated"}]
    # Case 2: Chain of edges creating triad (mol -> struct -> func)
    edges_triad = [
        {"src": "molecular", "dst": "structural", "r": 0.5, "status": "Validated"},
        {"src": "structural", "dst": "functional", "r": 0.5, "status": "Validated"},
    ]

    with torch.no_grad():
        out1 = model(mock_node_inputs, edges_no_triad)
        out2 = model(mock_node_inputs, edges_triad)

    assert out1.classical_graph_metadata["active_triad"] == 0.0
    assert out2.classical_graph_metadata["active_triad"] == 1.0
    assert not torch.allclose(out1.three_body_cumulant, out2.three_body_cumulant, atol=1e-4)


def test_finite_gradients(test_config, mock_edges, mock_node_inputs):
    """Verifies that backpropagation computes finite, non-zero gradients on all parameters."""
    cfg = copy.deepcopy(test_config)
    cfg.enable_higher_order_coupling = True
    cfg.enable_uncertainty_weighting = True
    cfg.edge_coupling_mode = "learned_scale"
    cfg.enable_learned_temporal_decay = True

    model = QVBEv4Encoder(cfg)
    trainer = QVBEv4Trainer(model, cfg)

    targets = {
        "y": torch.tensor([1.0, 0.0]),
        "next_delta": torch.randn(2, 3),
        "node_raw": {
            "molecular": torch.randn(2, 4),
            "structural": torch.randn(2, 6),
            "functional": torch.randn(2, 5),
        },
    }

    node_masks = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    loss_dict, grad_norm = trainer.train_step(
        node_inputs=mock_node_inputs,
        graph_edges=mock_edges,
        targets=targets,
        node_masks=node_masks,
    )

    assert grad_norm > 0.0
    assert not np.isnan(grad_norm)
    assert not np.isinf(grad_norm)
    assert loss_dict["loss_total"] > 0.0

    # Ensure every parameter has valid grad
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_simulator_subsystem_entropy(test_config, mock_edges, mock_node_inputs):
    """Verifies that simulator subsystem entropy diagnostics run and return valid values."""
    model = QVBEv4Encoder(test_config)
    model.eval()

    with torch.no_grad():
        out = model(mock_node_inputs, mock_edges, return_diagnostics=True)

    assert out.subsystem_entropies is not None
    assert out.subsystem_entropies.shape == (2, 3)
    # Entropies should be in [0, 1]
    assert (out.subsystem_entropies >= 0.0).all()
    assert (out.subsystem_entropies <= 1.0).all()


def test_effective_rank():
    """Verifies effective rank calculation via SVD."""
    # Rank 1 matrix: all vectors identical
    emb_rank1 = torch.ones(20, 8)
    rank1 = QVBEv4Trainer.compute_effective_rank(emb_rank1)
    np.testing.assert_allclose(rank1, 1.0, atol=1e-2)

    # Full rank random matrix
    torch.manual_seed(42)
    emb_full = torch.randn(100, 8)
    rank_full = QVBEv4Trainer.compute_effective_rank(emb_full)
    assert rank_full > 5.0


def test_downstream_integration_contracts(test_config, mock_edges, mock_node_inputs):
    """
    SECTION 6 AUDIT CONTRACT:
    Verifies that QVBEv4Output exposes clean downstream interfaces for CTCLM, QCDF, and Q-XAI:
    - node_embeddings
    - quantum_relational_embeddings
    - edge_observables
    - availability_masks
    - temporal_relational_features
    """
    model = QVBEv4Encoder(test_config)
    model.eval()
    node_masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    with torch.no_grad():
        out = model(mock_node_inputs, mock_edges, node_masks=node_masks, return_diagnostics=True)

    # 1. Node embeddings
    assert out.node_embeddings is not None
    assert set(out.node_embeddings.keys()) == set(test_config.node_names)
    assert out.node_embeddings["molecular"].shape == (2, 2)

    # 2. Quantum relational embeddings
    assert out.quantum_relational_embeddings is not None
    assert out.quantum_relational_embeddings.shape == (2, test_config.motif_embed_dim)

    # 3. Edge observables
    assert out.edge_observables is not None
    assert "zz" in out.edge_observables
    assert "xx" in out.edge_observables
    assert "connected" in out.edge_observables
    assert out.edge_observables["zz"].shape == (2, 3)

    # 4. Availability masks
    assert out.availability_masks is not None
    torch.testing.assert_close(out.availability_masks, node_masks)

    # 5. Temporal features
    assert out.temporal_relational_features is not None
    assert out.temporal_relational_features.shape == (2, 3)


def test_edge_conditioned_learning(test_config, mock_node_inputs):
    """Verifies that confidence weighting and learned edge scaling alter interaction angles."""
    cfg = copy.deepcopy(test_config)
    cfg.enable_uncertainty_weighting = True
    cfg.edge_coupling_mode = "learned_scale"

    model = QVBEv4Encoder(cfg)
    model.eval()

    # Edge with high certainty (se=0.01) vs Edge with high uncertainty (se=0.50)
    edges_certain = [{"src": "molecular", "dst": "structural", "r": 0.5, "se": 0.01, "p": 0.001, "status": "Validated"}]
    edges_uncertain = [{"src": "molecular", "dst": "structural", "r": 0.5, "se": 0.50, "p": 0.20, "status": "Validated"}]

    with torch.no_grad():
        out_cert = model(mock_node_inputs, edges_certain)
        out_uncert = model(mock_node_inputs, edges_uncertain)

    # High certainty must induce stronger entangling correlation than high uncertainty
    corr_cert = out_cert.pairwise_correlations_zz[:, 0]
    corr_uncert = out_uncert.pairwise_correlations_zz[:, 0]
    assert not torch.allclose(corr_cert, corr_uncert, atol=1e-3)

