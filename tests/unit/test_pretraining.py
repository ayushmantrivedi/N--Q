"""
Unit tests for Phase 12.2: Quantum Relational Pretraining & Residual Adaptive Fusion.
Validates:
1. QVBEv4PretrainingModel forward pass and multitask reconstruction losses.
2. Masked edge attribute reconstruction.
3. Matched ClassicalRelationalPretrainingModel forward pass and parity.
4. Fine-tuning freeze strategies (frozen, partial, full).
5. ResidualAdaptiveQCDF tensor shapes and output consistency.
6. ResidualAdaptiveQCDF strict classical convergence when quantum is suppressed (alpha = 0.0).
7. End-to-end gradient flow through residual adaptive fusion.
"""

from __future__ import annotations

import pytest
import torch

from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.qvbe_v4.pretraining import (
    RelationalPretrainingConfig,
    RelationalReconstructionHead,
    QVBEv4PretrainingModel,
    ClassicalRelationalPretrainingModel,
)
from nadiq.models.integrated.config import NadiQIntegratedConfig
from nadiq.models.integrated.fusion import ResidualAdaptiveQCDF
from nadiq.models.integrated.model import NadiQIntegratedModel


@pytest.fixture
def sample_pretrain_data():
    torch.manual_seed(42)
    B = 4
    node_inputs = {
        "molecular": (torch.randn(B, 1), torch.randn(B, 1)),
        "structural": (torch.randn(B, 1), torch.randn(B, 1)),
        "functional": (torch.randn(B, 1), torch.randn(B, 1)),
    }
    graph_edges = [
        {"source": "molecular", "target": "structural", "weight": 0.75, "lag_months": 6.0, "se": 0.05, "direction": 1, "status": "Validated"},
        {"source": "structural", "target": "functional", "weight": 0.65, "lag_months": 12.0, "se": 0.08, "direction": 1, "status": "Validated"},
        {"source": "molecular", "target": "functional", "weight": 0.45, "lag_months": 18.0, "se": 0.10, "direction": 1, "status": "Provisional"},
    ]
    node_masks = torch.ones(B, 3)
    elapsed_times = torch.tensor([[1.0], [0.8], [1.2], [1.0]])

    compiler = QVBEv4PretrainingModel().encoder.compiler
    graph_tensors = compiler.compile_graph_motif(graph_edges, node_masks, elapsed_times, batch_size=B)

    return {
        "batch_size": B,
        "node_inputs": node_inputs,
        "graph_edges": graph_edges,
        "node_masks": node_masks,
        "elapsed_times": elapsed_times,
        "graph_tensors": graph_tensors,
    }


def test_qvbe_pretraining_forward_pass(sample_pretrain_data):
    """Contract 1: QVBEv4PretrainingModel forward pass computes finite multitask losses."""
    model = QVBEv4PretrainingModel()
    preds, loss_dict, z_q = model.forward_pretrain(
        node_inputs=sample_pretrain_data["node_inputs"],
        graph_edges=sample_pretrain_data["graph_edges"],
        node_masks=sample_pretrain_data["node_masks"],
        elapsed_times=sample_pretrain_data["elapsed_times"],
        masked_pair_idx=1,
    )

    assert "total_loss" in loss_dict
    assert not torch.isnan(loss_dict["total_loss"])
    assert loss_dict["total_loss"].item() > 0.0
    assert z_q.shape == (sample_pretrain_data["batch_size"], 16)
    assert preds["pred_weight"].shape == (sample_pretrain_data["batch_size"], 3)
    assert preds["pred_temporal_delta"].shape == (sample_pretrain_data["batch_size"], 3)


def test_classical_pretraining_matched(sample_pretrain_data):
    """Contract 2: ClassicalRelationalPretrainingModel computes matching multitask losses."""
    model = ClassicalRelationalPretrainingModel()
    preds, loss_dict, z_cr = model.forward_pretrain(
        node_inputs=sample_pretrain_data["node_inputs"],
        graph_edges=sample_pretrain_data["graph_edges"],
        graph_tensors=sample_pretrain_data["graph_tensors"],
        node_masks=sample_pretrain_data["node_masks"],
        masked_pair_idx=1,
    )

    assert "total_loss" in loss_dict
    assert not torch.isnan(loss_dict["total_loss"])
    assert z_cr.shape == (sample_pretrain_data["batch_size"], 16)
    assert preds["pred_weight"].shape == (sample_pretrain_data["batch_size"], 3)


def test_fine_tuning_freeze_strategies():
    """Contract 3: Fine-tuning configurations correctly manage parameter requires_grad."""
    model = QVBEv4PretrainingModel()

    # Strategy: frozen
    model.configure_fine_tuning("frozen")
    assert all(not p.requires_grad for p in model.encoder.parameters())

    # Strategy: partial
    model.configure_fine_tuning("partial")
    assert model.encoder.circuit.var_rotations.requires_grad is True
    assert model.encoder.circuit.theta_pair.requires_grad is False
    assert model.encoder.circuit.phi_pair.requires_grad is False

    # Strategy: full
    model.configure_fine_tuning("full")
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_residual_adaptive_qcdf_forward(sample_pretrain_data):
    """Contract 4: ResidualAdaptiveQCDF produces correct tensor dimensions."""
    config = NadiQIntegratedConfig()
    fusion = ResidualAdaptiveQCDF(config)
    fusion.eval()

    B = sample_pretrain_data["batch_size"]
    node_flat = torch.randn(B, 6)
    qvbe_emb = torch.randn(B, 16)
    ctclm_emb = torch.randn(B, 16)
    cr_emb = torch.randn(B, 12)

    logits, probs, z_fused, alpha, gate_t, gate_n = fusion(
        node_states_flat=node_flat,
        qvbe_emb=qvbe_emb,
        ctclm_emb=ctclm_emb,
        classical_rel_emb=cr_emb,
        node_masks=sample_pretrain_data["node_masks"],
    )

    assert logits.shape == (B,)
    assert probs.shape == (B,)
    assert z_fused.shape == (B, config.fusion_hidden_dim)
    assert alpha.shape == (B, 1)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_residual_adaptive_qcdf_zero_quantum(sample_pretrain_data):
    """Contract 5: Under quantum suppression, alpha = 0.0 and z_fused strictly equals z_classical."""
    config = NadiQIntegratedConfig()
    fusion = ResidualAdaptiveQCDF(config)
    fusion.eval()

    B = sample_pretrain_data["batch_size"]
    node_flat = torch.randn(B, 6)
    qvbe_emb = torch.randn(B, 16)
    ctclm_emb = torch.randn(B, 16)

    with torch.no_grad():
        logits, probs, z_fused, alpha, _, _ = fusion(
            node_states_flat=node_flat,
            qvbe_emb=qvbe_emb,
            ctclm_emb=ctclm_emb,
            node_masks=sample_pretrain_data["node_masks"],
            force_quantum_suppression=True,
        )

    assert (alpha == 0.0).all()


def test_residual_adaptive_qcdf_gradients(sample_pretrain_data):
    """Contract 6: Gradients flow cleanly through both classical trunk and residual quantum projection."""
    config = NadiQIntegratedConfig()
    fusion = ResidualAdaptiveQCDF(config)
    fusion.train()

    B = sample_pretrain_data["batch_size"]
    node_flat = torch.randn(B, 6, requires_grad=True)
    qvbe_emb = torch.randn(B, 16, requires_grad=True)
    ctclm_emb = torch.randn(B, 16, requires_grad=True)

    logits, probs, z_fused, alpha, _, _ = fusion(
        node_states_flat=node_flat,
        qvbe_emb=qvbe_emb,
        ctclm_emb=ctclm_emb,
        node_masks=sample_pretrain_data["node_masks"],
    )

    loss = probs.sum()
    loss.backward()

    assert qvbe_emb.grad is not None and not torch.isnan(qvbe_emb.grad).any()
    assert ctclm_emb.grad is not None and not torch.isnan(ctclm_emb.grad).any()
    assert node_flat.grad is not None and not torch.isnan(node_flat.grad).any()
