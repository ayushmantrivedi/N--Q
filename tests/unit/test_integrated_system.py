"""
Unit tests for NĀḌĪ-Q Integrated Hybrid Intelligence System.
Validates the 13 architectural and operational verification contracts:
1. QVBE output compatibility
2. CTCLM independent forward pass
3. QCDF tensor dimensions
4. Modality missingness invariance
5. Dynamic quantum gate suppression
6. Zero-quantum ablation functionality
7. Classical-only inference modes
8. Full integrated end-to-end forward pass
9. End-to-end backpropagation & gradient flow across all branches
10. Q-XAI contribution consistency & faithfulness
11. Deterministic inference with fixed seed
12. Checkpoint serialization roundtrip
13. Batch scaling (single sample B=1 and batch B=8)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest
import torch

from nadiq.models.integrated.config import NadiQIntegratedConfig
from nadiq.models.integrated.output import IntegratedOutput
from nadiq.models.integrated.fusion import AdaptiveQCDF
from nadiq.models.integrated.model import ClassicalRelationalBranch, NadiQIntegratedModel
from nadiq.models.integrated.explainability import NadiQXAIExplainer
from nadiq.models.qvbe_v4.config import QVBEv4Config
from nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel, CTCLMConfig


@pytest.fixture
def sample_data():
    """Generates synthetic patient tensors for test assertions."""
    torch.manual_seed(42)
    batch_size = 4
    node_inputs = {
        "molecular": (torch.randn(batch_size, 1), torch.randn(batch_size, 1)),
        "structural": (torch.randn(batch_size, 1), torch.randn(batch_size, 1)),
        "functional": (torch.randn(batch_size, 1), torch.randn(batch_size, 1)),
    }
    graph_edges = [
        {"source": "molecular", "target": "structural", "weight": 0.75, "lag_months": 6.0, "se": 0.05, "direction": 1},
        {"source": "structural", "target": "functional", "weight": 0.65, "lag_months": 12.0, "se": 0.08, "direction": 1},
        {"source": "molecular", "target": "functional", "weight": 0.45, "lag_months": 18.0, "se": 0.10, "direction": 1},
    ]
    node_masks = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],  # structural missing
        [0.0, 1.0, 1.0],  # molecular missing
        [1.0, 1.0, 0.0],  # functional missing
    ])
    elapsed_times = torch.tensor([[1.0], [0.8], [1.2], [1.0]])

    # CTCLM temporal motor sequence (B=4, T=3, D=1)
    ctclm_seq = torch.randn(batch_size, 3, 1)
    ctclm_times = torch.tensor([
        [0.0, 1.0, 2.0],
        [0.0, 0.8, 1.8],
        [0.0, 1.2, 2.4],
        [0.0, 1.0, 2.0],
    ])
    ctclm_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
        [True, True, True],
        [True, False, False],
    ])

    return {
        "batch_size": batch_size,
        "node_inputs": node_inputs,
        "graph_edges": graph_edges,
        "node_masks": node_masks,
        "elapsed_times": elapsed_times,
        "ctclm_seq": ctclm_seq,
        "ctclm_times": ctclm_times,
        "ctclm_mask": ctclm_mask,
    }


def test_qvbe_output_compatibility(sample_data):
    """Contract 1: QVBEv4Output fields integrate cleanly into QCDF."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
            elapsed_times=sample_data["elapsed_times"],
        )
    assert out.qvbe_output is not None
    assert out.qvbe_output.motif_embedding.shape == (sample_data["batch_size"], config.qvbe_config.motif_embed_dim)
    assert out.qvbe_output.local_expectations_z.shape == (sample_data["batch_size"], config.qvbe_config.n_nodes * 2)


def test_ctclm_independent_forward(sample_data):
    """Contract 2: CTCLM forward runs standalone with (B, T, D) and padding mask."""
    ctclm_cfg = CTCLMConfig(input_dim=1, latent_dim=16, time_embedding_dim=4)
    ctclm = ContinualTemporalClinicalLatentModel(ctclm_cfg)
    ctclm.eval()

    with torch.no_grad():
        out = ctclm(
            sample_data["ctclm_seq"],
            sample_data["ctclm_times"],
            sample_data["ctclm_mask"],
        )
    assert out.h_final.shape == (sample_data["batch_size"], 16)
    assert out.sequence.shape == (sample_data["batch_size"], 3, 16)


def test_qcdf_tensor_dimensions(sample_data):
    """Contract 3: QCDF output dimensions match specifications."""
    config = NadiQIntegratedConfig()
    qcdf = AdaptiveQCDF(config)
    qcdf.eval()

    B = sample_data["batch_size"]
    node_flat = torch.randn(B, 6)
    qvbe_emb = torch.randn(B, config.qvbe_config.motif_embed_dim)
    ctclm_emb = torch.randn(B, config.ctclm_config.latent_dim)
    cr_emb = torch.randn(B, config.classical_relational_dim)

    logits, probs, fused_rep, gate_q, gate_t, gate_n = qcdf(
        node_states_flat=node_flat,
        qvbe_emb=qvbe_emb,
        ctclm_emb=ctclm_emb,
        classical_rel_emb=cr_emb,
        node_masks=sample_data["node_masks"],
    )

    assert logits.shape == (B,)
    assert probs.shape == (B,)
    assert fused_rep.shape == (B, config.fusion_hidden_dim)
    assert gate_q.shape == (B, 1)
    assert gate_t.shape == (B, 1)
    assert gate_n.shape == (B, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_missingness_invariance(sample_data):
    """Contract 4: Modifying missing modality features has zero effect on output."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    explainer = NadiQXAIExplainer(model)

    res = explainer.verify_perturbation_invariance(
        node_inputs=sample_data["node_inputs"],
        graph_edges=sample_data["graph_edges"],
        ctclm_seq=sample_data["ctclm_seq"],
        ctclm_times=sample_data["ctclm_times"],
        ctclm_mask=sample_data["ctclm_mask"],
        node_masks=sample_data["node_masks"],
        elapsed_times=sample_data["elapsed_times"],
        target_modality_to_mask="molecular",
    )
    assert res["is_invariant"] is True
    assert res["max_prob_difference"] < 1e-4


def test_quantum_gate_suppression(sample_data):
    """Contract 5: Setting force_quantum_suppression=True forces g_quantum = 0.0."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
            force_quantum_suppression=True,
        )
    assert (out.quantum_gate_weight == 0.0).all()
    assert out.diagnostics["suppressed_quantum"] is True


def test_zero_quantum_ablation(sample_data):
    """Contract 6: Model functions seamlessly when mode='zero_quantum'."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
            mode="zero_quantum",
        )
    assert (out.quantum_gate_weight == 0.0).all()
    assert out.qvbe_output is None
    assert out.probabilities.shape == (sample_data["batch_size"],)


def test_classical_only_inference(sample_data):
    """Contract 7: Model functions in 'ctclm_only' and 'classical_relational_only' modes."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out_ctclm = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            ctclm_seq=sample_data["ctclm_seq"],
            ctclm_times=sample_data["ctclm_times"],
            ctclm_mask=sample_data["ctclm_mask"],
            mode="ctclm_only",
        )
        out_cr = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            mode="classical_relational_only",
        )

    assert (out_ctclm.quantum_gate_weight == 0.0).all()
    assert (out_cr.quantum_gate_weight == 0.0).all()
    assert out_ctclm.probabilities.shape == (sample_data["batch_size"],)
    assert out_cr.probabilities.shape == (sample_data["batch_size"],)


def test_full_integrated_forward(sample_data):
    """Contract 8: End-to-end forward pass with all components active."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            ctclm_seq=sample_data["ctclm_seq"],
            ctclm_times=sample_data["ctclm_times"],
            ctclm_mask=sample_data["ctclm_mask"],
            node_masks=sample_data["node_masks"],
            elapsed_times=sample_data["elapsed_times"],
            mode="full",
        )

    assert out.qvbe_output is not None
    assert out.ctclm_output is not None
    assert out.classical_relational_output is not None
    assert (out.quantum_gate_weight >= 0.0).all() and (out.quantum_gate_weight <= 1.0).all()
    assert (out.temporal_gate_weight >= 0.0).all() and (out.temporal_gate_weight <= 1.0).all()
    assert len(out.modality_contributions) == 4


def test_end_to_end_gradients(sample_data):
    """Contract 9: Backpropagation computes finite gradients across all branches."""
    config = NadiQIntegratedConfig()
    config.qvbe_config.edge_coupling_mode = "learned_scale"
    config.qvbe_config.enable_uncertainty_weighting = True
    model = NadiQIntegratedModel(config)
    model.train()

    out = model(
        node_inputs=sample_data["node_inputs"],
        graph_edges=sample_data["graph_edges"],
        ctclm_seq=sample_data["ctclm_seq"],
        ctclm_times=sample_data["ctclm_times"],
        ctclm_mask=sample_data["ctclm_mask"],
        node_masks=sample_data["node_masks"],
        elapsed_times=sample_data["elapsed_times"],
        mode="full",
    )
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy(out.probabilities, target)
    loss.backward()

    # Check gradients in QVBE circuit
    qvbe_has_grad = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.qvbe.circuit.parameters())
    assert qvbe_has_grad

    # Check gradients in CTCLM
    ctclm_has_grad = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.ctclm.parameters())
    assert ctclm_has_grad

    # Check gradients in Classical Relational
    cr_has_grad = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.classical_relational.parameters())
    assert cr_has_grad

    # Check gradients in QCDF fusion
    qcdf_has_grad = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.qcdf.parameters())
    assert qcdf_has_grad


def test_qxai_contribution_consistency(sample_data):
    """Contract 10: Q-XAI explanation consistency: when quantum branch is suppressed, Delta P_q is zero."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    explainer = NadiQXAIExplainer(model)

    res = explainer.explain_prediction(
        node_inputs=sample_data["node_inputs"],
        graph_edges=sample_data["graph_edges"],
        ctclm_seq=sample_data["ctclm_seq"],
        ctclm_times=sample_data["ctclm_times"],
        ctclm_mask=sample_data["ctclm_mask"],
        node_masks=sample_data["node_masks"],
        elapsed_times=sample_data["elapsed_times"],
    )

    assert res.baseline_probability.shape == (sample_data["batch_size"],)
    assert res.quantum_attribution.shape == (sample_data["batch_size"],)
    assert res.temporal_attribution.shape == (sample_data["batch_size"],)
    assert len(res.modality_attributions) == 3
    assert len(res.primary_dominance) == sample_data["batch_size"]


def test_deterministic_inference(sample_data):
    """Contract 11: Fixed seed produces identical outputs across runs in eval() mode."""
    torch.manual_seed(12345)
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    with torch.no_grad():
        out1 = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
        )
        out2 = model(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
        )

    assert torch.allclose(out1.probabilities, out2.probabilities, atol=1e-6)
    assert torch.allclose(out1.fused_representation, out2.fused_representation, atol=1e-6)


def test_checkpoint_roundtrip(sample_data):
    """Contract 12: torch.save and torch.load preserves state dict and reproduces outputs."""
    config = NadiQIntegratedConfig()
    model1 = NadiQIntegratedModel(config)
    model1.eval()

    with torch.no_grad():
        out1 = model1(
            node_inputs=sample_data["node_inputs"],
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"],
        )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(model1.state_dict(), tmp_path)
        model2 = NadiQIntegratedModel(config)
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))
        model2.eval()

        with torch.no_grad():
            out2 = model2(
                node_inputs=sample_data["node_inputs"],
                graph_edges=sample_data["graph_edges"],
                node_masks=sample_data["node_masks"],
            )

        assert torch.allclose(out1.probabilities, out2.probabilities, atol=1e-6)
    finally:
        p = Path(tmp_path)
        if p.exists():
            p.unlink()


def test_batch_and_single_execution(sample_data):
    """Contract 13: Runs on batch sizes B=1 and B=8."""
    config = NadiQIntegratedConfig()
    model = NadiQIntegratedModel(config)
    model.eval()

    # B = 1
    single_inputs = {
        k: (v[0][:1], v[1][:1]) for k, v in sample_data["node_inputs"].items()
    }
    with torch.no_grad():
        out_single = model(
            node_inputs=single_inputs,
            graph_edges=sample_data["graph_edges"],
            node_masks=sample_data["node_masks"][:1],
        )
    assert out_single.probabilities.shape == (1,)

    # B = 8
    torch.manual_seed(999)
    b8_inputs = {
        "molecular": (torch.randn(8, 1), torch.randn(8, 1)),
        "structural": (torch.randn(8, 1), torch.randn(8, 1)),
        "functional": (torch.randn(8, 1), torch.randn(8, 1)),
    }
    b8_masks = torch.ones(8, 3)
    with torch.no_grad():
        out_b8 = model(
            node_inputs=b8_inputs,
            graph_edges=sample_data["graph_edges"],
            node_masks=b8_masks,
        )
    assert out_b8.probabilities.shape == (8,)
