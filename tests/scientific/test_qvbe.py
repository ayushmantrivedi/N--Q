"""
Focused tests for QVBE (Quantum Variational Biomarker Encoder).

Tests cover:
  - Input/output shapes
  - Multiple input dimensions
  - Different qubit counts
  - Different circuit depths
  - Deterministic initialization
  - Gradient propagation
  - Batch processing
  - Integration with the frozen ClinicalSample / FeatureBlock contracts

These tests are marked @pytest.mark.requires_quantum and will be skipped
if PennyLane is not installed. They run on the CPU-exact statevector
simulator (no shots) for determinism and gradient support.

IMPORTANT: These tests verify computational correctness of the QVBE module.
They do NOT demonstrate quantum advantage or validate clinical hypotheses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nadiq.models.quantum.qvbe import (
    QVBEConfig,
    QVBEOutput,
    QuantumVariationalBiomarkerEncoder,
)

# ---------------------------------------------------------------------------
# Mark — skip if PennyLane unavailable (CI without quantum deps)
# ---------------------------------------------------------------------------

try:
    import pennylane  # noqa: F401
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

requires_quantum = pytest.mark.skipif(
    not HAS_PENNYLANE,
    reason="PennyLane not installed; skipping QVBE tests.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> QVBEConfig:
    return QVBEConfig(input_dim=13, n_qubits=4, n_layers=2, seed=42)


@pytest.fixture
def default_qvbe(default_config: QVBEConfig) -> QuantumVariationalBiomarkerEncoder:
    return QuantumVariationalBiomarkerEncoder(default_config)


@pytest.fixture
def batch_x(default_config: QVBEConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(3, default_config.input_dim)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestQVBEConfig:
    @requires_quantum
    def test_default_latent_dim_equals_n_qubits(self) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2)
        assert cfg.latent_dim == cfg.n_qubits

    @requires_quantum
    def test_param_count_formula(self) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2)
        expected_vqc = 3 * 4 * 2  # 3 rots * 4 qubits * 2 layers = 24
        assert cfg.n_vqc_params == expected_vqc

    @requires_quantum
    def test_invalid_input_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            QVBEConfig(input_dim=0, n_qubits=4, n_layers=2)

    @requires_quantum
    def test_invalid_n_qubits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits"):
            QVBEConfig(input_dim=8, n_qubits=0, n_layers=2)

    @requires_quantum
    def test_invalid_n_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            QVBEConfig(input_dim=8, n_qubits=4, n_layers=0)


# ---------------------------------------------------------------------------
# 2. Output shapes
# ---------------------------------------------------------------------------


class TestQVBEOutputShapes:
    @requires_quantum
    def test_embedding_shape_batch3(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        out = default_qvbe(batch_x)
        assert isinstance(out, QVBEOutput)
        assert out.embedding.shape == (3, default_qvbe.config.n_qubits)

    @requires_quantum
    def test_pre_quantum_shape(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        out = default_qvbe(batch_x)
        assert out.pre_quantum.shape == (3, default_qvbe.config.n_qubits)

    @requires_quantum
    def test_single_sample_auto_unsqueeze(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        default_config: QVBEConfig,
    ) -> None:
        x = torch.randn(default_config.input_dim)
        out = default_qvbe(x)
        assert out.embedding.shape == (1, default_config.n_qubits)

    @requires_quantum
    def test_embedding_range(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        """Pauli-Z expectation values must lie in [-1, 1]."""
        out = default_qvbe(batch_x)
        emb = out.embedding.detach().numpy()
        assert np.all(emb >= -1.0 - 1e-5), f"Embedding below -1: {emb.min()}"
        assert np.all(emb <=  1.0 + 1e-5), f"Embedding above +1: {emb.max()}"

    @requires_quantum
    def test_pre_quantum_in_angle_range(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        """After tanh*pi the projected angles must lie in (-pi, pi)."""
        out = default_qvbe(batch_x)
        angles = out.pre_quantum.detach().numpy()
        assert np.all(np.abs(angles) < math.pi + 1e-5)


# ---------------------------------------------------------------------------
# 3. Different input dimensions
# ---------------------------------------------------------------------------


class TestQVBEInputDimensions:
    @requires_quantum
    @pytest.mark.parametrize("input_dim", [1, 4, 8, 13, 20])
    def test_variable_input_dims(self, input_dim: int) -> None:
        cfg = QVBEConfig(input_dim=input_dim, n_qubits=4, n_layers=2, seed=42)
        qvbe = QuantumVariationalBiomarkerEncoder(cfg)
        x = torch.randn(2, input_dim)
        out = qvbe(x)
        assert out.embedding.shape == (2, 4)


# ---------------------------------------------------------------------------
# 4. Different qubit counts
# ---------------------------------------------------------------------------


class TestQVBEQubitCounts:
    @requires_quantum
    @pytest.mark.parametrize("n_qubits", [1, 2, 4, 6])
    def test_variable_qubit_counts(self, n_qubits: int) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=n_qubits, n_layers=2, seed=42)
        qvbe = QuantumVariationalBiomarkerEncoder(cfg)
        x = torch.randn(2, 8)
        out = qvbe(x)
        assert out.embedding.shape == (2, n_qubits)


# ---------------------------------------------------------------------------
# 5. Different circuit depths
# ---------------------------------------------------------------------------


class TestQVBECircuitDepths:
    @requires_quantum
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_variable_circuit_depths(self, n_layers: int) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=4, n_layers=n_layers, seed=42)
        qvbe = QuantumVariationalBiomarkerEncoder(cfg)
        x = torch.randn(2, 8)
        out = qvbe(x)
        assert out.embedding.shape == (2, 4)

    @requires_quantum
    def test_param_count_scales_with_layers(self) -> None:
        cfg1 = QVBEConfig(input_dim=8, n_qubits=4, n_layers=1, seed=42)
        cfg2 = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2, seed=42)
        q1 = QuantumVariationalBiomarkerEncoder(cfg1)
        q2 = QuantumVariationalBiomarkerEncoder(cfg2)
        # VQC params scale linearly with layers
        assert q2.n_parameters > q1.n_parameters


# ---------------------------------------------------------------------------
# 6. Deterministic initialization
# ---------------------------------------------------------------------------


class TestQVBEDeterministicInit:
    @requires_quantum
    def test_same_seed_same_weights(self) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2, seed=42)
        q1 = QuantumVariationalBiomarkerEncoder(cfg)
        q2 = QuantumVariationalBiomarkerEncoder(cfg)
        for p1, p2 in zip(q1.parameters(), q2.parameters()):
            assert torch.allclose(p1, p2), "Same seed must yield identical weights."

    @requires_quantum
    def test_different_seeds_different_weights(self) -> None:
        cfg1 = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2, seed=42)
        cfg2 = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2, seed=99)
        q1 = QuantumVariationalBiomarkerEncoder(cfg1)
        q2 = QuantumVariationalBiomarkerEncoder(cfg2)
        # At least some weights should differ
        any_different = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(q1.parameters(), q2.parameters())
        )
        assert any_different, "Different seeds should yield different VQC weights."

    @requires_quantum
    def test_forward_deterministic_same_input(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        """Same input + same weights → identical embeddings (statevector mode)."""
        out1 = default_qvbe(batch_x)
        out2 = default_qvbe(batch_x)
        assert torch.allclose(out1.embedding, out2.embedding, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Gradient propagation
# ---------------------------------------------------------------------------


class TestQVBEGradients:
    @requires_quantum
    def test_loss_has_gradient_to_vqc_params(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        """
        A scalar loss computed from the embedding must back-propagate
        gradients to the VQC weights.
        """
        default_qvbe.train()
        out = default_qvbe(batch_x)
        loss = out.embedding.mean()
        loss.backward()

        assert default_qvbe.vqc.weights.grad is not None, (
            "VQC weights must receive gradients via autograd."
        )
        assert not torch.all(default_qvbe.vqc.weights.grad == 0), (
            "VQC weight gradients must be non-zero."
        )

    @requires_quantum
    def test_loss_has_gradient_to_projection_params(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        """Gradients must reach the classical projection layer."""
        default_qvbe.train()
        out = default_qvbe(batch_x)
        loss = out.embedding.mean()
        loss.backward()

        assert default_qvbe.projection.weight.grad is not None
        assert not torch.all(default_qvbe.projection.weight.grad == 0)


# ---------------------------------------------------------------------------
# 8. Batch processing
# ---------------------------------------------------------------------------


class TestQVBEBatchProcessing:
    @requires_quantum
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_sizes(self, batch_size: int) -> None:
        cfg = QVBEConfig(input_dim=8, n_qubits=4, n_layers=2, seed=42)
        qvbe = QuantumVariationalBiomarkerEncoder(cfg)
        x = torch.randn(batch_size, 8)
        out = qvbe(x)
        assert out.embedding.shape == (batch_size, 4)

    @requires_quantum
    def test_nan_input_handled_gracefully(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
    ) -> None:
        """NaN values in input should be zeroed, not crash."""
        x = torch.full((2, 13), float("nan"))
        out = default_qvbe(x)
        # Should not raise; embedding should be finite
        assert torch.isfinite(out.embedding).all()


# ---------------------------------------------------------------------------
# 9. Quantum backend integration
# ---------------------------------------------------------------------------


class TestQVBEBackendIntegration:
    @requires_quantum
    def test_vqc_params_snapshot_correct_shape(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
        batch_x: torch.Tensor,
    ) -> None:
        out = default_qvbe(batch_x)
        expected_shape = (
            default_qvbe.config.n_layers,
            default_qvbe.config.n_qubits,
            3,
        )
        assert out.vqc_params.shape == expected_shape

    @requires_quantum
    def test_get_config_is_serialisable(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
    ) -> None:
        cfg = default_qvbe.get_config()
        import json
        json.dumps(cfg)  # Must not raise

    @requires_quantum
    def test_repr_contains_key_info(
        self,
        default_qvbe: QuantumVariationalBiomarkerEncoder,
    ) -> None:
        r = repr(default_qvbe)
        assert "input_dim" in r
        assert "n_qubits" in r
        assert "n_layers" in r


# ---------------------------------------------------------------------------
# 10. Integration with frozen ClinicalSample / FeatureBlock
# ---------------------------------------------------------------------------


class TestQVBEContractIntegration:
    @requires_quantum
    def test_feature_block_feeds_qvbe(self) -> None:
        """
        Verify that a FeatureBlock.values tensor can be passed directly
        through QVBE without modification.
        """
        from nadiq.data.contracts import (
            FeatureBlock,
            FeatureDtype,
            ModalityType,
        )

        # Simulate the TADPOLE clinical feature block (13 features, 2 patients)
        feat_values = np.random.randn(2, 13).astype(np.float32)
        missing = np.isnan(feat_values)

        fb = FeatureBlock(
            modality=ModalityType.TABULAR,
            values=feat_values,
            feature_names=[f"feat_{i}" for i in range(13)],
            missing_mask=missing,
            dtype=FeatureDtype.CONTINUOUS,
        )

        cfg = QVBEConfig(input_dim=13, n_qubits=4, n_layers=2, seed=42)
        qvbe = QuantumVariationalBiomarkerEncoder(cfg)

        x = torch.tensor(fb.values)
        out = qvbe(x)
        assert out.embedding.shape == (2, 4)
