"""
Scientific unit tests for QuantumVariationalBiomarkerEncoderV2 (QVBE-v2) — Nāḍī-Q Phase 4.

Tests cover:
  1. Input/output shapes (8-qubit: M=8, 16, 23; 12-qubit: M=35)
  2. Full cyclic U3 unitaries vs fallback rotations
  3. Hierarchical Quantum Attention-Fourier (QAF) controlled phase rotations
  4. Multi-basis expectation outputs (values bounded in [-1.0, 1.0])
  5. Determinism across identical seeds
  6. Freeze / unfreeze lifecycle enforcement
  7. Exact numerical round-trip serialization (save / load)
  8. Parameter-shift contrastive optimization sanity check (Mode B)
  9. Intrinsic Q-XAI subsystem von Neumann entropy extraction
  10. Finite-shot stochastic simulation (shots=1024)
  11. Geometric and spectral diagnostic profiling
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from nadiq.models.quantum.qvbe_v2 import (
    QVBEV2Config,
    QuantumVariationalBiomarkerEncoderV2,
)

try:
    import pennylane  # noqa: F401
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

requires_quantum = pytest.mark.skipif(
    not HAS_PENNYLANE,
    reason="PennyLane not installed; skipping QVBE-v2 tests.",
)


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 16)
    y = rng.randint(0, 2, size=20)
    return X, y


@requires_quantum
class TestQVBEV2:

    def test_output_shapes_and_bases(self, synthetic_data):
        X, y = synthetic_data
        
        # 8-qubit Z-only -> M=8
        cfg_z = QVBEV2Config(input_dim=16, n_qubits=8, n_layers=1, measurement_basis="z_only", seed=42)
        encoder_z = QuantumVariationalBiomarkerEncoderV2(cfg_z)
        Q_z = encoder_z.fit_transform(X, y)
        assert Q_z.shape == (20, 8)
        assert cfg_z.output_dim == 8

        # 8-qubit dual_basis -> M=16
        cfg_dual = QVBEV2Config(input_dim=16, n_qubits=8, n_layers=1, measurement_basis="dual_basis", seed=42)
        encoder_dual = QuantumVariationalBiomarkerEncoderV2(cfg_dual)
        Q_dual = encoder_dual.fit_transform(X, y)
        assert Q_dual.shape == (20, 16)
        assert cfg_dual.output_dim == 16

        # 8-qubit full -> M=23 (3*8 - 1)
        cfg_full_8 = QVBEV2Config(input_dim=16, n_qubits=8, n_layers=1, measurement_basis="full", seed=42)
        encoder_full_8 = QuantumVariationalBiomarkerEncoderV2(cfg_full_8)
        Q_full_8 = encoder_full_8.fit_transform(X, y)
        assert Q_full_8.shape == (20, 23)
        assert cfg_full_8.output_dim == 23

        # 12-qubit full -> M=35 (3*12 - 1)
        cfg_full_12 = QVBEV2Config(input_dim=16, n_qubits=12, n_layers=1, measurement_basis="full", seed=42)
        encoder_full_12 = QuantumVariationalBiomarkerEncoderV2(cfg_full_12)
        Q_full_12 = encoder_full_12.fit_transform(X, y)
        assert Q_full_12.shape == (20, 35)
        assert cfg_full_12.output_dim == 35

    def test_multi_basis_observables_bounded(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=2, measurement_basis="full", seed=42)
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        Q = encoder.fit_transform(X, y)
        # All Pauli expectation values must be in [-1.0, 1.0] (allowing small numerical tolerance)
        assert np.all(Q >= -1.0 - 1e-6)
        assert np.all(Q <= 1.0 + 1e-6)

    def test_determinism_across_identical_seeds(self, synthetic_data):
        X, y = synthetic_data
        cfg1 = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=2, seed=123)
        cfg2 = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=2, seed=123)

        enc1 = QuantumVariationalBiomarkerEncoderV2(cfg1)
        enc2 = QuantumVariationalBiomarkerEncoderV2(cfg2)

        Q1 = enc1.fit_transform(X, y)
        Q2 = enc2.fit_transform(X, y)

        np.testing.assert_allclose(Q1, Q2, atol=1e-10)

    def test_qaf_layer_activation(self, synthetic_data):
        X, y = synthetic_data
        cfg_with_qaf = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, use_qaf=True, seed=42)
        cfg_without_qaf = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, use_qaf=False, seed=42)

        enc_qaf = QuantumVariationalBiomarkerEncoderV2(cfg_with_qaf)
        enc_no_qaf = QuantumVariationalBiomarkerEncoderV2(cfg_without_qaf)

        Q_qaf = enc_qaf.fit_transform(X, y)
        Q_no_qaf = enc_no_qaf.fit_transform(X, y)

        # Output with QAF should differ from output without QAF
        assert not np.allclose(Q_qaf, Q_no_qaf, atol=1e-4)

    def test_u3_vs_ry_rz(self, synthetic_data):
        X, y = synthetic_data
        cfg_u3 = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, use_u3=True, seed=42)
        cfg_no_u3 = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, use_u3=False, seed=42)

        enc_u3 = QuantumVariationalBiomarkerEncoderV2(cfg_u3)
        enc_no_u3 = QuantumVariationalBiomarkerEncoderV2(cfg_no_u3)

        Q_u3 = enc_u3.fit_transform(X, y)
        Q_no_u3 = enc_no_u3.fit_transform(X, y)

        assert not np.allclose(Q_u3, Q_no_u3, atol=1e-4)

    def test_freeze_unfreeze_lifecycle(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, seed=42)
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        encoder.fit(X, y)
        encoder.freeze()

        assert encoder.is_frozen

        # Fitting frozen encoder must raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot fit a frozen"):
            encoder.fit(X, y)

        # Transform still works when frozen
        Q = encoder.transform(X[:5])
        assert Q.shape == (5, cfg.output_dim)

        # Unfreeze allows fit again
        encoder.unfreeze()
        assert not encoder.is_frozen
        encoder.fit(X, y)

    def test_save_and_load_exactness(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=2, measurement_basis="full", seed=42)
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        encoder.fit(X, y)
        encoder.freeze()

        Q_orig = encoder.transform(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            encoder.save(temp_path)
            loaded_encoder = QuantumVariationalBiomarkerEncoderV2.load(temp_path)

            assert loaded_encoder.is_frozen
            assert loaded_encoder.is_fitted
            assert loaded_encoder.config.n_qubits == cfg.n_qubits
            assert loaded_encoder.config.output_dim == cfg.output_dim

            Q_loaded = loaded_encoder.transform(X)
            np.testing.assert_allclose(Q_orig, Q_loaded, atol=1e-10)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parameter_shift_contrastive_optimization(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(
            input_dim=16,
            n_qubits=4,
            n_layers=1,
            mode="trainable",
            learning_rate=0.1,
            n_iter=2,
            seed=42
        )
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        initial_params = encoder.u3_params.copy()
        
        encoder.fit(X, y)
        
        # In trainable mode, parameters should be updated
        assert not np.allclose(initial_params, encoder.u3_params)
        Q = encoder.transform(X[:5])
        assert Q.shape == (5, cfg.output_dim)

    def test_intrinsic_qxai_von_neumann_entropy(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=2, seed=42)
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        encoder.fit(X, y)

        entropy = encoder.get_state_entropy(X[0])
        assert isinstance(entropy, float)
        assert entropy >= 0.0  # Entropy must be non-negative
        assert entropy <= math.log(2.0) + 1e-6  # For 1-qubit reduced subsystem, max entropy is ln(2) = 0.6931

    def test_finite_shot_simulation(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(
            input_dim=16,
            n_qubits=4,
            n_layers=1,
            measurement_basis="dual_basis",
            shots=1024,
            seed=42
        )
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        Q = encoder.fit_transform(X[:6], y[:6])
        assert Q.shape == (6, 8)
        assert np.all(Q >= -1.0)
        assert np.all(Q <= 1.0)

    def test_geometric_diagnostics(self, synthetic_data):
        X, y = synthetic_data
        cfg = QVBEV2Config(input_dim=16, n_qubits=4, n_layers=1, measurement_basis="full", seed=42)
        encoder = QuantumVariationalBiomarkerEncoderV2(cfg)
        Q = encoder.fit_transform(X, y)
        diag = encoder.get_diagnostics(Q)

        assert "effective_rank" in diag
        assert "condition_number" in diag
        assert "off_diagonal_similarity_mean" in diag
        assert "feature_variance_mean" in diag
        assert diag["effective_rank"] > 0.0
        assert not np.isnan(diag["condition_number"])

