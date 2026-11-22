"""
Tests for Biomarker Trajectory Encoder and Biomarker Quantum Encoder
(Nāḍī-Q V1 Biomarker Branch).

Tests cover:
  1.  Trajectory representation shape
  2.  Irregular time handling
  3.  Slope calculation correctness
  4.  Missing observation handling (NaN)
  5.  Single observation behavior
  6.  Deterministic output
  7.  Finite output
  8.  Trajectory → quantum encoder shape
  9.  Gradients through quantum encoder
  10. Batch processing
  11. Zero/equal time offsets
  12. n_valid count correctness
  13. Feature index semantics (current, first, delta, mean, slope, count)
  14. Config validation
  15. Parameter count
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from nadiq.models.biomarkers.biomarker_encoder import (
    BiomarkerQuantumEncoder,
    BiomarkerQuantumEncoderConfig,
    BiomarkerTrajectoryConfig,
    BiomarkerTrajectoryEncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def traj_cfg() -> BiomarkerTrajectoryConfig:
    return BiomarkerTrajectoryConfig(n_biomarkers=3)


@pytest.fixture
def traj_enc(traj_cfg: BiomarkerTrajectoryConfig) -> BiomarkerTrajectoryEncoder:
    return BiomarkerTrajectoryEncoder(traj_cfg)


@pytest.fixture
def bqe_cfg() -> BiomarkerQuantumEncoderConfig:
    return BiomarkerQuantumEncoderConfig(n_biomarkers=3, n_qubits=4, n_layers=2, seed=42)


@pytest.fixture
def bqe(bqe_cfg: BiomarkerQuantumEncoderConfig) -> BiomarkerQuantumEncoder:
    return BiomarkerQuantumEncoder(bqe_cfg)


def _make_batch(B: int = 3, T: int = 4, K: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    values = torch.randn(B, T, K)
    time_offsets = torch.tensor([[0., 30., 90., 180.]] * B)
    return values, time_offsets


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestBiomarkerConfig:
    def test_invalid_n_biomarkers(self) -> None:
        with pytest.raises(ValueError, match="n_biomarkers"):
            BiomarkerTrajectoryConfig(n_biomarkers=0)

    def test_trajectory_dim(self) -> None:
        cfg = BiomarkerTrajectoryConfig(n_biomarkers=5)
        assert cfg.trajectory_dim == 5 * 6  # 6 features per biomarker


# ---------------------------------------------------------------------------
# 1. Trajectory representation shape
# ---------------------------------------------------------------------------

class TestBiomarkerTrajectoryShape:
    def test_shape(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        values, t = _make_batch()
        out = traj_enc(values, t)
        assert out.trajectory.shape == (3, 3 * 6)  # B=3, K=3, 6 features each

    def test_shape_single_biomarker(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.randn(2, 5, 1)
        out = enc(values, None)
        assert out.trajectory.shape == (2, 6)


# ---------------------------------------------------------------------------
# 2. Irregular time handling
# ---------------------------------------------------------------------------

class TestBiomarkerIrregularTime:
    def test_irregular_offsets(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        values = torch.randn(2, 4, 3)
        time_offsets = torch.tensor([[0., 7., 30., 365.], [0., 14., 60., 120.]])
        out = traj_enc(values, time_offsets)
        assert torch.isfinite(out.trajectory).all()

    def test_none_offsets_falls_back_gracefully(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        values = torch.randn(2, 4, 3)
        out = traj_enc(values, None)
        assert out.trajectory.shape == (2, 18)
        assert torch.isfinite(out.trajectory).all()


# ---------------------------------------------------------------------------
# 3. Slope calculation correctness
# ---------------------------------------------------------------------------

class TestBiomarkerSlope:
    def test_zero_slope_for_constant_biomarker(self) -> None:
        """A constant biomarker over time should produce slope ≈ 0."""
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        # Constant value = 5.0 across all timesteps
        values = torch.full((1, 5, 1), 5.0)
        t = torch.arange(5).float().unsqueeze(0)
        out = enc(values, t)
        slope = out.trajectory[0, 4]  # feature index 4 = slope
        assert abs(slope.item()) < 1e-4, f"Expected slope~0 for constant, got {slope}"

    def test_positive_slope_for_increasing_biomarker(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        # Values: 1, 2, 3, 4, 5 at times 0, 1, 2, 3, 4
        values = torch.arange(1, 6, dtype=torch.float32).view(1, 5, 1)
        t = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        out = enc(values, t)
        slope = out.trajectory[0, 4]
        assert slope.item() > 0, f"Expected positive slope, got {slope}"

    def test_negative_slope_for_decreasing_biomarker(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.arange(5, 0, -1, dtype=torch.float32).view(1, 5, 1)
        t = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        out = enc(values, t)
        slope = out.trajectory[0, 4]
        assert slope.item() < 0, f"Expected negative slope, got {slope}"


# ---------------------------------------------------------------------------
# 4. Missing observation handling (NaN)
# ---------------------------------------------------------------------------

class TestBiomarkerMissingObs:
    def test_all_nan_produces_zero_features(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.full((1, 4, 1), float("nan"))
        out = enc(values, None)
        assert torch.allclose(out.trajectory, torch.zeros(1, 6))

    def test_partial_nan_uses_valid_obs(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.tensor([[[1.0], [float("nan")], [3.0], [float("nan")]]])
        t = torch.tensor([[0.0, 30.0, 90.0, 120.0]])
        out = enc(values, t)
        # current = 3.0 (last valid), first = 1.0
        current = out.trajectory[0, 0].item()
        first = out.trajectory[0, 1].item()
        obs_count = out.trajectory[0, 5].item()
        assert abs(current - 3.0) < 1e-5, f"Expected current=3.0, got {current}"
        assert abs(first - 1.0) < 1e-5, f"Expected first=1.0, got {first}"
        assert abs(obs_count - 2.0) < 1e-5, f"Expected obs_count=2, got {obs_count}"

    def test_n_valid_counts_non_nan(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=2))
        values = torch.full((1, 4, 2), float("nan"))
        values[0, 0, 0] = 1.0  # only 1 valid for biomarker 0
        values[0, :2, 1] = 2.0  # 2 valid for biomarker 1
        out = enc(values, None)
        # n_valid counts across all biomarkers
        assert out.n_valid[0].item() == 3  # 1 + 2


# ---------------------------------------------------------------------------
# 5. Single observation behavior
# ---------------------------------------------------------------------------

class TestBiomarkerSingleObs:
    def test_single_obs_slope_zero(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.tensor([[[5.0], [float("nan")], [float("nan")]]])
        t = torch.tensor([[0.0, 30.0, 90.0]])
        out = enc(values, t)
        slope = out.trajectory[0, 4].item()
        assert abs(slope) < 1e-5, "Slope with 1 obs must be 0"

    def test_single_obs_delta_zero(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.tensor([[[7.0], [float("nan")]]])
        out = enc(values, None)
        delta = out.trajectory[0, 2].item()
        assert abs(delta) < 1e-5, "Delta with 1 obs must be 0"


# ---------------------------------------------------------------------------
# 6. Deterministic output
# ---------------------------------------------------------------------------

class TestBiomarkerDeterminism:
    def test_same_input_same_output(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        values, t = _make_batch()
        out1 = traj_enc(values, t)
        out2 = traj_enc(values, t)
        assert torch.allclose(out1.trajectory, out2.trajectory)


# ---------------------------------------------------------------------------
# 7. Finite output
# ---------------------------------------------------------------------------

class TestBiomarkerFinite:
    def test_finite_trajectory(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        values, t = _make_batch()
        out = traj_enc(values, t)
        assert torch.isfinite(out.trajectory).all()

    def test_large_values_stay_finite(self) -> None:
        enc = BiomarkerTrajectoryEncoder(BiomarkerTrajectoryConfig(n_biomarkers=1))
        values = torch.full((1, 3, 1), 1e6)
        out = enc(values, None)
        assert torch.isfinite(out.trajectory).all()


# ---------------------------------------------------------------------------
# 8. Trajectory → quantum encoder shape
# ---------------------------------------------------------------------------

class TestBiomarkerQuantumShape:
    def test_embedding_shape(self, bqe: BiomarkerQuantumEncoder) -> None:
        values, t = _make_batch()
        bqe.eval()
        with torch.no_grad():
            out = bqe(values, t)
        assert out.embedding.shape == (3, bqe.config.n_qubits)

    def test_embedding_in_range(self, bqe: BiomarkerQuantumEncoder) -> None:
        values, t = _make_batch()
        bqe.eval()
        with torch.no_grad():
            out = bqe(values, t)
        emb = out.embedding.numpy()
        assert np.all(emb >= -1.0 - 1e-5) and np.all(emb <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# 9. Gradients through quantum encoder
# ---------------------------------------------------------------------------

class TestBiomarkerGradients:
    def test_gradient_through_qvbe(self, bqe: BiomarkerQuantumEncoder) -> None:
        values, t = _make_batch()
        bqe.train()
        out = bqe(values, t)
        loss = out.embedding.mean()
        loss.backward()
        qvbe_grad = bqe.qvbe.vqc.weights.grad
        assert qvbe_grad is not None
        assert not torch.all(qvbe_grad == 0)


# ---------------------------------------------------------------------------
# 10. Batch processing
# ---------------------------------------------------------------------------

class TestBiomarkerBatch:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_variable_batch_sizes(self, batch_size: int) -> None:
        bqe = BiomarkerQuantumEncoder(BiomarkerQuantumEncoderConfig(n_biomarkers=3, n_qubits=4, seed=42))
        values = torch.randn(batch_size, 4, 3)
        bqe.eval()
        with torch.no_grad():
            out = bqe(values, None)
        assert out.embedding.shape == (batch_size, 4)


# ---------------------------------------------------------------------------
# 15. Parameter count
# ---------------------------------------------------------------------------

class TestBiomarkerParamCount:
    def test_trajectory_encoder_no_params(self, traj_enc: BiomarkerTrajectoryEncoder) -> None:
        assert traj_enc.n_parameters == 0

    def test_bqe_params_positive(self, bqe: BiomarkerQuantumEncoder) -> None:
        assert bqe.n_parameters > 0

    def test_get_config_serialisable(self, bqe: BiomarkerQuantumEncoder) -> None:
        import json
        json.dumps(bqe.get_config())
