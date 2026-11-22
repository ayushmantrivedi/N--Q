"""
Biomarker Trajectory Encoder — Nāḍī-Q V1.

Scientific Position
-------------------
This module implements the biomarker representation branch of Nāḍī-Q V1.
Biomarkers are NOT treated as ordinary static features. Longitudinal
accumulation and rate-of-change are scientifically meaningful signals.
This is a prototype implementation requiring empirical validation.
Does NOT establish clinical superiority or quantum advantage.

Role in Nāḍī-Q Architecture
----------------------------
    Longitudinal biomarker measurements
        ↓
    BiomarkerTrajectoryEncoder (deterministic trajectory features)
        ↓
    trajectory representation ∈ ℝ^trajectory_dim
        ↓
    QVBE (quantum encoding)
        ↓
    biomarker quantum embedding ∈ ℝ^n_qubits

Trajectory Features (V1)
------------------------
For each biomarker, given observations at irregular times t₀ < t₁ < … < tₙ:

    current_value   = value at the last valid observation
    first_value     = value at the first valid observation
    delta           = current_value - first_value
    mean_value      = mean of valid observations
    slope           = linear regression slope over (Δt, value) pairs
    obs_count       = number of valid (non-missing) observations

If only one observation is available:
    delta = 0.0, slope = 0.0

If no observations are available:
    all features = 0.0

The representation is concatenated over all biomarkers:
    trajectory_dim = n_biomarkers × n_features_per_biomarker  (default 6)

This module is intentionally deterministic and interpretable.
No trainable parameters are used in trajectory extraction.
Trainable parameters exist ONLY in the QVBE quantum encoder.

Irregular Interval Handling
----------------------------
    Never assume t₁ - t₀ == t₂ - t₁.
    Slopes are computed with explicit Δt values.
    A minimum of two observations at different time points is required
    for a finite slope; otherwise slope = 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nadiq.models.quantum.qvbe import (
    QVBEConfig,
    QVBEOutput,
    QuantumVariationalBiomarkerEncoder,
)

__all__ = [
    "BiomarkerTrajectoryConfig",
    "BiomarkerTrajectoryOutput",
    "BiomarkerTrajectoryEncoder",
    "BiomarkerQuantumEncoderConfig",
    "BiomarkerQuantumEncoderOutput",
    "BiomarkerQuantumEncoder",
]


# ---------------------------------------------------------------------------
# Trajectory Feature Extraction
# ---------------------------------------------------------------------------

# Number of features extracted per biomarker (fixed for V1)
_FEATURES_PER_BIOMARKER = 6  # current, first, delta, mean, slope, obs_count


@dataclass
class BiomarkerTrajectoryConfig:
    """
    Configuration for the biomarker trajectory extractor.

    Attributes:
        n_biomarkers   : Number of longitudinal biomarker channels.
        missing_fill   : Value used to fill missing biomarker observations.
                         NaN is the preferred convention; set to 0.0 for masked input.
    """

    n_biomarkers: int
    missing_fill: float = float("nan")

    def __post_init__(self) -> None:
        if self.n_biomarkers < 1:
            raise ValueError(f"n_biomarkers must be >= 1, got {self.n_biomarkers}")

    @property
    def trajectory_dim(self) -> int:
        """Dimensionality of the output trajectory feature vector."""
        return self.n_biomarkers * _FEATURES_PER_BIOMARKER


@dataclass
class BiomarkerTrajectoryOutput:
    """
    Output of the biomarker trajectory extractor.

    Attributes:
        trajectory : Trajectory feature tensor, shape (B, trajectory_dim).
                     For each biomarker: [current, first, delta, mean, slope, obs_count].
        n_valid    : Number of valid (non-missing) observations per sample, shape (B,).
    """

    trajectory: torch.Tensor   # (B, trajectory_dim)
    n_valid: torch.Tensor      # (B,) int


class BiomarkerTrajectoryEncoder(nn.Module):
    """
    Deterministic trajectory feature extractor for longitudinal biomarkers.

    NO trainable parameters. Produces a compact, interpretable feature vector
    from irregular longitudinal biomarker sequences.

    Input:
        values      : (B, T, n_biomarkers) — biomarker measurements over time.
                      NaN = missing observation for that biomarker at that visit.
        time_offsets: (B, T) — elapsed time from first visit (e.g., days).
                      If None, integer indices [0, 1, ..., T-1] are used.

    Output:
        trajectory  : (B, trajectory_dim)
                      = (B, n_biomarkers × 6)

    Features per biomarker:
        [0] current_value  — last valid observation
        [1] first_value    — first valid observation
        [2] delta          — current - first
        [3] mean_value     — mean of valid observations
        [4] slope          — Δvalue / Δt linear regression slope
        [5] obs_count      — number of valid observations
    """

    def __init__(self, config: BiomarkerTrajectoryConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def n_parameters(self) -> int:
        return 0  # deterministic

    def forward(
        self,
        values: torch.Tensor,
        time_offsets: torch.Tensor | None = None,
    ) -> BiomarkerTrajectoryOutput:
        """
        Extract trajectory features from longitudinal biomarker sequences.

        Args:
            values      : (B, T, n_biomarkers) — raw biomarker values.
                          Use NaN to indicate missing observations.
            time_offsets: (B, T) — relative elapsed time from first visit.
                          If None, assumes [0, 1, ..., T-1].

        Returns:
            BiomarkerTrajectoryOutput with .trajectory of shape (B, trajectory_dim).
        """
        B, T, K = values.shape
        assert K == self.config.n_biomarkers, (
            f"Expected {self.config.n_biomarkers} biomarkers, got {K}"
        )

        device = values.device

        if time_offsets is None:
            time_offsets = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)
            time_offsets = time_offsets.expand(B, T)

        time_offsets = time_offsets.float()

        # Output buffer: (B, K, 6)
        feat = torch.zeros(B, K, _FEATURES_PER_BIOMARKER, device=device)

        # valid mask: (B, T, K) — True where not NaN
        valid = ~torch.isnan(values)

        for k in range(K):
            v_k = values[:, :, k]         # (B, T) — biomarker k values
            m_k = valid[:, :, k].float()  # (B, T) — 1 = valid, 0 = missing

            n_valid = m_k.sum(dim=1)      # (B,) — valid count per sample

            # Safe: replace NaN with 0 for masked aggregations
            v_safe = torch.nan_to_num(v_k, nan=0.0)

            # --- obs_count ---
            feat[:, k, 5] = n_valid

            # For samples with at least 1 valid observation:
            has_any = (n_valid >= 1)

            if has_any.any():
                # --- mean_value ---
                sum_v = (v_safe * m_k).sum(dim=1)
                mean_v = torch.where(has_any, sum_v / n_valid.clamp(min=1), torch.zeros_like(sum_v))
                feat[:, k, 3] = mean_v

                # --- first_value: earliest valid obs ---
                # Find index of first valid observation per sample
                first_idx = self._first_valid_idx(m_k, T, device)
                first_val = v_safe.gather(1, first_idx.unsqueeze(1)).squeeze(1)
                first_val = torch.where(has_any, first_val, torch.zeros_like(first_val))
                feat[:, k, 1] = first_val

                # --- current_value: latest valid obs ---
                last_idx = self._last_valid_idx(m_k, T, device)
                last_val = v_safe.gather(1, last_idx.unsqueeze(1)).squeeze(1)
                last_val = torch.where(has_any, last_val, torch.zeros_like(last_val))
                feat[:, k, 0] = last_val

                # --- delta ---
                feat[:, k, 2] = torch.where(has_any, last_val - first_val, torch.zeros_like(last_val))

            # --- slope: requires >= 2 valid obs at different times ---
            has_two = (n_valid >= 2)
            if has_two.any():
                slope = self._compute_slope(v_safe, m_k, time_offsets, has_two)
                feat[:, k, 4] = slope

        trajectory = feat.view(B, -1)  # (B, K * 6)

        # Total valid observations across all biomarkers (for diagnostics)
        n_valid_total = valid.float().sum(dim=[1, 2]).long()

        return BiomarkerTrajectoryOutput(
            trajectory=trajectory,
            n_valid=n_valid_total,
        )

    @staticmethod
    def _first_valid_idx(mask: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        """Return the index of the first valid (True) position per row."""
        # mask: (B, T) float
        # Reverse cumsum trick: first nonzero position
        idx_range = torch.arange(T, device=device).float().unsqueeze(0)  # (1, T)
        large = T * (1 - mask) + idx_range * mask  # non-valid get large value
        return large.long().argmin(dim=1).clamp(0, T - 1)

    @staticmethod
    def _last_valid_idx(mask: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        """Return the index of the last valid (True) position per row."""
        idx_range = torch.arange(T, device=device).float().unsqueeze(0)
        neg_large = -T * (1 - mask) + idx_range * mask
        return neg_large.long().argmax(dim=1).clamp(0, T - 1)

    @staticmethod
    def _compute_slope(
        values: torch.Tensor,
        mask: torch.Tensor,
        time: torch.Tensor,
        has_two: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample linear regression slope (Δv / Δt) for valid observations.

        Uses the closed-form ordinary least-squares formula:
            slope = Σ(Δtᵢ × Δvᵢ) / Σ(Δtᵢ²)
        where Δtᵢ = tᵢ - mean_t  and  Δvᵢ = vᵢ - mean_v.
        """
        B = values.shape[0]
        n_valid = mask.sum(dim=1).clamp(min=1)  # (B,)

        # Masked mean
        t_sum = (time * mask).sum(dim=1)
        v_sum = (values * mask).sum(dim=1)
        t_mean = t_sum / n_valid
        v_mean = v_sum / n_valid

        dt = time - t_mean.unsqueeze(1)   # (B, T)
        dv = values - v_mean.unsqueeze(1)  # (B, T)

        numerator   = (dt * dv * mask).sum(dim=1)   # Σ Δtᵢ×Δvᵢ
        denominator = (dt * dt * mask).sum(dim=1)   # Σ Δtᵢ²

        slope = torch.where(
            denominator.abs() > 1e-8,
            numerator / denominator,
            torch.zeros_like(numerator),
        )

        return torch.where(has_two, slope, torch.zeros_like(slope))


# ---------------------------------------------------------------------------
# Biomarker Quantum Encoder
# ---------------------------------------------------------------------------


@dataclass
class BiomarkerQuantumEncoderConfig:
    """
    Unified config for biomarker trajectory → QVBE pipeline.

    Attributes:
        n_biomarkers       : Number of longitudinal biomarker channels.
        n_qubits           : QVBE qubit count.
        n_layers           : QVBE circuit depth.
        backend            : PennyLane device string.
        seed               : Shared seed for deterministic initialization.
    """

    n_biomarkers: int
    n_qubits: int = 4
    n_layers: int = 2
    backend: str = "default.qubit"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_biomarkers < 1:
            raise ValueError(f"n_biomarkers must be >= 1, got {self.n_biomarkers}")

    @property
    def trajectory_dim(self) -> int:
        return self.n_biomarkers * _FEATURES_PER_BIOMARKER


@dataclass
class BiomarkerQuantumEncoderOutput:
    """
    Output of the biomarker quantum encoder pipeline.

    Attributes:
        trajectory_output : BiomarkerTrajectoryOutput (features + n_valid).
        qvbe_output       : QVBEOutput (embedding, pre_quantum, vqc_params).
        embedding         : Final biomarker quantum embedding, shape (B, n_qubits).
    """

    trajectory_output: BiomarkerTrajectoryOutput
    qvbe_output: QVBEOutput
    embedding: torch.Tensor   # (B, n_qubits)


class BiomarkerQuantumEncoder(nn.Module):
    """
    Biomarker representation branch for Nāḍī-Q V1.

    Pipeline:
        longitudinal biomarker measurements (B, T, n_biomarkers)
            ↓
        BiomarkerTrajectoryEncoder (deterministic, no weights)
            ↓
        trajectory features (B, trajectory_dim)
            ↓
        QVBE (quantum encoding, trainable)
            ↓
        biomarker quantum embedding (B, n_qubits)

    CTCLM is NOT used in this branch. CTCLM is the kinematics branch.
    Biomarkers use deterministic trajectory features instead.

    Usage:
        config = BiomarkerQuantumEncoderConfig(n_biomarkers=5, n_qubits=4)
        bqe = BiomarkerQuantumEncoder(config)
        values = torch.randn(B, T, 5)    # T visits, 5 biomarkers
        values[0, 2:, :] = float('nan') # missing observations
        t_offsets = torch.tensor(...)   # (B, T) elapsed days
        out = bqe(values, t_offsets)
        embedding = out.embedding        # (B, 4)
    """

    def __init__(self, config: BiomarkerQuantumEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.trajectory_encoder = BiomarkerTrajectoryEncoder(
            BiomarkerTrajectoryConfig(n_biomarkers=config.n_biomarkers)
        )

        self.qvbe = QuantumVariationalBiomarkerEncoder(
            QVBEConfig(
                input_dim=config.trajectory_dim,
                n_qubits=config.n_qubits,
                n_layers=config.n_layers,
                backend=config.backend,
                seed=config.seed,
            )
        )

    @property
    def n_parameters(self) -> int:
        # Trajectory encoder has 0 trainable params; QVBE has all of them
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        values: torch.Tensor,
        time_offsets: torch.Tensor | None = None,
    ) -> BiomarkerQuantumEncoderOutput:
        """
        Encode longitudinal biomarker sequences into quantum embeddings.

        Args:
            values      : (B, T, n_biomarkers) — longitudinal biomarker values.
                          NaN = missing observation.
            time_offsets: (B, T) — elapsed time. If None, uses [0, 1, ..., T-1].

        Returns:
            BiomarkerQuantumEncoderOutput with .embedding of shape (B, n_qubits).
        """
        traj_out = self.trajectory_encoder(values, time_offsets)
        qvbe_out = self.qvbe(traj_out.trajectory)
        return BiomarkerQuantumEncoderOutput(
            trajectory_output=traj_out,
            qvbe_output=qvbe_out,
            embedding=qvbe_out.embedding,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "n_biomarkers": self.config.n_biomarkers,
            "trajectory_dim": self.config.trajectory_dim,
            "n_qubits": self.config.n_qubits,
            "n_layers": self.config.n_layers,
            "backend": self.config.backend,
            "seed": self.config.seed,
            "n_parameters_trajectory": 0,
            "n_parameters_qvbe": self.qvbe.n_parameters,
            "n_parameters_total": self.n_parameters,
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"BiomarkerQuantumEncoder("
            f"n_biomarkers={c.n_biomarkers}, "
            f"trajectory_dim={c.trajectory_dim}, "
            f"n_qubits={c.n_qubits}, "
            f"params={self.n_parameters})"
        )
