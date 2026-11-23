"""
Continual Temporal Clinical Latent Model (CTCLM) — Nāḍī-Q V1.

CTCLM is the temporal representation layer between modality encoders (QVBE)
and the downstream fusion/prediction layer (QCDF).

Scientific Position
-------------------
CTCLM V1 is a baseline temporal representation mechanism. It does NOT
establish continual learning, catastrophic-forgetting resistance,
quantum advantage, or clinical superiority. The "continual" terminology
reflects architectural and research intent. Future research variants may
investigate: temporal transformers, continuous-time models, neural ODEs,
state-space models, explicit representation-drift modeling.

Architecture (V1)
-----------------
Input:
    x            : R^(B × T × D)  — per-timestep embeddings (e.g., from QVBE)
    time_offsets : R^(B × T)      — elapsed time (Δt from first observation)
    padding_mask : bool (B × T)   — True at VALID positions (False = PAD)

Forward pass:
    1. Time embedding:   Δt ∈ R → R^time_embedding_dim  via 2-layer MLP
    2. Concatenation:    [x_t ; time_t] → R^(D + time_embedding_dim)
    3. Optional projection if D + time_embedding_dim ≠ latent_dim
    4. GRU (batch_first) over the concatenated sequence
    5. Extract last-valid-timestep hidden state as h_final

Temporal Semantics
------------------
time_offsets must encode RELATIVE elapsed time (Δt).
CTCLM does NOT sort sequences. Chronological ordering is the
responsibility of the upstream TemporalLeakageChecker / PatientTimeline.

Masking
-------
padding_mask follows PyTorch convention used by pack_padded_sequence:
    True  = VALID observation
    False = PAD (should not influence the representation)
Implemented via torch.nn.utils.rnn.pack_padded_sequence.

Dataset/Disease Agnosticism
----------------------------
No TADPOLE, ADNI, or disease-specific logic exists in this module.
The module operates on arbitrary (B, T, D) tensors.

Output
------
CTCLMOutput:
    .sequence   : R^(B × T × latent_dim)  — per-timestep hidden states
    .h_final    : R^(B × latent_dim)       — last VALID timestep hidden state
    .seq_lengths: (B,) int64 tensor        — actual sequence lengths used

References
----------
- Chung et al. (2014) — Empirical Evaluation of Gated Recurrent Neural Networks
- Horn et al. (2020) — Set Functions for Time Series (time embedding idea)
- Rubanova et al. (2019) — Latent ODEs for Irregularly-Sampled Time Series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = [
    "CTCLMConfig",
    "CTCLMOutput",
    "ContinualTemporalClinicalLatentModel",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CTCLMConfig:
    """
    Full configuration for a CTCLM instance.

    Attributes:
        input_dim        : Dimensionality of each timestep embedding (QVBE output = n_qubits).
        latent_dim       : Hidden state dimension of the GRU.
        time_embedding_dim: Dimensionality of the time-encoding MLP output.
        num_layers       : Number of stacked GRU layers.
        dropout          : Dropout probability between GRU layers (only if num_layers > 1).
        seed             : Random seed for deterministic parameter initialization.
    """

    input_dim: int
    latent_dim: int = 32
    time_embedding_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.time_embedding_dim < 1:
            raise ValueError(f"time_embedding_dim must be >= 1, got {self.time_embedding_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def gru_input_dim(self) -> int:
        """Effective input dimension to GRU after time concatenation."""
        return self.input_dim + self.time_embedding_dim

    @property
    def n_time_mlp_params(self) -> int:
        """Parameters in the time-encoding MLP (1 hidden → output)."""
        # Input(1) → Hidden(time_embedding_dim) → Output(time_embedding_dim)
        hidden = self.time_embedding_dim
        return (1 * hidden + hidden) + (hidden * hidden + hidden)

    @property
    def n_gru_params(self) -> int:
        """Approximate GRU parameter count."""
        D = self.gru_input_dim
        H = self.latent_dim
        # 3 gates × (input weights + hidden weights + bias×2) per layer
        return self.num_layers * 3 * (D * H + H * H + 2 * H)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class CTCLMOutput:
    """
    Structured output of CTCLM forward pass.

    Attributes:
        sequence    : Per-timestep GRU hidden states, shape (B, T, latent_dim).
                      Padded positions contain the GRU's carry-over state and
                      should be masked by downstream consumers using padding_mask.
        h_final     : Final valid hidden state per patient, shape (B, latent_dim).
                      This is extracted at the LAST VALID timestep, not the padded one.
        seq_lengths : Actual sequence lengths per patient, shape (B,).
    """

    sequence: torch.Tensor    # (B, T, latent_dim)
    h_final: torch.Tensor     # (B, latent_dim)
    seq_lengths: torch.Tensor  # (B,) int64


# ---------------------------------------------------------------------------
# Time Encoding MLP
# ---------------------------------------------------------------------------


class _TimeEncoder(nn.Module):
    """
    Encode a scalar Δt value into a time embedding vector.

    Architecture: Linear(1 → H) → ReLU → Linear(H → H)
    Input  : R^1 (elapsed time in arbitrary units)
    Output : R^time_embedding_dim
    """

    def __init__(self, time_embedding_dim: int) -> None:
        super().__init__()
        H = time_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(1, H),
            nn.ReLU(),
            nn.Linear(H, H),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: Elapsed time tensor, shape (B, T) or (B, T, 1).
        Returns:
            time embeddings, shape (B, T, time_embedding_dim).
        """
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)  # (B, T, 1)
        return self.net(dt)  # (B, T, time_embedding_dim)


# ---------------------------------------------------------------------------
# CTCLM
# ---------------------------------------------------------------------------


class ContinualTemporalClinicalLatentModel(nn.Module):
    """
    Continual Temporal Clinical Latent Model (CTCLM) V1.

    Temporal representation layer for irregularly-sampled clinical sequences.
    Accepts per-timestep embeddings from QVBE (or any upstream encoder) and
    produces a temporal latent representation suitable for downstream prediction.

    This module is dataset-agnostic and disease-agnostic.

    Usage:
        config = CTCLMConfig(input_dim=4, latent_dim=32, time_embedding_dim=8)
        ctclm = ContinualTemporalClinicalLatentModel(config)

        # x: (B, T, D) — e.g., D = QVBE n_qubits = 4
        # time_offsets: (B, T) — elapsed time from first visit in each sequence
        # padding_mask: (B, T) bool — True = valid, False = pad
        out = ctclm(x, time_offsets=time_offsets, padding_mask=padding_mask)

        h_final = out.h_final   # (B, latent_dim) — patient-level latent state
        seq     = out.sequence  # (B, T, latent_dim) — full trajectory
    """

    def __init__(self, config: CTCLMConfig) -> None:
        super().__init__()
        self.config = config

        with torch.random.fork_rng():
            if config.seed is not None:
                torch.manual_seed(config.seed)

            # Time encoder: scalar Δt → R^time_embedding_dim
            self.time_encoder = _TimeEncoder(config.time_embedding_dim)

            # GRU temporal model: [x ; time_emb] → hidden states
            gru_dropout = config.dropout if config.num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=config.gru_input_dim,
                hidden_size=config.latent_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=gru_dropout,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        time_offsets: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> CTCLMOutput:
        """
        Encode a batch of variable-length clinical sequences.

        Args:
            x             : Sequence of per-timestep embeddings.
                            Shape: (B, T, input_dim).
                            May contain values from QVBE or any other encoder.
            time_offsets  : Elapsed time for each timestep relative to the
                            first observation in the sequence.
                            Shape: (B, T). Units are caller-defined (e.g. days).
                            If None, zero offsets are used (uniform spacing assumed).
            padding_mask  : Boolean mask. True = VALID, False = PAD.
                            Shape: (B, T).
                            If None, all positions are considered valid.

        Returns:
            CTCLMOutput with .sequence (B, T, latent_dim) and
            .h_final (B, latent_dim) at the last valid timestep.
        """
        B, T, D = x.shape
        device = x.device

        # ── 1. Build time_offsets ─────────────────────────────────────
        if time_offsets is None:
            # Fallback: treat as uniformly spaced [0, 1, 2, ..., T-1]
            time_offsets = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)
            time_offsets = time_offsets.expand(B, T)

        # Replace NaN in offsets with 0
        time_offsets = torch.nan_to_num(time_offsets, nan=0.0)

        # ── 2. Time embedding: (B, T) → (B, T, time_embedding_dim) ───
        time_emb = self.time_encoder(time_offsets.float())  # (B, T, time_embedding_dim)

        # ── 3. Replace NaN in x with 0 (safety guard) ─────────────────
        x = torch.nan_to_num(x, nan=0.0)

        # ── 4. Concatenate: [x ; time_emb] → (B, T, gru_input_dim) ───
        x_cat = torch.cat([x, time_emb], dim=-1)  # (B, T, D + time_embedding_dim)

        # ── 5. Compute sequence lengths from padding_mask ─────────────
        if padding_mask is not None:
            # padding_mask: True = valid. Sum along T dim.
            seq_lengths = padding_mask.long().sum(dim=1).clamp(min=1)  # (B,)
        else:
            seq_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        # ── 6. Pack → GRU → unpack ────────────────────────────────────
        # Sort by descending length for pack_padded_sequence
        seq_lengths_cpu = seq_lengths.cpu()
        sorted_lengths, sort_idx = seq_lengths_cpu.sort(descending=True)
        _, unsort_idx = sort_idx.sort()

        x_sorted = x_cat[sort_idx]  # sort batch
        packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.gru(packed)
        sequence_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )
        sequence = sequence_sorted[unsort_idx]  # restore original order

        # ── 7. Extract h_final = hidden state at last VALID timestep ──
        # Index is (seq_length - 1) per sample
        last_idx = (seq_lengths - 1).clamp(min=0)  # (B,)
        # Gather: sequence[:, last_idx_per_sample, :]
        idx = last_idx.view(B, 1, 1).expand(B, 1, self.config.latent_dim).to(device)
        h_final = sequence.gather(1, idx).squeeze(1)  # (B, latent_dim)

        return CTCLMOutput(
            sequence=sequence,
            h_final=h_final,
            seq_lengths=seq_lengths,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return serialisable configuration dictionary."""
        cfg = self.config
        return {
            "input_dim": cfg.input_dim,
            "latent_dim": cfg.latent_dim,
            "time_embedding_dim": cfg.time_embedding_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "seed": cfg.seed,
            "gru_input_dim": cfg.gru_input_dim,
            "n_parameters_actual": self.n_parameters,
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"ContinualTemporalClinicalLatentModel("
            f"input_dim={c.input_dim}, latent_dim={c.latent_dim}, "
            f"time_emb={c.time_embedding_dim}, "
            f"num_layers={c.num_layers}, params={self.n_parameters})"
        )
