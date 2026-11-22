"""
MRI Quantum Encoder — Nāḍī-Q V1.

Wraps the MRI representation pipeline:
    3D T1 MRI features (volume)
        ↓
    BetaVAE (encoder path only for inference)
        ↓
    latent z ∈ ℝ^latent_dim
        ↓
    QVBE (quantum variational encoding)
        ↓
    MRI quantum embedding ∈ ℝ^n_qubits

Scientific Position
-------------------
- This module is the full MRI branch of Nāḍī-Q V1.
- QVBE currently uses RY/RZ/RY rotations. U3-based circuit is a documented
  future variant and has NOT been falsely claimed here.
- This is a prototype implementation requiring real-data empirical validation.
- Does NOT establish clinical superiority or quantum advantage.

Architecture note
-----------------
BetaVAE.latent_dim MUST equal QVBEConfig.input_dim so that the latent
vector z can be directly fed into the QVBE projection layer.
MRIQuantumEncoderConfig enforces this constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nadiq.models.mri.bvae import BetaVAE, BetaVAEConfig, BetaVAEOutput
from nadiq.models.quantum.qvbe import (
    QVBEConfig,
    QVBEOutput,
    QuantumVariationalBiomarkerEncoder,
)

__all__ = [
    "MRIQuantumEncoderConfig",
    "MRIQuantumEncoderOutput",
    "MRIQuantumEncoder",
]


@dataclass
class MRIQuantumEncoderConfig:
    """
    Unified configuration for the MRI → β-VAE → QVBE pipeline.

    Attributes:
        mri_input_shape: Dimensionality of the 3D MRI input, e.g. (C, D, H, W)
        latent_dim     : β-VAE latent size AND QVBE input_dim (must match).
        beta           : KL divergence weight for β-VAE.
        n_qubits       : Number of qubits in the QVBE quantum circuit.
        n_layers       : VQC depth (number of variational ansatz repetitions).
        backend        : PennyLane device string.
        seed           : Shared seed for deterministic initialization.
    """

    mri_input_shape: tuple[int, int, int, int]
    latent_dim: int = 16
    beta: float = 4.0
    n_qubits: int = 4
    n_layers: int = 2
    backend: str = "default.qubit"
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.mri_input_shape) != 4:
            raise ValueError(f"mri_input_shape must be (C, D, H, W), got {self.mri_input_shape}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")


@dataclass
class MRIQuantumEncoderOutput:
    """
    Output of the MRI quantum encoder pipeline.

    Attributes:
        vae_output   : Full BetaVAEOutput (z, mu, logvar, x_hat, losses).
        qvbe_output  : Full QVBEOutput (embedding, pre_quantum, vqc_params).
        embedding    : Final MRI quantum embedding, shape (B, n_qubits).
                       Alias for qvbe_output.embedding.
    """

    vae_output: BetaVAEOutput
    qvbe_output: QVBEOutput
    embedding: torch.Tensor   # (B, n_qubits)


class MRIQuantumEncoder(nn.Module):
    """
    Full MRI representation branch for Nāḍī-Q V1.

    Pipeline:
        3D MRI features → BetaVAE → latent z → QVBE → MRI quantum embedding

    The BetaVAE latent_dim is always equal to the QVBE input_dim.
    The QVBE uses the existing RY/RZ/RY variational quantum circuit.
    A U3-based circuit variant is a documented future replacement.

    Usage (training, full loss):
        encoder = MRIQuantumEncoder(config)
        encoder.train()
        x = torch.randn(B, *mri_input_shape)
        out = encoder(x)
        out.vae_output.total_loss.backward()  # β-VAE reconstruction + KL

    Usage (inference, embedding only):
        encoder.eval()
        with torch.no_grad():
            out = encoder(x)
        embedding = out.embedding  # (B, n_qubits)
    """

    def __init__(self, config: MRIQuantumEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.bvae = BetaVAE(
            BetaVAEConfig(
                input_shape=config.mri_input_shape,
                latent_dim=config.latent_dim,
                beta=config.beta,
                seed=config.seed,
            )
        )

        # QVBE takes the β-VAE latent as its input
        self.qvbe = QuantumVariationalBiomarkerEncoder(
            QVBEConfig(
                input_dim=config.latent_dim,   # = bvae.latent_dim
                n_qubits=config.n_qubits,
                n_layers=config.n_layers,
                backend=config.backend,
                seed=config.seed,
            )
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> MRIQuantumEncoderOutput:
        """
        Encode MRI features into quantum embedding.

        Args:
            x: MRI feature tensor, shape (B, *mri_input_shape).
        Returns:
            MRIQuantumEncoderOutput with .embedding of shape (B, n_qubits).
        """
        vae_out = self.bvae(x)
        qvbe_out = self.qvbe(vae_out.z)
        return MRIQuantumEncoderOutput(
            vae_output=vae_out,
            qvbe_output=qvbe_out,
            embedding=qvbe_out.embedding,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mri_input_shape": self.config.mri_input_shape,
            "latent_dim": self.config.latent_dim,
            "beta": self.config.beta,
            "n_qubits": self.config.n_qubits,
            "n_layers": self.config.n_layers,
            "backend": self.config.backend,
            "seed": self.config.seed,
            "n_parameters_bvae": self.bvae.n_parameters,
            "n_parameters_qvbe": self.qvbe.n_parameters,
            "n_parameters_total": self.n_parameters,
            "circuit_note": (
                "Current VQC uses RY/RZ/RY rotations. "
                "U3-based circuit is a documented future variant."
            ),
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"MRIQuantumEncoder("
            f"mri_input={c.mri_input_shape}, "
            f"latent={c.latent_dim}, "
            f"n_qubits={c.n_qubits}, "
            f"params={self.n_parameters})"
        )
