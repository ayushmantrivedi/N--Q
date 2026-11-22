"""
Beta-VAE for MRI Latent Representation — Nāḍī-Q V1.

Scientific Position
-------------------
This module implements the MRI representation branch of Nāḍī-Q V1.
It is an architectural/prototype implementation. It does NOT establish
clinical superiority, quantum advantage, or validated diagnostic performance.
Real-data empirical validation is required before any clinical claims.

Role in Nāḍī-Q Architecture
----------------------------
    3D T1 MRI input
        ↓
    BetaVAE (3D encoder + reparameterization)
        ↓
    latent vector  μ ∈ ℝ^latent_dim
        ↓
    QVBE (via MRIQuantumEncoder wrapper)
        ↓
    MRI quantum embedding ∈ ℝ^n_qubits

Architecture
------------
Encoder : 3D Conv → ReLU → MaxPool3d → Flatten → Linear(μ, logσ²)
Decoder : Linear → Unflatten → 3D ConvTranspose → Sigmoid/Linear
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BetaVAEConfig",
    "BetaVAEOutput",
    "BetaVAE",
]


@dataclass
class BetaVAEConfig:
    """
    Configuration for the 3D MRI β-VAE.

    Attributes:
        input_shape : Dimensions of the 3D MRI input, e.g. (C, D, H, W)
        latent_dim  : Latent space dimensionality (μ/logσ² output size).
        beta        : KL divergence weight.
        seed        : Random seed for deterministic parameter initialization.
    """
    input_shape: tuple[int, int, int, int]
    latent_dim: int = 16
    beta: float = 4.0
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.input_shape) != 4:
            raise ValueError(f"input_shape must be (C, D, H, W), got {self.input_shape}")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")


@dataclass
class BetaVAEOutput:
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    x_hat: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    total_loss: torch.Tensor


class BetaVAE(nn.Module):
    """
    Compact 3D Beta-Variational Autoencoder for MRI latent representation.
    """
    def __init__(self, config: BetaVAEConfig) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        C, D, H, W = config.input_shape
        L = config.latent_dim

        # Minimal 3D CNN encoder to keep parameter count manageable on CPU
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(C, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, C, D, H, W)
            dummy_out = self.encoder_conv(dummy)
            self.flat_size = dummy_out.numel()
            self.conv_out_shape = dummy_out.shape[1:]  # (C', D', H', W')

        self.fc_mu = nn.Linear(self.flat_size, L)
        self.fc_logvar = nn.Linear(self.flat_size, L)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(L, self.flat_size),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(8, C, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h_flat = torch.flatten(h, start_dim=1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.decoder_fc(z)
        h = h_flat.view(-1, *self.conv_out_shape)
        
        x_hat = self.decoder_conv(h)
        # Handle potential size mismatch due to strides/padding if input_shape isn't perfectly divisible by 8
        C, D, H, W = self.config.input_shape
        if x_hat.shape[2:] != (D, H, W):
            x_hat = F.interpolate(x_hat, size=(D, H, W), mode='trilinear', align_corners=False)
            
        return x_hat

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.config.beta * kl_loss
        return recon_loss, kl_loss, total_loss

    def forward(self, x: torch.Tensor) -> BetaVAEOutput:
        x = torch.nan_to_num(x, nan=0.0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        recon_loss, kl_loss, total_loss = self.compute_loss(x, x_hat, mu, logvar)
        return BetaVAEOutput(
            z=z, mu=mu, logvar=logvar, x_hat=x_hat,
            recon_loss=recon_loss, kl_loss=kl_loss, total_loss=total_loss
        )
