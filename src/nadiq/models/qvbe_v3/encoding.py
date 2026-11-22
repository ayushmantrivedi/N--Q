"""
NĀḌĪ-Q QVBE-v3 Encoding and Input Transformations.
Implements Data Re-uploading and Trainable Linear Adapters to eliminate the PCA bottleneck.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from nadiq.models.qvbe_v3.config import QVBEv3Config


class TrainableInputAdapter(nn.Module):
    """
    Trainable linear projection adapter from high-dimensional inputs to quantum register.
    Eliminates fixed classical PCA bottleneck by learning task-aligned subspace projection.
    """
    def __init__(self, input_dim: int, n_qubits: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        # Parameter named 'adapter' with shape (n_qubits, input_dim)
        self.adapter = nn.Parameter(torch.randn(n_qubits, input_dim) * (1.0 / math.sqrt(input_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> (B, n_qubits)
        return torch.matmul(x, self.adapter.t())


class DataReuploadingEncoder(nn.Module):
    """
    Data Re-uploading Encoder (Pérez-Salinas et al., 2020).
    Applies input features sequentially across multiple circuit layers with trainable scale factors,
    expanding the accessible Fourier frequency spectrum from {-1, 0, 1} to {-K, ..., K}.
    """
    def __init__(self, config: QVBEv3Config):
        super().__init__()
        self.config = config
        self.n_uploads = config.n_uploads
        self.n_qubits = config.n_qubits

        # Scaling weights for each upload layer
        torch.manual_seed(config.seed)
        self.upload_scales = nn.Parameter(torch.ones(self.n_uploads, self.n_qubits))

    def get_angles(self, x_proj: torch.Tensor, upload_idx: int) -> torch.Tensor:
        """
        Compute rotation angles for a specific upload stage.
        x_proj: (B, n_qubits)
        returns: (B, n_qubits) in [0, pi]
        """
        # Smooth bounded angle mapping: sigmoid mapping to [0, pi]
        scaled = x_proj * self.upload_scales[upload_idx].unsqueeze(0)
        return torch.sigmoid(scaled) * math.pi
