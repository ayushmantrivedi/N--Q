"""
MRI Preprocessing Interface — Nāḍī-Q V1.

Scientific Position
-------------------
Provides a configurable, safe MRI preprocessing interface that retains 3D spatial 
structure. Does NOT silently convert 3D volumes into unrelated 2D slices.

Role in Nāḍī-Q Architecture
----------------------------
raw 3D T1 volume
    ↓
MRIPreprocessor (orientation/spatial/intensity normalization)
    ↓
normalized 3D tensor [B, C, D, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "MRIPreprocessingConfig",
    "MRIPreprocessor",
]


@dataclass
class MRIPreprocessingConfig:
    """
    Configuration for MRI preprocessing.

    Attributes:
        target_shape       : Desired output dimensions (D, H, W).
        normalize_intensity: Whether to apply zero-mean unit-variance normalization.
        brain_masking      : Placeholder for future skull-stripping logic.
    """
    target_shape: tuple[int, int, int]
    normalize_intensity: bool = True
    brain_masking: bool = False

    def __post_init__(self) -> None:
        if len(self.target_shape) != 3:
            raise ValueError(f"target_shape must be (D, H, W), got {self.target_shape}")


class MRIPreprocessor:
    """
    Applies configurable normalization and spatial resampling to 3D MRI volumes.

    Input shape: (B, C, D, H, W)
    Output shape: (B, C, target_D, target_H, target_W)
    """
    def __init__(self, config: MRIPreprocessingConfig) -> None:
        self.config = config

    def __call__(self, volumes: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of MRI volumes.

        Args:
            volumes: torch.Tensor of shape (B, C, D, H, W)
                     representing raw 3D T1 MRI data.

        Returns:
            torch.Tensor of shape (B, C, target_D, target_H, target_W)
        """
        if volumes.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {volumes.ndim}D")

        # Optional Brain Masking (Placeholder for V1)
        if self.config.brain_masking:
            # Masking would be applied here; currently bypassed to preserve data
            pass

        # Spatial Normalization (Resampling)
        if volumes.shape[2:] != self.config.target_shape:
            # Trilinear interpolation for continuous 3D medical images
            volumes = F.interpolate(
                volumes,
                size=self.config.target_shape,
                mode="trilinear",
                align_corners=False
            )

        # Intensity Normalization (Zero-mean, unit-variance per volume)
        if self.config.normalize_intensity:
            # Compute stats over the spatial dimensions (D, H, W)
            # Avoid dividing by zero standard deviation
            means = volumes.mean(dim=(2, 3, 4), keepdim=True)
            stds = volumes.std(dim=(2, 3, 4), unbiased=False, keepdim=True)
            stds = torch.clamp(stds, min=1e-8)
            volumes = (volumes - means) / stds

        return volumes

    def get_config(self) -> dict[str, Any]:
        return {
            "target_shape": self.config.target_shape,
            "normalize_intensity": self.config.normalize_intensity,
            "brain_masking": self.config.brain_masking,
        }
