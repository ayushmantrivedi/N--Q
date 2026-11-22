"""
Quantum Clinical Decision Fusion (QCDF) — Nāḍī-Q V1.

Scientific Position
-------------------
This module implements the final multimodal fusion step of the Nāḍī-Q architecture.
It takes independent `ModalityRepresentation` objects (MRI, Biomarkers, Kinematics),
projects them to a common subspace, and feeds them into a dense classifier.

Role in Nāḍī-Q Architecture
----------------------------
MRI quantum embedding
        \\
         \\
Biomarker quantum embedding
           → modality projections
           → availability indicators
           → concatenation
           → small MLP
           → prediction
         /
Kinematic CTCLM embedding
        /

Design Requirements
-------------------
- Small, interpretable MLP (no transformers/attention).
- Consumes explicit availability indicators.
- Handles missing modalities safely.
- Retains disease-agnostic structure (no hardcoded ADNI semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from nadiq.models.representation import ModalityBundle

__all__ = [
    "QCDFConfig",
    "QCDFOutput",
    "QCDF",
]


@dataclass
class QCDFConfig:
    """
    Configuration for the QCDF layer.

    Attributes:
        mri_dim           : Dimension of the MRI embedding.
        biomarker_dim     : Dimension of the Biomarker embedding.
        kinematics_dim    : Dimension of the Kinematics embedding.
        projection_dim    : Dimension to which each available modality is projected.
        hidden_dim        : Dimension of the MLP hidden layers.
        num_classes       : Number of target clinical classes.
        dropout           : Dropout probability for MLP.
        seed              : Random seed for deterministic initialization.
    """
    mri_dim: int = 4
    biomarker_dim: int = 4
    kinematics_dim: int = 32
    projection_dim: int = 16
    hidden_dim: int = 32
    num_classes: int = 3
    dropout: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 for classification")


@dataclass
class QCDFOutput:
    """
    Output from the QCDF fusion layer.

    Attributes:
        logits       : Classification logits, shape (B, num_classes).
        probabilities: Softmax probabilities, shape (B, num_classes).
        fused_vector : Intermediate dense fused representation, shape (B, hidden_dim).
    """
    logits: torch.Tensor
    probabilities: torch.Tensor
    fused_vector: torch.Tensor


class QCDF(nn.Module):
    """
    Quantum Clinical Decision Fusion layer.

    Fuses MRI, Biomarkers, and Kinematics representations.
    Explicitly accounts for availability indicators to avoid interpreting
    zero-padding as valid measurements.
    """

    def __init__(self, config: QCDFConfig) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        # Modality-specific projection layers
        self.proj_mri = nn.Linear(config.mri_dim, config.projection_dim)
        self.proj_biomarkers = nn.Linear(config.biomarker_dim, config.projection_dim)
        self.proj_kinematics = nn.Linear(config.kinematics_dim, config.projection_dim)

        # Each projection is concatenated alongside a 1D availability indicator (per modality)
        # Total concatenated size = 3 * projection_dim + 3 (availability bits)
        self.concat_dim = (3 * config.projection_dim) + 3

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _project_modality(self,
                          embedding: Optional[torch.Tensor],
                          available: bool,
                          proj_layer: nn.Module,
                          batch_size: int,
                          device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Projects a modality embedding if available, otherwise returns zeros."""
        
        # Availability tensor: (B, 1)
        avail_tensor = torch.zeros(batch_size, 1, device=device)

        if available and embedding is not None:
            # (B, projection_dim)
            proj = proj_layer(embedding)
            avail_tensor.fill_(1.0)
        else:
            proj = torch.zeros(batch_size, self.config.projection_dim, device=device)
            
        return proj, avail_tensor

    def forward(self, bundle: ModalityBundle) -> QCDFOutput:
        """
        Forward pass fusing available modalities.

        Args:
            bundle: ModalityBundle containing optional MRI, Biomarkers, Kinematics embeddings.

        Returns:
            QCDFOutput containing logits and probabilities.
        """
        bundle.validate_batch_consistency()

        # Determine batch size and device from any available embedding
        batch_size = 1
        device = torch.device('cpu')
        for rep in [bundle.mri, bundle.biomarkers, bundle.kinematics]:
            if rep is not None:
                batch_size = rep.batch_size
                device = rep.embedding.device
                break

        # MRI
        mri_avail = bundle.mri is not None and bundle.mri.available
        mri_emb = bundle.mri.embedding if mri_avail else None
        mri_proj, mri_ind = self._project_modality(
            mri_emb, mri_avail, self.proj_mri, batch_size, device
        )

        # Biomarkers
        bio_avail = bundle.biomarkers is not None and bundle.biomarkers.available
        bio_emb = bundle.biomarkers.embedding if bio_avail else None
        bio_proj, bio_ind = self._project_modality(
            bio_emb, bio_avail, self.proj_biomarkers, batch_size, device
        )

        # Kinematics
        kin_avail = bundle.kinematics is not None and bundle.kinematics.available
        kin_emb = bundle.kinematics.embedding if kin_avail else None
        kin_proj, kin_ind = self._project_modality(
            kin_emb, kin_avail, self.proj_kinematics, batch_size, device
        )

        # Concatenate: (B, concat_dim)
        fused_input = torch.cat(
            [mri_proj, bio_proj, kin_proj, mri_ind, bio_ind, kin_ind], dim=-1
        )

        fused_vector = self.mlp(fused_input)
        logits = self.classifier(fused_vector)
        probs = torch.softmax(logits, dim=-1)

        return QCDFOutput(
            logits=logits,
            probabilities=probs,
            fused_vector=fused_vector
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "mri_dim": self.config.mri_dim,
            "biomarker_dim": self.config.biomarker_dim,
            "kinematics_dim": self.config.kinematics_dim,
            "projection_dim": self.config.projection_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_classes": self.config.num_classes,
            "n_parameters": self.n_parameters,
        }
