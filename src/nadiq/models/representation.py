"""
Modality Representation Contract — Nāḍī-Q V1.

Defines the unified output type that all modality encoder branches
must produce before feeding into QCDF (Quantum Clinical Decision Fusion).

Nāḍī-Q V1 Modality Architecture (FROZEN)
-----------------------------------------
    MRI        : BetaVAE → QVBE → MRI quantum embedding
    Biomarkers : Trajectory analysis → QVBE → Biomarker quantum embedding
    Kinematics : CTCLM → Kinematic temporal latent

    All three → QCDF (not yet implemented)

The ModalityRepresentation type allows QCDF to consume all three
representations without knowing how they were generated. This enforces
strict branch isolation.

Design Rules
------------
- ClinicalSample is NOT modified by this abstraction.
- ModalityRepresentation is a downstream intermediate — it exists between
  the modality encoders and QCDF.
- The CTCLM branch outputs a classical_temporal representation.
  The MRI and biomarker branches output quantum representations.
- These types should NOT be confused or interchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import torch

__all__ = [
    "RepresentationType",
    "ModalityRepresentation",
    "ModalityBundle",
]


class RepresentationType(Enum):
    """The type of representation produced by a modality encoder branch."""

    QUANTUM = auto()             # Produced by QVBE-based quantum encoding
    CLASSICAL_TEMPORAL = auto()  # Produced by CTCLM (kinematics branch)
    CLASSICAL_STATIC = auto()    # Fallback for simple static features


@dataclass
class ModalityRepresentation:
    """
    The output of one modality encoder branch.

    Attributes:
        modality         : Identifier string (e.g., "mri", "biomarkers", "kinematics").
        embedding        : The dense representation tensor.
                           Shape: (B, embedding_dim).
        representation_type : RepresentationType enum.
        available        : Whether this modality was actually observed.
                           If False, embedding contains zeros (zero-mask convention).
        metadata         : Optional diagnostic / provenance information.
    """

    modality: str
    embedding: torch.Tensor        # (B, embedding_dim)
    representation_type: RepresentationType
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.embedding.shape[0]


@dataclass
class ModalityBundle:
    """
    A collection of modality representations for a patient batch.

    Intended for consumption by QCDF (not yet implemented).
    All present representations must have the same batch size.

    Attributes:
        mri          : MRI quantum embedding representation (optional).
        biomarkers   : Biomarker quantum embedding representation (optional).
        kinematics   : CTCLM kinematic temporal representation (optional).
    """

    mri: Optional[ModalityRepresentation] = None
    biomarkers: Optional[ModalityRepresentation] = None
    kinematics: Optional[ModalityRepresentation] = None

    @property
    def available_modalities(self) -> list[str]:
        """Returns names of modalities present and available."""
        result = []
        for name, rep in [("mri", self.mri), ("biomarkers", self.biomarkers), ("kinematics", self.kinematics)]:
            if rep is not None and rep.available:
                result.append(name)
        return result

    @property
    def n_available(self) -> int:
        return len(self.available_modalities)

    def validate_batch_consistency(self) -> None:
        """Assert all present representations share the same batch size."""
        sizes = []
        for rep in [self.mri, self.biomarkers, self.kinematics]:
            if rep is not None:
                sizes.append(rep.batch_size)
        if len(set(sizes)) > 1:
            raise ValueError(
                f"ModalityBundle has inconsistent batch sizes: {sizes}. "
                "All modality embeddings must share the same batch dimension."
            )
