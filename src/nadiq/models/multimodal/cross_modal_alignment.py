"""
CrossModalAlignment — Interface Definition (NOT IMPLEMENTED).

Nāḍī-Q V1.3 — Pre-PPMI Integration Preparation.

Scientific Position
-------------------
This module defines the INTERFACE for future cross-modal alignment between
quantum/classical modality embeddings. The alignment algorithm itself has NOT
been chosen and is NOT implemented here.

Design decision is DEFERRED pending:
  1. Real paired multimodal PPMI data acquisition.
  2. Empirical analysis of embedding geometry across modalities.
  3. Selection of an appropriate alignment paradigm.

Candidate alignment strategies under consideration (NOT implemented here):
  - Contrastive learning (e.g., NT-Xent, SimCLR-style)
  - Canonical Correlation Analysis (CCA / DCCA)
  - Cross-attention transformer bridge
  - Latent space regularisation (e.g., shared Gaussian prior)

Current Status
--------------
INTERFACE ONLY. No algorithm is implemented.
The `align()` method raises NotImplementedError.
Subclasses will implement specific alignment strategies once:
  a) paired training data exists, and
  b) an alignment paradigm has been experimentally validated.

Usage (Future)
--------------
    class ContrastiveAlignment(CrossModalAlignment):
        def align(self, inputs: CrossModalInput) -> CrossModalOutput:
            ...  # implement contrastive cross-modal loss here

    aligner = ContrastiveAlignment(config)
    output = aligner.align(inputs)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = [
    "CrossModalInput",
    "CrossModalOutput",
    "CrossModalAlignmentConfig",
    "CrossModalAlignment",
]


@dataclass
class CrossModalInput:
    """
    Input to the cross-modal alignment interface.

    All tensors must share the same batch size B.
    Missing modalities are represented by availability=False and zero tensors.

    Tensor shapes:
        z_mri:         (B, 4)  — MRI quantum embedding from QVBE.
        z_bio:         (B, 4)  — Biomarker quantum embedding from QVBE.
        z_kin:         (B, 32) — Kinematics temporal embedding from CTCLM.
        availability:  (B, 3) or (3,) — Binary availability indicators [mri, bio, kin].
        patient_ids:   List of B patient identifiers (for traceability).
        event_ids:     List of B event identifiers (for traceability).
    """

    z_mri: torch.Tensor           # (B, 4)
    z_bio: torch.Tensor           # (B, 4)
    z_kin: torch.Tensor           # (B, 32)
    availability: torch.Tensor    # (B, 3) or (3,)
    patient_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        batch_dims = set()
        for name, t in [("z_mri", self.z_mri), ("z_bio", self.z_bio), ("z_kin", self.z_kin)]:
            if t.dim() < 1:
                raise ValueError(f"{name} must be at least 1-dimensional.")
            batch_dims.add(t.shape[0])
        if len(batch_dims) > 1:
            raise ValueError(f"CrossModalInput batch size mismatch: {batch_dims}")


@dataclass
class CrossModalOutput:
    """
    Output of a cross-modal alignment operation.

    Attributes:
        aligned_z_mri:    Aligned MRI representation.   (B, alignment_dim)
        aligned_z_bio:    Aligned biomarker representation. (B, alignment_dim)
        aligned_z_kin:    Aligned kinematics representation. (B, alignment_dim)
        shared_z:         Optional shared/fused representation. (B, alignment_dim)
        alignment_loss:   Cross-modal alignment loss (for training).
        patient_ids:      Forwarded patient identifiers (traceability).
        event_ids:        Forwarded event identifiers (traceability).
        metadata:         Diagnostic metadata from the alignment step.
    """

    aligned_z_mri: torch.Tensor
    aligned_z_bio: torch.Tensor
    aligned_z_kin: torch.Tensor
    shared_z: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None
    patient_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class CrossModalAlignmentConfig:
    """
    Base configuration for cross-modal alignment.

    Subclasses may extend this with algorithm-specific hyperparameters.

    Attributes:
        alignment_dim:   Target dimensionality for aligned representations.
        temperature:     Temperature parameter (relevant for contrastive methods).
        seed:            Random seed for deterministic initialization.
    """

    alignment_dim: int = 16
    temperature: float = 0.07
    seed: int = 42


class CrossModalAlignment(ABC):
    """
    Abstract base class for cross-modal alignment algorithms.

    INTERFACE ONLY — no implementation is provided here.

    This interface accepts individual modality embeddings and returns aligned
    representations that have been brought into a shared geometric space.

    The alignment algorithm (contrastive, CCA, cross-attention, etc.) is
    chosen by the concrete subclass. This design allows plug-in replacement
    of alignment strategies without modifying the upstream modality encoders
    (β-VAE, QVBE, CTCLM) or the downstream fusion layer (QCDF).

    Contract
    --------
    - Input: CrossModalInput containing Z_MRI (B,4), Z_BIO (B,4), Z_KIN (B,32),
             availability mask, and patient/event traceability IDs.
    - Output: CrossModalOutput with aligned embeddings and optional shared_z.
    - Traceability: patient_ids and event_ids must be forwarded to the output.

    Implementation Requirements (for future subclasses)
    ---------------------------------------------------
    1. Must NOT modify β-VAE, QVBE, CTCLM, or QCDF internals.
    2. Must NOT fabricate embeddings for missing modalities.
    3. Must preserve patient_id and event_id in all outputs.
    4. alignment_loss must be a differentiable scalar tensor when training.

    IMPORTANT: Do not implement this until paired multimodal PPMI data exists
    and an alignment strategy has been empirically selected.
    """

    def __init__(self, config: CrossModalAlignmentConfig) -> None:
        self.config = config

    @abstractmethod
    def align(self, inputs: CrossModalInput) -> CrossModalOutput:
        """
        Align modality embeddings into a shared representation space.

        Subclasses implement the specific alignment algorithm here.

        Args:
            inputs: CrossModalInput with Z_MRI, Z_BIO, Z_KIN, availability,
                    patient_ids, event_ids.

        Returns:
            CrossModalOutput with aligned embeddings and traceability forwarded.

        Raises:
            NotImplementedError: Always, until a concrete subclass is implemented.
        """
        raise NotImplementedError(
            "CrossModalAlignment is an interface. "
            "Implement a concrete subclass once paired PPMI data and an "
            "alignment strategy have been selected."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alignment_dim={self.config.alignment_dim}, "
            f"temperature={self.config.temperature})"
        )
