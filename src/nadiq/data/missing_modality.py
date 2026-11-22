"""
Missing-Modality Handling Utilities — Nāḍī-Q V1.3.

Provides utilities for working with arbitrary modality availability patterns
without fabricating embeddings or imputing missing clinical observations.

Scientific Position
-------------------
Missing modalities are a first-class condition in clinical data, not an error.
NĀḌĪ-Q handles them via explicit binary availability indicators passed to QCDF.
These utilities enforce the contract:

  MISSING MODALITY → zero tensor + availability=False → QCDF availability bit=0.0

No interpolation, imputation, or estimation of missing modalities is performed.
The absence of a modality is preserved and propagated through the pipeline.

Design Rules
------------
- Zero tensors are NOT embeddings. They signal absence.
- Availability masks are binary. There are no partial availabilities.
- Fabrication of embeddings for missing modalities is PROHIBITED.
- All functions are stateless and do NOT modify input tensors.
"""

from __future__ import annotations

from typing import Optional

import torch

from nadiq.data.patient_timeline import ModalityAvailability

__all__ = [
    "make_zero_embedding",
    "make_availability_tensor",
    "validate_embedding_against_availability",
    "describe_availability_pattern",
    "MISSING_MODALITY_PATTERNS",
]


# ---------------------------------------------------------------------------
# Known availability patterns (informational, not exhaustive)
# ---------------------------------------------------------------------------

MISSING_MODALITY_PATTERNS: dict[str, ModalityAvailability] = {
    "all_available": ModalityAvailability(mri=True, biomarker=True, kinematics=True),
    "bio_only": ModalityAvailability(mri=False, biomarker=True, kinematics=False),
    "mri_and_bio": ModalityAvailability(mri=True, biomarker=True, kinematics=False),
    "mri_only": ModalityAvailability(mri=True, biomarker=False, kinematics=False),
    "kin_only": ModalityAvailability(mri=False, biomarker=False, kinematics=True),
    "none_available": ModalityAvailability(mri=False, biomarker=False, kinematics=False),
}
"""
Pre-defined availability patterns for reference and testing.
These cover the main real-world cases in PPMI/PD studies.
They are NOT used to fabricate data — only as named constants.
"""


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def make_zero_embedding(batch_size: int, embedding_dim: int) -> torch.Tensor:
    """
    Return a zero tensor representing an absent modality.

    This is NOT an embedding. It is a structural placeholder that signals
    absence to QCDF via the accompanying availability indicator (0.0).

    Args:
        batch_size:    Number of samples in the batch.
        embedding_dim: Embedding dimensionality for the absent modality.

    Returns:
        Zero tensor of shape (batch_size, embedding_dim).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
    return torch.zeros(batch_size, embedding_dim)


def make_availability_tensor(availability: ModalityAvailability) -> torch.Tensor:
    """
    Convert a ModalityAvailability to a float tensor for QCDF.

    Returns:
        Tensor of shape (3,) with values in {0.0, 1.0}.
        Order: [mri, biomarker, kinematics].
    """
    return torch.tensor(
        [float(availability.mri),
         float(availability.biomarker),
         float(availability.kinematics)],
        dtype=torch.float32,
    )


def validate_embedding_against_availability(
    modality: str,
    embedding: Optional[torch.Tensor],
    available: bool,
    expected_dim: int,
) -> None:
    """
    Assert that an embedding's presence/absence matches the availability flag.

    Raises:
        ValueError: If the embedding is present when availability=False, or
                    if the embedding is absent when availability=True, or
                    if the embedding dim does not match expected_dim.

    Args:
        modality:     Human-readable modality name for error messages.
        embedding:    The embedding tensor (or None if absent).
        available:    Whether the modality was acquired.
        expected_dim: Expected last dimension of the embedding tensor.
    """
    if available and embedding is None:
        raise ValueError(
            f"Modality '{modality}' is marked available but embedding is None. "
            "Either provide the embedding or set availability=False."
        )
    if not available and embedding is not None:
        raise ValueError(
            f"Modality '{modality}' is marked unavailable but a non-None embedding "
            "was provided. Do not fabricate embeddings for missing modalities. "
            "Set the embedding to None and availability=False."
        )
    if embedding is not None and embedding.shape[-1] != expected_dim:
        raise ValueError(
            f"Modality '{modality}' embedding dim {embedding.shape[-1]} "
            f"does not match expected_dim {expected_dim}."
        )


def describe_availability_pattern(availability: ModalityAvailability) -> str:
    """
    Return a human-readable string describing a modality availability pattern.

    Args:
        availability: ModalityAvailability instance.

    Returns:
        Compact description, e.g. "MRI+BIO" or "BIO only" or "NONE".
    """
    parts = []
    if availability.mri:
        parts.append("MRI")
    if availability.biomarker:
        parts.append("BIO")
    if availability.kinematics:
        parts.append("KIN")
    if not parts:
        return "NONE (all modalities absent)"
    return "+".join(parts)
