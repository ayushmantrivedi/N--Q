"""
ModalityAssembler — Nāḍī-Q Multimodal Integration Layer.

Assembles independent modality embeddings (produced by separate encoder branches)
into a canonical ModalityBundle suitable for QCDF consumption.

Scientific Position
-------------------
This module performs DATA PLUMBING only. It assembles pre-computed embeddings
into the QCDF input contract. It does NOT learn, align, or transform embeddings.
All scientific modality encoders (β-VAE, QVBE, CTCLM) are invoked BEFORE
this module, and their outputs are fed INTO the assembler.

Architecture Position
---------------------
    MRI → β-VAE → QVBE → Z_MRI (B, 4)            ┐
    BIO → representation → QVBE → Z_BIO (B, 4)   ├─→ ModalityAssembler → ModalityBundle → QCDF
    KIN → CTCLM → Z_KIN (B, 32)                   ┘

Missing Modality Contract
-------------------------
If a modality embedding is None or not available:
  - embedding is set to a zero tensor of the expected dim.
  - ModalityRepresentation.available = False.
  - QCDF receives an explicit availability indicator = 0.0.
No fabrication or imputation of missing embeddings is performed.

Traceability Contract
---------------------
The assembled ModalityBundle carries patient_id and event_id in its metadata,
maintaining end-to-end provenance from raw data to QCDF prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from nadiq.data.patient_timeline import ModalityAvailability, PatientTimeline
from nadiq.models.representation import (
    ModalityBundle,
    ModalityRepresentation,
    RepresentationType,
)

__all__ = [
    "AssemblerConfig",
    "AssembledSample",
    "ModalityAssembler",
]


@dataclass
class AssemblerConfig:
    """
    Configuration for the ModalityAssembler.

    Attributes:
        mri_embedding_dim:       Expected dimension of MRI encoder output. Default: 4 (QVBE n_qubits).
        biomarker_embedding_dim: Expected dimension of biomarker encoder output. Default: 4.
        kinematics_embedding_dim: Expected dimension of CTCLM output. Default: 32.
    """

    mri_embedding_dim: int = 4
    biomarker_embedding_dim: int = 4
    kinematics_embedding_dim: int = 32

    def __post_init__(self) -> None:
        for name, dim in [
            ("mri_embedding_dim", self.mri_embedding_dim),
            ("biomarker_embedding_dim", self.biomarker_embedding_dim),
            ("kinematics_embedding_dim", self.kinematics_embedding_dim),
        ]:
            if dim < 1:
                raise ValueError(f"{name} must be >= 1, got {dim}")


@dataclass
class AssembledSample:
    """
    The output of ModalityAssembler for a single patient event.

    Attributes:
        patient_id:  Participant identifier (traceable to source).
        event_id:    Visit/event identifier (traceable to source).
        bundle:      ModalityBundle ready for QCDF consumption.
        availability: Explicit modality availability mask.
        metadata:    Provenance metadata forwarded from PatientTimeline.
    """

    patient_id: str
    event_id: str
    bundle: ModalityBundle
    availability: ModalityAvailability
    metadata: dict = field(default_factory=dict)

    @property
    def sample_key(self) -> str:
        """Unique key combining patient_id and event_id."""
        return f"{self.patient_id}|{self.event_id}"

    def availability_tensor(self) -> torch.Tensor:
        """
        Return a (3,) float tensor encoding modality availability.
        Order: [mri, biomarker, kinematics]
        """
        return torch.tensor(
            [float(self.availability.mri),
             float(self.availability.biomarker),
             float(self.availability.kinematics)],
            dtype=torch.float32,
        )


class ModalityAssembler:
    """
    Assembles pre-computed modality embeddings into a canonical ModalityBundle.

    This class is intentionally NOT a nn.Module — it performs no learning.
    It enforces the QCDF input contract:
      MRI embedding (B, 4) + BIO embedding (B, 4) + KIN embedding (B, 32) +
      availability mask (3,) → ModalityBundle

    Usage
    -----
        config = AssemblerConfig(mri_embedding_dim=4, biomarker_embedding_dim=4, kinematics_embedding_dim=32)
        assembler = ModalityAssembler(config)

        sample = assembler.assemble(
            timeline=timeline,
            mri_embedding=mri_z,          # (B, 4) or None
            biomarker_embedding=bio_z,    # (B, 4) or None
            kinematics_embedding=kin_z,   # (B, 32) or None
        )
        bundle = sample.bundle  # ModalityBundle ready for QCDF
    """

    def __init__(self, config: AssemblerConfig) -> None:
        self.config = config

    def _make_representation(
        self,
        modality: str,
        embedding: Optional[torch.Tensor],
        available: bool,
        expected_dim: int,
        rep_type: RepresentationType,
        batch_size: int,
    ) -> ModalityRepresentation:
        """
        Build a ModalityRepresentation, using a zero tensor if unavailable.
        No fabrication of real-valued content for missing modalities.
        """
        if available and embedding is not None:
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if embedding.shape[-1] != expected_dim:
                raise ValueError(
                    f"Embedding dim mismatch for {modality}: "
                    f"expected {expected_dim}, got {embedding.shape[-1]}"
                )
            if embedding.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch for {modality}: "
                    f"expected {batch_size}, got {embedding.shape[0]}"
                )
            return ModalityRepresentation(
                modality=modality,
                embedding=embedding,
                representation_type=rep_type,
                available=True,
            )
        else:
            # Zero tensor — signals absence to QCDF via availability indicator.
            zero = torch.zeros(batch_size, expected_dim)
            return ModalityRepresentation(
                modality=modality,
                embedding=zero,
                representation_type=rep_type,
                available=False,
            )

    def assemble(
        self,
        timeline: PatientTimeline,
        mri_embedding: Optional[torch.Tensor] = None,
        biomarker_embedding: Optional[torch.Tensor] = None,
        kinematics_embedding: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> AssembledSample:
        """
        Assemble modality embeddings into a canonical multimodal sample.

        Args:
            timeline:              PatientTimeline carrying patient_id, event_id,
                                   and modality availability flags.
            mri_embedding:         MRI quantum embedding tensor, shape (B, mri_dim).
                                   Pass None if MRI is unavailable.
            biomarker_embedding:   Biomarker quantum embedding, shape (B, bio_dim).
                                   Pass None if biomarker is unavailable.
            kinematics_embedding:  CTCLM output, shape (B, kin_dim).
                                   Pass None if kinematics is unavailable.
            batch_size:            Batch size to use for zero tensors when a
                                   modality is missing. Must match embedding batch
                                   sizes when embeddings are provided.

        Returns:
            AssembledSample with full traceability and a ModalityBundle for QCDF.
        """
        avail = timeline.modality_availability

        # Determine actual batch size from any provided embedding.
        for emb in [mri_embedding, biomarker_embedding, kinematics_embedding]:
            if emb is not None:
                batch_size = emb.shape[0] if emb.dim() > 1 else 1
                break

        mri_rep = self._make_representation(
            modality="mri",
            embedding=mri_embedding,
            available=avail.mri and mri_embedding is not None,
            expected_dim=self.config.mri_embedding_dim,
            rep_type=RepresentationType.QUANTUM,
            batch_size=batch_size,
        )

        bio_rep = self._make_representation(
            modality="biomarkers",
            embedding=biomarker_embedding,
            available=avail.biomarker and biomarker_embedding is not None,
            expected_dim=self.config.biomarker_embedding_dim,
            rep_type=RepresentationType.QUANTUM,
            batch_size=batch_size,
        )

        kin_rep = self._make_representation(
            modality="kinematics",
            embedding=kinematics_embedding,
            available=avail.kinematics and kinematics_embedding is not None,
            expected_dim=self.config.kinematics_embedding_dim,
            rep_type=RepresentationType.CLASSICAL_TEMPORAL,
            batch_size=batch_size,
        )

        bundle = ModalityBundle(mri=mri_rep, biomarkers=bio_rep, kinematics=kin_rep)
        bundle.validate_batch_consistency()

        return AssembledSample(
            patient_id=timeline.patient_id,
            event_id=timeline.event_id,
            bundle=bundle,
            availability=avail,
            metadata={
                "dataset_id": timeline.dataset_id,
                "clinical_state": timeline.clinical_state.value,
                **timeline.metadata,
            },
        )
