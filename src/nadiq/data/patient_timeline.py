"""
PatientTimeline — Nāḍī-Q Multimodal Integration Layer.

Defines the canonical per-patient, per-event data structure used to assemble
multimodal samples for QCDF (Quantum Clinical Decision Fusion).

Scientific Position
-------------------
This module provides data-plumbing abstractions only. It does NOT perform
any cross-modal learning, clinical prediction, or introduce any scientific
assumptions about multimodal relationships.

Design Rules
------------
- PatientTimeline stores REFERENCES to modality data, not embeddings directly.
  Embeddings are attached by the ModalityAssembler after encoder forward passes.
- Every assembled multimodal sample must preserve patient_id + event_id for
  full traceability through the pipeline.
- Missing modalities are never fabricated. Absence is encoded via the
  ModalityAvailability mask and zero-tensors in ModalityBundle.
- This module is dataset-agnostic. PPMI-specific field mapping belongs in
  the PPMIAdapter, not here.

PPMI Compatibility
------------------
PPMI uses:
  PATNO  — participant number (maps to patient_id)
  EVENT_ID — visit code (maps to event_id)
These are mapped by the PPMIAdapter skeleton in:
  src/nadiq/data/sources/ppmi.py

Usage
-----
    timeline = PatientTimeline(
        patient_id="P001",
        event_id="BL",
        clinical_state=ClinicalState.PD,
        modality_availability=ModalityAvailability(mri=True, biomarker=True, kinematics=False),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

__all__ = [
    "ClinicalState",
    "ModalityAvailability",
    "ModalityReference",
    "PatientTimeline",
]


class ClinicalState(Enum):
    """
    Clinical state of a participant at a given event/visit.

    These values are dataset-agnostic. Dataset-specific labels (e.g., PPMI
    cohort codes HC, PD, PROD) must be mapped to this enum by the adapter.
    """

    HEALTHY_CONTROL = "HC"
    PARKINSON_DISEASE = "PD"
    PRODROMAL = "PROD"       # Pre-clinical / at-risk cohort
    MCI = "MCI"              # Mild Cognitive Impairment
    UNKNOWN = "UNKNOWN"      # Label unavailable or pending adjudication


@dataclass
class ModalityAvailability:
    """
    Binary availability mask for the three NĀḌĪ-Q modalities.

    True = modality was acquired and is present for this patient/event.
    False = modality is absent (not acquired, failed QC, or not applicable).

    Availability must reflect ACTUAL data presence, not assumed or imputed.
    """

    mri: bool = False
    biomarker: bool = False
    kinematics: bool = False

    def as_list(self) -> list[bool]:
        """Return [mri, biomarker, kinematics] availability as ordered list."""
        return [self.mri, self.biomarker, self.kinematics]

    def n_available(self) -> int:
        """Count of available modalities."""
        return sum([self.mri, self.biomarker, self.kinematics])

    def __repr__(self) -> str:
        parts = []
        if self.mri: parts.append("MRI")
        if self.biomarker: parts.append("BIO")
        if self.kinematics: parts.append("KIN")
        avail = "+".join(parts) if parts else "NONE"
        return f"ModalityAvailability({avail})"


@dataclass
class ModalityReference:
    """
    A lightweight reference to raw/processed modality data.

    This stores file paths, dataset keys, or in-memory identifiers so that
    the ModalityAssembler can locate and forward-pass the raw data through
    the appropriate encoder.

    Attributes:
        modality: One of "mri", "biomarker", "kinematics".
        data_key: Path, ID, or registry key pointing to the raw/preprocessed data.
        preprocessing_version: Hash or version string of the preprocessing applied.
        metadata: Arbitrary metadata (e.g., acquisition parameters, QC flags).
    """

    modality: str
    data_key: str
    preprocessing_version: str = "unprocessed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientTimeline:
    """
    A single patient observation at a specific clinical event/visit.

    This is the canonical unit of multimodal data assembly in NĀḌĪ-Q.
    It records WHAT data is available for a patient at a given event,
    without storing the embeddings themselves (those are produced by encoders
    and stored in a ModalityBundle by the ModalityAssembler).

    Attributes:
        patient_id:           Anonymized participant identifier (e.g., PPMI PATNO).
        event_id:             Visit/event code (e.g., PPMI EVENT_ID like 'BL', 'V02').
        clinical_state:       Known or inferred clinical state at this event.
        modality_availability: Which modalities were successfully acquired.
        modality_refs:        Optional references to locate raw/processed data.
        dataset_id:           Source dataset registry key (e.g., 'ppmi', 'uci_pd').
        metadata:             Arbitrary provenance metadata (acquisition date, cohort, etc.).

    Traceability
    ------------
    patient_id + event_id together form a globally unique key within a dataset.
    Every downstream ModalityBundle and QCDF prediction must carry both fields.

    PPMI Mapping
    ------------
    PPMI: PATNO  → patient_id
    PPMI: EVENT_ID → event_id
    """

    patient_id: str
    event_id: str
    clinical_state: ClinicalState = ClinicalState.UNKNOWN
    modality_availability: ModalityAvailability = field(default_factory=ModalityAvailability)
    modality_refs: dict[str, ModalityReference] = field(default_factory=dict)
    dataset_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_key(self) -> str:
        """Unique key combining patient_id and event_id."""
        return f"{self.patient_id}|{self.event_id}"

    def has_modality(self, modality: str) -> bool:
        """Check if a named modality is available."""
        available = {
            "mri": self.modality_availability.mri,
            "biomarker": self.modality_availability.biomarker,
            "kinematics": self.modality_availability.kinematics,
        }
        return available.get(modality, False)

    def get_modality_ref(self, modality: str) -> Optional[ModalityReference]:
        """Return the reference for a modality, or None if absent."""
        return self.modality_refs.get(modality, None)

    def __repr__(self) -> str:
        return (
            f"PatientTimeline(patient_id={self.patient_id!r}, event_id={self.event_id!r}, "
            f"state={self.clinical_state.value}, "
            f"availability={self.modality_availability!r})"
        )
