"""
PPMI Dataset Adapter Skeleton — Nāḍī-Q V1.3.

Schema/interface ONLY. No real PPMI files are accessed here.

Scientific Position
-------------------
This is an ADAPTER SKELETON. It defines the interface and field mapping
contract for PPMI (Parkinson's Progression Markers Initiative) data.
No real PPMI data is loaded, no column mappings beyond documented generic
fields are guessed, and no clinical assumptions are made.

PPMI Identifiers
----------------
PPMI uses two primary identifiers:
  PATNO     — Participant Number (integer), maps to PatientTimeline.patient_id
  EVENT_ID  — Visit code (string, e.g. "BL", "V02", "V04"), maps to event_id

Key PPMI file categories (NOT yet mapped here):
  Motor assessments  (UPDRS)
  Cognitive assessments
  Imaging (MRI, DAT-SCAN)
  Biospecimen (CSF, blood)
  Demographic

Column mapping is intentionally LEFT INCOMPLETE pending:
  1. Actual PPMI data file download and schema inspection.
  2. Selection of the specific PPMI sub-cohorts to include.
  3. Validation of column presence across PPMI data releases.

Usage (Future)
--------------
    source = PPMILocalSource(data_dir="/path/to/ppmi/downloads")
    adapter = PPMIAdapter(source=source, cohorts=["PD", "HC"])
    for sample in adapter.load_split(DataSplit.TRAIN):
        ...

Reference
---------
PPMI: https://www.ppmi-info.org/
PPMI Data Dictionary: https://ida.loni.usc.edu/
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from nadiq.data.contracts import (
    BaseDatasetLoader,
    ClinicalLabel,
    ClinicalSample,
    DataSplit,
    DatasetSchema,
    DatasetStats,
    FeatureBlock,
    FeatureDtype,
    LabelType,
    ModalityType,
    SampleMetadata,
)
from nadiq.data.patient_timeline import ClinicalState, ModalityAvailability, PatientTimeline
from nadiq.data.sources.base import AccessMode, DataSource, DataSourceCapability

__all__ = [
    "PPMISchemaConstants",
    "PPMICohort",
    "PPMIAdapterConfig",
    "PPMILocalSource",
    "PPMIAdapter",
]

PPMI_DATASET_ID = "ppmi"


class PPMICohort:
    """
    PPMI cohort codes used in EVENT_ID / COHORT columns.

    These are the documented PPMI cohort identifiers.
    Exact column names may differ across PPMI data releases.
    """

    PD = "PD"
    HEALTHY_CONTROL = "HC"
    PRODROMAL = "PROD"
    SWEDD = "SWEDD"  # Scans Without Evidence of a Dopaminergic Deficit


class PPMISchemaConstants:
    """
    Documented PPMI column names and field definitions.

    These are the ONLY fields pre-mapped here. All others require
    inspection of the actual downloaded PPMI CSV files.

    INCOMPLETE BY DESIGN — do not add guessed column names.
    """

    # Primary identifiers — present in virtually all PPMI tables.
    PATNO = "PATNO"         # Participant Number (integer)
    EVENT_ID = "EVENT_ID"   # Visit code (e.g., "BL", "V02", "V04", "ST")

    # Standard PPMI visit codes (documented in PPMI Data Dictionary)
    VISIT_BASELINE = "BL"
    VISIT_SCREENING = "SC"
    VISIT_UNSCHEDULED = "U01"

    # Standard cohort field (present in most PPMI subject files)
    COHORT = "COHORT"

    # ── Fields below are INTENTIONALLY BLANK ──────────────────────────────
    # Motor, cognitive, imaging, biospecimen column names are NOT hardcoded
    # here. They must be mapped after actual PPMI file inspection.
    # Use PPMIAdapter.map_columns() to register column mappings at runtime.


@dataclass
class PPMIAdapterConfig:
    """
    Configuration for the PPMI dataset adapter.

    Attributes:
        data_dir:         Path to the local directory containing downloaded PPMI files.
        cohorts:          Which PPMI cohorts to include. Default: PD + HC.
        visit_ids:        Specific visit codes to include. None means all visits.
        include_mri:      Whether to attempt MRI modality loading.
        include_biomarker: Whether to attempt biomarker modality loading.
        include_kinematics: Whether to attempt kinematics modality loading.
        split_by:         Splitting strategy. Only 'subject' is scientifically valid.
    """

    data_dir: str = ""
    cohorts: list[str] = field(default_factory=lambda: [PPMICohort.PD, PPMICohort.HEALTHY_CONTROL])
    visit_ids: Optional[list[str]] = None
    include_mri: bool = False          # Blocked pending data acquisition
    include_biomarker: bool = True
    include_kinematics: bool = False   # Blocked pending data acquisition
    split_by: str = "subject"          # MUST remain 'subject' to prevent leakage


class PPMILocalSource(DataSource):
    """
    DataSource implementation for locally downloaded PPMI files.

    Stub implementation — real file I/O is deferred until PPMI data
    is downloaded and the file structure has been inspected.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

    def authenticate(self) -> bool:
        """PPMI requires registration at ppmi-info.org. This stub does not handle auth."""
        return False

    def metadata(self) -> DataSourceCapability:
        return DataSourceCapability(
            status="SKELETON — data not loaded",
            access_mode=AccessMode.AUTHORIZED_REMOTE,
            bulk_download_supported=True,
            query_supported=False,
            local_data_required=True,
            estimated_size_gb=50.0,
            required_action=(
                "Register at https://www.ppmi-info.org/, download data files, "
                "set data_dir to the local path, then implement column mappings."
            ),
        )

    def list_available(self) -> list[str]:
        """Return list of CSV files available locally. Stub returns empty list."""
        return []

    def query(self, query_str: str) -> Any:
        raise NotImplementedError("PPMILocalSource does not support SQL queries.")

    def stream(self, resource_id: str) -> Iterable[bytes]:
        raise NotImplementedError("PPMILocalSource.stream() not yet implemented.")

    def fetch_subset(self, resource_ids: list[str], dest_dir: str) -> None:
        raise NotImplementedError(
            "PPMILocalSource.fetch_subset() not yet implemented. "
            "Download PPMI data manually from https://ida.loni.usc.edu/ "
            "and place files in the configured data_dir."
        )


class PPMIAdapter(BaseDatasetLoader):
    """
    PPMI dataset adapter for Nāḍī-Q.

    SKELETON IMPLEMENTATION — No real data access is performed.

    This adapter defines:
      1. The interface contract for loading PPMI ClinicalSamples.
      2. The patient_id (PATNO) and event_id (EVENT_ID) mapping.
      3. The cohort-to-ClinicalState mapping.

    Column Mapping (INCOMPLETE)
    ---------------------------
    Column names for motor, cognitive, imaging, and biospecimen features
    are NOT mapped here. They must be added via map_columns() after inspecting
    the actual PPMI CSV files.

    Subject-Level Splits
    ---------------------
    All data splits MUST be performed at the SUBJECT level (by PATNO),
    not by row. Row-level splits cause participant identity leakage.
    See: scripts/training/run_v2_biomarker_real.py for a validated example.

    Scientific Warning
    ------------------
    Do not add column mappings or data loading logic until actual PPMI files
    have been inspected. Guessing column names introduces silent schema errors.
    """

    # Mapping from PPMI cohort string → ClinicalState enum
    _COHORT_TO_STATE: dict[str, ClinicalState] = {
        PPMICohort.PD: ClinicalState.PARKINSON_DISEASE,
        PPMICohort.HEALTHY_CONTROL: ClinicalState.HEALTHY_CONTROL,
        PPMICohort.PRODROMAL: ClinicalState.PRODROMAL,
        PPMICohort.SWEDD: ClinicalState.UNKNOWN,
    }

    # ── Column mapping registry (populate after actual file inspection) ──────
    # Maps semantic field names → actual PPMI CSV column names.
    # Empty until real files are inspected.
    _COLUMN_MAP: dict[str, str] = {
        # "updrs_total": "NP3TOT",  # Example — uncomment after validation
        # "age_at_visit": "AGE_AT_VISIT",  # Example
    }

    def __init__(self, source: PPMILocalSource, config: PPMIAdapterConfig) -> None:
        self._source = source
        self._config = config
        self._column_map = dict(self._COLUMN_MAP)

    @property
    def dataset_id(self) -> str:
        return PPMI_DATASET_ID

    def map_columns(self, mapping: dict[str, str]) -> None:
        """
        Register semantic → actual column name mappings.

        Call this method after inspecting real PPMI CSV files.
        Do NOT add guessed column names — only validated ones.

        Args:
            mapping: Dict of {semantic_name: actual_csv_column}.

        Example (FUTURE — do not call until files are available):
            adapter.map_columns({"updrs_motor": "NP3TOT", "age": "AGE_AT_VISIT"})
        """
        self._column_map.update(mapping)

    def build_patient_timeline(
        self,
        patno: str,
        event_id: str,
        cohort: Optional[str] = None,
    ) -> PatientTimeline:
        """
        Construct a PatientTimeline from PPMI identifiers.

        Args:
            patno:    PPMI PATNO (participant number as string).
            event_id: PPMI EVENT_ID (visit code).
            cohort:   PPMI cohort string (maps to ClinicalState).

        Returns:
            PatientTimeline with patient_id=PATNO, event_id=EVENT_ID.
        """
        state = self._COHORT_TO_STATE.get(cohort or "", ClinicalState.UNKNOWN)
        return PatientTimeline(
            patient_id=str(patno),
            event_id=str(event_id),
            clinical_state=state,
            modality_availability=ModalityAvailability(
                mri=self._config.include_mri,
                biomarker=self._config.include_biomarker,
                kinematics=self._config.include_kinematics,
            ),
            dataset_id=PPMI_DATASET_ID,
            metadata={"cohort": cohort},
        )

    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        """
        SKELETON — raises NotImplementedError until real PPMI files exist.

        Implementation steps (deferred):
          1. Load PPMI CSV files from self._source.data_dir.
          2. Perform SUBJECT-level split by PATNO.
          3. Yield ClinicalSample objects.
        """
        raise NotImplementedError(
            "PPMIAdapter.load_split() requires local PPMI data. "
            "Download from https://www.ppmi-info.org/, "
            "set data_dir in PPMIAdapterConfig, and implement column mappings."
        )

    def get_schema(self) -> DatasetSchema:
        """
        Return the partial PPMI schema.

        Modalities are enabled based on config flags.
        Feature dims are UNKNOWN until column mapping is complete.
        """
        modalities: dict[str, ModalityType] = {}
        if self._config.include_biomarker:
            modalities["biomarker"] = ModalityType.TABULAR
        if self._config.include_mri:
            modalities["mri"] = ModalityType.IMAGE
        if self._config.include_kinematics:
            modalities["kinematics"] = ModalityType.TIME_SERIES

        return DatasetSchema(
            dataset_id=PPMI_DATASET_ID,
            modalities=modalities,
            feature_dims={},  # UNKNOWN — requires real file inspection
            label_def="Parkinson's Disease vs Healthy Control classification",
            label_type=LabelType.BINARY,
            n_classes=2,
        )

    def get_stats(self) -> DatasetStats:
        """SKELETON — Stats require real data loading."""
        raise NotImplementedError(
            "PPMIAdapter.get_stats() requires actual PPMI data files."
        )
