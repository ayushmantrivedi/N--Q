"""
Data contract definitions for Nāḍī-Q.

Defines the core data structures and abstract interfaces that all dataset
loaders must implement. This is the single point of truth for what a
'clinical sample' looks like within this platform.

IMPORTANT: These are interface definitions only.
No dataset loading logic belongs in this file.

See: docs/data_contract.md for full specification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, Optional

import numpy as np

__all__ = [
    "ModalityType",
    "LabelType",
    "DataSplit",
    "FeatureDtype",
    "FeatureBlock",
    "ClinicalLabel",
    "SampleMetadata",
    "ClinicalSample",
    "DatasetSchema",
    "DatasetStats",
    "BaseDatasetLoader",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ModalityType(Enum):
    """Supported clinical data modalities."""

    TABULAR = auto()      # Static tabular biomarkers
    TIME_SERIES = auto()  # Longitudinal measurements
    IMAGE = auto()        # Medical imaging (not yet implemented)
    GENOMIC = auto()      # Genomic/genetic features (not yet implemented)


class LabelType(Enum):
    """Prediction task types."""

    BINARY = auto()       # Binary classification (risk: yes/no)
    MULTICLASS = auto()   # Multi-class classification
    REGRESSION = auto()   # Continuous risk score prediction


class DataSplit(Enum):
    """Dataset partition identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class FeatureDtype(Enum):
    """Clinical feature data types."""

    CONTINUOUS = auto()    # Real-valued measurements (e.g., glucose level)
    CATEGORICAL = auto()   # Nominal categories (e.g., blood type)
    BINARY = auto()        # Binary indicators (e.g., presence of symptom)
    ORDINAL = auto()       # Ordered categories (e.g., disease severity 1-5)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureBlock:
    """
    A single modality's feature representation for one sample.

    Attributes:
        modality:      The type of clinical modality.
        values:        Feature array of shape (n_features,) for tabular
                       or (n_timesteps, n_features) for time series.
        feature_names: Human-readable names for each feature dimension.
        missing_mask:  Boolean mask; True indicates a missing value.
                       Must have the same shape as `values`.
        time_offsets:  Optional float array representing time deltas (e.g., days from baseline).
                       Required for irregular longitudinal sequences. Shape: (n_timesteps,).
        dtype:         The predominant data type of the features.
                       Mixed-type blocks should use CONTINUOUS.
    """

    modality: ModalityType
    values: np.ndarray
    feature_names: list[str]
    missing_mask: Optional[np.ndarray] = None
    time_offsets: Optional[np.ndarray] = None
    dtype: FeatureDtype = FeatureDtype.CONTINUOUS

    def __post_init__(self) -> None:
        if len(self.feature_names) != self.values.shape[-1]:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must match "
                f"last dimension of values ({self.values.shape[-1]})"
            )
        if self.missing_mask is not None and self.missing_mask.shape != self.values.shape:
            raise ValueError(
                f"missing_mask shape {self.missing_mask.shape} must match "
                f"values shape {self.values.shape}"
            )


@dataclass
class ClinicalLabel:
    """
    Supervision target for a clinical sample.

    Attributes:
        task:        Identifier for the prediction task (e.g., 'cvd_risk_5yr').
        value:       The label value (int for classification, float for regression).
        label_type:  The prediction task type.
        confidence:  Optional reliability score for the label (0–1). None if unknown.
        source:      Provenance string describing how the label was derived.
    """

    task: str
    value: Any  # int | float — mypy does not support Union in dataclass fields cleanly
    label_type: LabelType
    confidence: Optional[float] = None
    source: str = "unknown"


@dataclass
class SampleMetadata:
    """
    Provenance metadata for a clinical sample.

    Attributes:
        dataset_id:              Registry key of the source dataset.
        split:                   Which partition this sample belongs to.
        preprocessing_version:   SHA-256 hash of the preprocessing config applied.
        created_at:              ISO 8601 timestamp of when this sample was processed.
    """

    dataset_id: str
    split: DataSplit
    preprocessing_version: str = "unprocessed"
    created_at: str = ""


@dataclass
class ClinicalSample:
    """
    The fundamental unit of data in Nāḍī-Q.

    Represents a single patient observation (or time-indexed sequence).
    Contains one or more feature modality blocks.

    Attributes:
        sample_id: Anonymized identifier — must NOT contain PII.
        features:  Dict mapping modality name → FeatureBlock.
        label:     Supervision target (optional; None for unsupervised use).
        metadata:  Provenance and split information.
    """

    sample_id: str
    features: dict[str, FeatureBlock]
    label: Optional[ClinicalLabel] = None
    metadata: SampleMetadata = field(
        default_factory=lambda: SampleMetadata(dataset_id="unknown", split=DataSplit.TRAIN)
    )

    def get_modality(self, name: str) -> FeatureBlock:
        """Return a feature block by name, raising KeyError if absent."""
        if name not in self.features:
            raise KeyError(
                f"Modality '{name}' not found. Available: {list(self.features.keys())}"
            )
        return self.features[name]

    @property
    def modalities(self) -> list[str]:
        """Return the names of all available modality blocks."""
        return list(self.features.keys())


@dataclass
class DatasetSchema:
    """
    Describes the structure of a dataset without instantiating any data.

    Used by models to understand input dimensionality before loading data.

    Attributes:
        dataset_id:    Registry key.
        modalities:    Dict mapping modality name → ModalityType.
        feature_dims:  Dict mapping modality name → feature dimensionality.
        label_def:     Human-readable description of the label.
        label_type:    Prediction task type.
        n_classes:     Number of classes (for classification tasks).
    """

    dataset_id: str
    modalities: dict[str, ModalityType]
    feature_dims: dict[str, int]
    label_def: str
    label_type: LabelType
    n_classes: Optional[int] = None


@dataclass
class DatasetStats:
    """
    Descriptive statistics computed from the training split only.

    IMPORTANT: Must never be computed from the test split.

    Attributes:
        n_samples:       Dict mapping split → sample count.
        class_counts:    Dict mapping class label → count (training split).
        feature_means:   Dict mapping modality name → per-feature means.
        feature_stds:    Dict mapping modality name → per-feature stds.
        missing_rates:   Dict mapping modality name → per-feature missing rates.
    """

    n_samples: dict[str, int]
    class_counts: Optional[dict[Any, int]] = None
    feature_means: Optional[dict[str, np.ndarray]] = None
    feature_stds: Optional[dict[str, np.ndarray]] = None
    missing_rates: Optional[dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseDatasetLoader(ABC):
    """
    Abstract interface for all dataset loaders in Nāḍī-Q.

    Every dataset loader must implement this interface. No model code
    may import a concrete loader class directly — loaders are resolved
    via the dataset registry and this abstract interface.

    Subclasses must implement:
        load_split()    — Return an iterable of ClinicalSample for a split.
        get_schema()    — Return the DatasetSchema (no data loaded).
        get_stats()     — Return DatasetStats (computed from training split only).
        dataset_id      — Property returning the registry key.
    """

    @abstractmethod
    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        """
        Return an iterable of ClinicalSample objects for the given split.

        Implementations should be lazy (generator) where possible to
        avoid loading entire datasets into memory.

        Args:
            split: Which dataset partition to load.

        Yields:
            ClinicalSample objects.
        """

    @abstractmethod
    def get_schema(self) -> DatasetSchema:
        """
        Return the dataset schema without loading any samples.

        Used by models to determine input dimensions before instantiation.
        Must be callable with no data loaded.
        """

    @abstractmethod
    def get_stats(self) -> DatasetStats:
        """
        Return descriptive statistics about the dataset.

        IMPORTANT: Statistics must be computed from the TRAINING split only.
        This method may load and iterate the training split internally,
        but must not expose test split statistics.
        """

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Unique registry key for this dataset loader."""
