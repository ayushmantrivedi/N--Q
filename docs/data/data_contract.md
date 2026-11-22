# Data Contract — Nāḍī-Q

> **Status markers:**  
> **[A]** Established · **[B]** Hypothesis · **[C]** Experimental Claim · **[D]** Future Architecture

---

## 1. Purpose

This document defines the data interface contract that all dataset loaders in Nāḍī-Q must implement. It separates the *format* of data expected by models from the *source* of that data. This allows:

- Datasets to be swapped without rewriting models.
- Synthetic data to be used during development with no interface change.
- Real clinical datasets to be added later by implementing the same interface.

---

## 2. Core Data Structures

### 2.1 `ClinicalSample`

The fundamental unit of data in Nāḍī-Q is a `ClinicalSample`, representing a single patient observation (or a time-indexed sequence of observations).

```python
@dataclass
class ClinicalSample:
    sample_id: str                        # Anonymized identifier
    features: dict[str, FeatureBlock]     # Named modality blocks
    label: ClinicalLabel | None           # Supervision target (optional)
    metadata: SampleMetadata             # Provenance, split, timestamp
```

### 2.2 `FeatureBlock`

Each modality is represented as a `FeatureBlock`:

```python
@dataclass
class FeatureBlock:
    modality: ModalityType               # Enum: TABULAR, TIME_SERIES, IMAGE, GENOMIC
    values: np.ndarray                   # Feature matrix
    feature_names: list[str]             # Column names
    missing_mask: np.ndarray | None      # Boolean mask: True = missing
    dtype: FeatureDtype                  # Enum: CONTINUOUS, CATEGORICAL, BINARY
```

### 2.3 `ClinicalLabel`

```python
@dataclass
class ClinicalLabel:
    task: str                            # e.g., "cardiovascular_risk_5yr"
    value: int | float                   # Target value
    label_type: LabelType                # BINARY, MULTICLASS, REGRESSION
    confidence: float | None             # Label reliability (if known)
    source: str                          # Annotation provenance
```

### 2.4 `SampleMetadata`

```python
@dataclass
class SampleMetadata:
    dataset_id: str                      # Dataset registry key
    split: DataSplit                     # Enum: TRAIN, VAL, TEST
    created_at: str                      # ISO 8601 timestamp
    preprocessing_version: str           # Hash of preprocessing pipeline config
```

---

## 3. Dataset Loader Interface

All dataset loaders must implement `BaseDatasetLoader`:

```python
class BaseDatasetLoader(ABC):

    @abstractmethod
    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        """Return samples for a given split."""

    @abstractmethod
    def get_schema(self) -> DatasetSchema:
        """Return the schema: modalities, feature names, label definition."""

    @abstractmethod
    def get_stats(self) -> DatasetStats:
        """Return descriptive statistics (computed on training split only)."""

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Unique registry key for this dataset."""
```

**Rule:** No model code may import a specific dataset loader directly. Models receive `ClinicalSample` objects only. Dataset identity is resolved via the configuration system.

---

## 4. Modality Types [A]

| Modality | Description | Example Features |
|----------|-------------|-----------------|
| `TABULAR` | Static tabular biomarkers | Age, BMI, cholesterol, HbA1c |
| `TIME_SERIES` | Longitudinal measurements | Vital signs, lab values over time |
| `IMAGE` | Medical imaging | Chest X-ray, ECG trace (not yet supported) |
| `GENOMIC` | Genetic features | SNPs, gene expression (not yet supported) |

**Status of IMAGE and GENOMIC:** [D] Future architecture — not implemented, not expected in initial experiments.

---

## 5. Dataset Registry

Datasets are registered by a string key in `configs/data/<dataset_id>.yaml`.

Required fields in a dataset config:

```yaml
dataset_id: synthetic_tabular_v1
loader_class: nadiq.data.loaders.synthetic.SyntheticTabularLoader
data_root: null           # Path to raw data (null = generated on-the-fly)
task: binary_risk
modalities: [tabular]
n_samples: 5000           # For synthetic loaders
n_features: 32
seed: 42
split_strategy:
  type: stratified_kfold
  n_folds: 5
  test_fraction: 0.2
```

---

## 6. What This Contract Excludes

- **Raw data files** — never committed to this repository.
- **Patient identifiers** — no PII of any kind in code or configs.
- **Fabricated clinical data** — synthetic data used for testing must be clearly labelled as synthetic and must not be used to make clinical claims.
- **Dataset download logic** — loaders reference data that has already been obtained through proper ethical/legal channels.

---

## 7. Currently Supported Datasets

| Dataset ID | Type | Status | Notes |
|------------|------|--------|-------|
| `synthetic_tabular_v1` | Synthetic Gaussian | [C] To implement | For unit/integration tests only |
| `mimic_iii_icu` | Real ICU EHR | [D] Future | Requires PhysioNet credentialing |
| `ukbb_tabular` | Real UK Biobank | [D] Future | Requires approved application |

**Important:** No real clinical datasets are included or implied to be available. The `mimic_iii_icu` and `ukbb_tabular` entries are placeholders only, to define the interface they would need to implement.

---

## 8. Data Integrity Requirements

- Preprocessing pipelines must be fitted on the training split only.
- Statistics (mean, std, imputation values) must not be computed from the test split.
- The preprocessing pipeline configuration must be hashed and stored alongside any saved model or result.
- Reproducibility requires that the same pipeline config produce bit-identical preprocessed features given identical raw inputs.
