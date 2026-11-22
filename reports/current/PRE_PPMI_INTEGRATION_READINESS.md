# NĀḌĪ-Q Pre-PPMI Integration Readiness Report

**Date**: 2026-09-18  
**Status**: READY FOR PPMI DATA INGESTION  
**Tests**: 268 passed / 0 failed (full suite, including 41 new integration tests)

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `src/nadiq/data/patient_timeline.py` | PatientTimeline, ModalityAvailability, ClinicalState |
| `src/nadiq/data/modality_assembler.py` | ModalityAssembler, AssembledSample, AssemblerConfig |
| `src/nadiq/data/missing_modality.py` | Missing-modality utilities and availability tensors |
| `src/nadiq/models/multimodal/cross_modal_alignment.py` | CrossModalAlignment interface (deferred) |
| `src/nadiq/data/sources/ppmi.py` | PPMIAdapter skeleton, PPMISchemaConstants, PPMICohort |
| `tests/integration/test_ppmi_integration_layer.py` | 41 software-correctness tests |

No existing files were modified. The frozen architecture (β-VAE, QVBE, CTCLM, QCDF) is untouched.

---

## 2. Actual Interfaces

### 2.1 PatientTimeline
```python
PatientTimeline(
    patient_id: str,               # PPMI PATNO
    event_id: str,                 # PPMI EVENT_ID (e.g. "BL", "V02")
    clinical_state: ClinicalState, # HC / PD / PROD / MCI / UNKNOWN
    modality_availability: ModalityAvailability(mri, biomarker, kinematics),
    modality_refs: dict[str, ModalityReference],
    dataset_id: str,
    metadata: dict,
)
# Key property:
timeline.sample_key  →  "PATNO|EVENT_ID"
```

### 2.2 ModalityAssembler
```python
config = AssemblerConfig(
    mri_embedding_dim=4,        # QVBE n_qubits
    biomarker_embedding_dim=4,  # QVBE n_qubits
    kinematics_embedding_dim=32 # CTCLM latent_dim
)
assembler = ModalityAssembler(config)

sample = assembler.assemble(
    timeline=PatientTimeline,
    mri_embedding=Tensor(B, 4) | None,
    biomarker_embedding=Tensor(B, 4) | None,
    kinematics_embedding=Tensor(B, 32) | None,
) -> AssembledSample
```

**AssembledSample fields**: `patient_id`, `event_id`, `bundle (ModalityBundle)`, `availability`, `metadata`

### 2.3 CrossModalAlignment (Interface Only)
```python
class CrossModalAlignment(ABC):
    def align(self, inputs: CrossModalInput) -> CrossModalOutput:
        raise NotImplementedError  # DEFERRED — no algorithm chosen
```

```
CrossModalInput:
  z_mri:        (B, 4)
  z_bio:        (B, 4)
  z_kin:        (B, 32)
  availability: (B, 3)
  patient_ids:  list[str]
  event_ids:    list[str]

CrossModalOutput:
  aligned_z_mri:    (B, alignment_dim)
  aligned_z_bio:    (B, alignment_dim)
  aligned_z_kin:    (B, alignment_dim)
  shared_z:         (B, alignment_dim) | None
  alignment_loss:   scalar Tensor | None
  patient_ids, event_ids  (forwarded for traceability)
```

### 2.4 PPMIAdapter Skeleton
```python
PPMISchemaConstants.PATNO    = "PATNO"
PPMISchemaConstants.EVENT_ID = "EVENT_ID"
PPMICohort.PD / .HEALTHY_CONTROL / .PRODROMAL / .SWEDD

adapter.build_patient_timeline(patno, event_id, cohort) -> PatientTimeline
adapter.map_columns({"semantic_name": "actual_csv_col"})  # call after file inspection
adapter.load_split(DataSplit.TRAIN)  # raises NotImplementedError until data present
adapter.get_schema()                 # returns partial schema, feature_dims={}
```

### 2.5 Missing-Modality Utilities
```python
make_zero_embedding(batch_size, embedding_dim)   -> Tensor(B, D)  # zeros only
make_availability_tensor(ModalityAvailability)   -> Tensor(3,)    # {0.0, 1.0}
validate_embedding_against_availability(...)     # raises on fabrication attempts
describe_availability_pattern(availability)      -> str
MISSING_MODALITY_PATTERNS                        # named dict of common patterns
```

---

## 3. Tensor Dimensions (QCDF Contract)

| Modality | Source | Dim | Availability bit |
|----------|--------|-----|-----------------|
| MRI | β-VAE → QVBE | (B, 4) | `[0]` |
| Biomarker | Tabular → QVBE | (B, 4) | `[1]` |
| Kinematics | Raw → CTCLM | (B, 32) | `[2]` |
| Availability mask | ModalityAssembler | (3,) or (B, 3) | — |
| QCDF fused input | 3×proj(16) + 3×avail | (B, 51) | — |
| QCDF output (logits) | Classifier | (B, num_classes) | — |

---

## 4. Test Results

```
tests/integration/test_ppmi_integration_layer.py  41 passed
Full suite:                                        268 passed, 4 warnings
Exit code: 0
```

**Test categories:**
- `TestPatientTimeline` — 11 tests (construction, PPMI IDs, modality refs)
- `TestModalityAssembler` — 5 tests (all/partial/none modalities, batch, dim validation)
- `TestTraceability` — 6 tests (patient_id, event_id, metadata, PPMI style)
- `TestMissingModalityUtilities` — 13 tests (zero tensors, availability tensor, validation, error cases)
- `TestQCDFIntegration` — 6 tests (**⚠️ SYNTHETIC TENSORS — software interface test only**)

---

## 5. Current Readiness Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| PatientTimeline | ✅ READY | Full construction and query interface |
| ModalityAssembler | ✅ READY | All availability patterns tested |
| Missing-modality handling | ✅ READY | Zero tensor + mask contract enforced |
| Traceability | ✅ READY | patient_id + event_id preserved end-to-end |
| QCDF structural integration | ✅ READY | Forward pass shape-correct for all patterns |
| CrossModalAlignment | 🔶 INTERFACE ONLY | Algorithm deferred pending paired data |
| PPMIAdapter skeleton | 🔶 SKELETON | Schema defined, data loading not implemented |
| PPMI column mapping | ❌ REQUIRES FILES | No guesses — inspect real CSVs first |

---

## 6. Unresolved Items Requiring Real PPMI Metadata

1. **Column names**: The `_COLUMN_MAP` in `PPMIAdapter` is empty by design. Motor (UPDRS), cognitive, biospecimen, and imaging column names must be populated only after downloading and inspecting the actual PPMI CSV files from `ida.loni.usc.edu`.

2. **Cohort sub-selection**: PPMI includes PD, HC, PRODROMAL, SWEDD, and several longitudinal sub-cohorts. Which sub-cohorts to include must be decided after reviewing the study design.

3. **Visit coverage**: PPMI visit codes vary by cohort and data release. The `visit_ids` filter in `PPMIAdapterConfig` should be populated with the actual visit codes present in the downloaded data.

4. **MRI file naming convention**: PPMI MRI files use a PATNO-based BIDS-adjacent structure. The exact path template must be mapped in `PPMIAdapter` after data inspection.

5. **Cross-modal alignment algorithm**: `CrossModalAlignment.align()` raises `NotImplementedError`. The algorithm choice (contrastive, CCA, cross-attention) requires empirical analysis of paired embedding geometry once PPMI data produces real `z_mri`, `z_bio`, `z_kin` tensors for the same subjects.

6. **Subject-level splits**: PPMI provides a pre-defined train/test cohort structure in some releases. `PPMIAdapter.load_split()` must implement subject-level splitting by `PATNO` (not by row) — the same strategy validated in `run_v2_biomarker_real.py`.

---

## 7. Next Steps (Post-PPMI Data Acquisition)

1. Download PPMI data from `https://ida.loni.usc.edu/`
2. Inspect CSV schema → populate `PPMIAdapter.map_columns()`
3. Implement `PPMIAdapter.load_split()` returning real `ClinicalSample` objects
4. Run `PPMIAdapter.get_stats()` on training split only
5. Run modality encoders (β-VAE/QVBE, QVBE, CTCLM) on real PPMI data
6. Inspect `z_mri`, `z_bio`, `z_kin` embedding geometry to guide alignment algorithm choice
7. Implement a concrete `CrossModalAlignment` subclass
8. Evaluate full QCDF pipeline on real paired subjects
