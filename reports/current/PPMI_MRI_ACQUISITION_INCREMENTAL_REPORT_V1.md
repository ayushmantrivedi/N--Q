# NĀḌĪ-Q — INCREMENTAL PPMI MRI ACQUISITION VALIDATION REPORT
**Audit Date**: 2026-10-29 11:55:00  
**Audit Scope**: Incremental Validation of Newly Placed MRI Downloads under `F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI\`  
**Auditor**: Antigravity MRI Integrity & Data Provenance Agent  

---

## 1. EXECUTIVE SUMMARY & COHORT COUNTS
- **Expected Additional Downloads**: ~4 participants requested.
- **Actual New Participant Folders Discovered**: **5 participants** (`149006`, `151050`, `170472`, `183263`, `222893`).
- **Newly Verified Baseline 3D T1 Series**: **5 participants** (100% pass rate).
- **Previously Verified Count**: **5 participants** (`220494`, `229409`, `278950`, `311004`, `315185`).
- **Updated Total Verified Cohort (Batch 001)**: **10 participants** (`PREPROCESS_PENDING`).
- **Remaining Missing from Batch 001**: **14 participants** (Awaiting download from LONI/IDA).
- **Quarantined Extra Participants (Outside Batch 001)**: **9 participants** (Unchanged, untouched).
- **Frozen Reference Prototype Participants**: **15 participants** (In `MRI_raw\PPMI\`, untouched).
- **Zero Modifications**: No raw MRI files were modified, moved, renamed, or preprocessed.

---

## 2. AUDIT OF NEWLY DISCOVERED PARTICIPANTS (N=5)

All 5 new folders correspond directly to participants in the approved [ppmi_mri_batch_001_corrected_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_corrected_v1.csv). Each series was grouped by `SeriesInstanceUID`, ordered physically, tested for full 3D volumetric reconstruction, checked for finite values and positive variance, and reconciled against official PPMI baseline acquisition dates in `Magnetic_Resonance_Imaging` and `Xing_Core_Lab`:

| PATNO | manifest_visit | observed_visit | local_path | file_count | series_count | sequence | dimensions | voxel_spacing | identity_check | visit_check | reconstruction_check | final_status | reason |
| :---: | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `149006` | BL | BL | `MRI image download_dataset\PPMI\149006` | 384 | 1 | 3D T1-weighted | 512x512x384 | [0.5, 0.5] | `PASS` | `PASS (2022-07-14 BL)` | `PASS` | **`VERIFIED_BASELINE_T1`** | High-res 0.5mm 3D T1 baseline series. Identity, date, geometry, finite voxels, and 3D reconstruction verified. |
| `151050` | BL | BL | `MRI image download_dataset\PPMI\151050` | 188 | 1 | 3D T1-weighted | 256x256x188 | [1.0, 1.0] | `PASS` | `PASS (2022-06-24 BL)` | `PASS` | **`VERIFIED_BASELINE_T1`** | Verified 3D T1 structural baseline series. Identity, date, geometry, finite voxels, and 3D reconstruction verified. |
| `170472` | BL | BL | `MRI image download_dataset\PPMI\170472` | 192 | 1 | SAG 3D T1 FSPGR | 256x256x192 | [1.0, 1.0] | `PASS` | `PASS (2022-10-24 BL)` | `PASS` | **`VERIFIED_BASELINE_T1`** | Verified 3D T1 structural baseline series. Identity, date, geometry, finite voxels, and 3D reconstruction verified. |
| `183263` | BL | BL | `MRI image download_dataset\PPMI\183263` | 192 | 1 | SAG 3D T1 FSPGR | 256x256x192 | [1.0, 1.0] | `PASS` | `PASS (2023-06-02 BL)` | `PASS` | **`VERIFIED_BASELINE_T1`** | Verified 3D T1 structural baseline series. Identity, date, geometry, finite voxels, and 3D reconstruction verified. |
| `222893` | BL | BL | `MRI image download_dataset\PPMI\222893` | 192 | 1 | SAG 3D T1 FSPGR | 256x256x192 | [1.0, 1.0] | `PASS` | `PASS (2023-05-24 BL)` | `PASS` | **`VERIFIED_BASELINE_T1`** | Verified 3D T1 structural baseline series. Identity, date, geometry, finite voxels, and 3D reconstruction verified. |

---

## 3. REMAINING MISSING PARTICIPANTS IN BATCH 001 (N=14)

The following 14 approved Batch 001 participants have not yet been downloaded to local disk:
```
[235630, 214284, 126138, 111801, 533529, 450524, 444452, 200133, 464544, 398345, 156304, 183051, 345635, 412845]
```

---

## 4. VERIFIER EXECUTION & MANIFEST UPDATE

- **Verifier Script**: `python scripts/verify_mri_batch.py`
- **Validation Manifest Updated**: [ppmi_mri_batch_001_download_validation_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_download_validation_v1.csv)
- **Execution Log**:
  - Previously verified participants skipped as required.
  - All 5 new participants updated from `SELECTED` to `PREPROCESS_PENDING`.
  - Stored parameters: `selected_series_uid`, `reconstructed_shape`, `voxel_spacing`, `slice_thickness`, `dicom_file_count`, `eligible_series_count`.
- **Regression Tests**:
  - Command: `python -m pytest tests/data/test_ppmi_preprocessing.py -q`
  - Result: **12 passed in 17.54s** (all N=15 prototype invariants and visit-rejection checks green).

---

## 5. RECONCILED BATCH 001 OVERALL STATUS (N=24)

| Category | Count | Participant IDs |
| :--- | :---: | :--- |
| **Verified & Ready for Preprocessing** | **10** | `149006`, `151050`, `170472`, `183263`, `220494`, `222893`, `229409`, `278950`, `311004`, `315185` |
| **Pending Acquisition (Missing)** | **14** | `111801`, `126138`, `156304`, `183051`, `200133`, `214284`, `235630`, `345635`, `398345`, `412845`, `444452`, `450524`, `464544`, `533529` |
| **Total Approved Batch 001** | **24** | 100% baseline-verified candidate cohort |

