# NĀḌĪ-Q — CRITICAL MRI ACQUISITION-TYPE RE-AUDIT REPORT
**Audit Date**: 2026-10-29 12:12:00  
**Audit Purpose**: Targeted Re-Audit of DICOM Acquisition Headers to Distinguish Genuine 3D T1 Volumetric Acquisitions from 2D Multislice Sequences across Batch 001  
**Auditor**: Antigravity MRI Integrity & Data Provenance Agent  

---

## 1. EXECUTIVE SUMMARY & CLINICAL VERDICT

A rigorous, read-only re-audit was conducted on all 10 participants previously marked `VERIFIED_BASELINE_T1` in Batch 001 to resolve concerns regarding 2D vs. 3D acquisition.

* **Exact Number of Eligible Genuine 3D T1 Scans**: **10 / 10** (`VERIFIED_3D_T1`).
* **Ineligible 2D Multislice Scans Found**: **0 / 10**.
* **Ambiguous Scans Needing Manual Review**: **0 / 10**.
* **Identity or Visit Mismatches**: **0 / 10**.
* **Verifier Logic Audit**: **CRITICAL DEFECT CONFIRMED & FIXED**. The original verifier checked only substring names and NumPy array dimensions (`len(vol.shape) == 3`), which would have allowed stacked 2D multislice scans to pass. The verifier has now been hardened with physical `MRAcquisitionType == '3D'` enforcement and backed by regression tests (14/14 tests passing).

---

## 2. PHYSICAL MRI EVIDENCE & WHY 2D WAS SUSPECTED

Stacking 2D multislice DICOM files into a 3D array in NumPy or Python does **not** prove a 3D volumetric acquisition. A true 3D acquisition requires two-dimensional phase encoding (phase and partition/slice encoding across a single excitation volume) rather than sequential 2D slice-selective RF excitation.

### Why a Manual Inspector May Have Suspected 2D:
1. **Single-Frame DICOM Distribution**: Standard PPMI data downloads distribute 3D volumes as individual 2D image slice files (`.dcm`, SOP Class UID `1.2.840.10008.5.1.4.1.1.4` MR Image Storage) rather than multi-frame DICOM (`1.2.840.10008.5.1.4.1.1.4.1`). In standard OS file browsers or basic image viewers, each file appears as a standalone 2D slice.
2. **Siemens `DIS2D` Image Type**: Siemens 3D MPRAGE headers record `ImageType: ['ORIGINAL', 'PRIMARY', 'M', 'NORM', 'DIS2D']`. In Siemens nomenclature, `DIS2D` signifies **2D in-plane gradient distortion correction** applied to reconstructed slices, not a 2D slice acquisition.
3. **`ScanningSequence: GR` / `['GR', 'IR']`**: In DICOM, Gradient Recalled is tagged as `GR`. Users commonly conflate `GR` with "2D Gradient Echo (GRE)". However, 3D MPRAGE and 3D FSPGR/BRAVO are physically gradient-echo readouts preceded by non-selective 180° inversion recovery (`IR`) pulses.

### Proof of Genuine 3D T1 Volumetric Physics:
1. **DICOM Tag (0018, 0023) `MRAcquisitionType`**: **`3D`** across 100% of all 2,096 slices inspected across all 10 participants.
2. **Inversion Preparation (`InversionTime` / `TI`)**: All 10 series possess non-zero `TI` (900 ms on Siemens/Toshiba; 400 ms on GE), proving non-selective inversion-recovery 3D T1 structural sequences (MPRAGE / IR-FSPGR / BRAVO), the international standard for ADNI/PPMI structural morphometry.
3. **Continuous Sub-Millimeter / 1.0mm Partition Slabs**: Partitions range from 188 to 384 slices with slice thickness $\le 1.0\text{ mm}$ covering the entire cranium continuously without slice gaps. (2D multislice scans typically have 20–40 slices with 3–5 mm thickness and interslice gaps).

---

## 3. COMPREHENSIVE PARTICIPANT RE-AUDIT TABLE (N=10)

| PATNO | Series Instance UID | MRAcquisitionType | Pulse Sequence & Details | Matrix & Slices | Spacing & Thick | TR / TE / TI / FA | Scanner & Field | Prior Status | Revised Status | Verdict |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `220494` | `...05206077173343538684` | **`3D`** | `*tfl3d1_ns` (SAG 3D MPRAGE, IR-TurboFLASH) | 256x256x192 | 1.0 / 1.0 mm | 2300 / 2.74 / 900 ms / 9° | Siemens Verio 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `229409` | `...0170623315771408403` | **`3D`** | `*tfl3d1_ns` (SAG 3D MPRAGE, IR-TurboFLASH) | 256x256x192 | 1.0 / 1.0 mm | 2300 / 2.74 / 900 ms / 9° | Siemens Verio 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `278950` | `...6341564590483236078` | **`3D`** | `*tfl3d1_16ns` (SAG 3D T1 MPRAGE, IR-TurboFLASH) | 256x256x192 | 1.0 / 1.0 mm | 2300 / 2.69 / 900 ms / 9° | Siemens Skyra 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `311004` | `...8582524138894402741` | **`3D`** | `3D T1-weighted` (GE IR-FSPGR / BRAVO) | 256x256x188 | 1.0 / 1.0 mm | 8.67 / 3.22 / 400 ms / 11° | GE SIGNA Arch 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `315185` | `...6923374003565700395` | **`3D`** | `SAG 3D T1 FSPGR` (GE IR-FSPGR / BRAVO) | 256x256x192 | 1.0 / 1.0 mm | 6.99 / 2.99 / 400 ms / 11° | GE SIGNA Arch 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `149006` | `...614367011982540875` | **`3D`** | `FFE3D` (3D T1-weighted IR-Fast Field Echo) | 512x512x384 | 0.5 / 1.0 mm | 5.0 / 1.90 / 900 ms / 15° | Toshiba Vantage 1.5T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `151050` | `...5998767067871609939` | **`3D`** | `3D T1-weighted` (GE IR-FSPGR / BRAVO) | 256x256x188 | 1.0 / 1.0 mm | 7.36 / 3.05 / 400 ms / 11° | GE MR750 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `170472` | `...02498165235185709749` | **`3D`** | `SAG 3D T1 FSPGR` (GE IR-FSPGR / BRAVO) | 256x256x192 | 1.0 / 1.0 mm | 6.98 / 2.98 / 400 ms / 11° | GE SIGNA Arch 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `183263` | `...6562993699364865606` | **`3D`** | `SAG 3D T1 FSPGR` (GE IR-FSPGR / BRAVO) | 256x256x192 | 1.0 / 1.0 mm | 6.98 / 2.97 / 400 ms / 11° | GE SIGNA Arch 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |
| `222893` | `...4083434051961546046` | **`3D`** | `SAG 3D T1 FSPGR` (GE IR-FSPGR / BRAVO) | 256x256x192 | 1.0 / 1.0 mm | 6.98 / 2.97 / 400 ms / 11° | GE SIGNA Arch 3T | `PREPROCESS_PENDING` | `PREPROCESS_PENDING` | **`VERIFIED_3D_T1`** |

---

## 4. PPMI SOURCE METADATA RECONCILIATION

Comparison with authoritative clinical metadata tables confirmed:
1. **`Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv`**:
   - `EVENT_ID == 'BL'`
   - `MRI_SEQ_T1_WEIGHTED == 'Yes'` for all 10 participants.
2. **`Magnetic_Resonance_Imaging__MRI__18Sep2026.csv`**:
   - `EVENT_ID == 'BL'`
   - `MRICMPLT == 1.0` (Completed acquisition)
   - Month/Year `INFODT` strictly matches DICOM `StudyDate` for all 10 participants.

---

## 5. AUDIT OF VERIFIER DEFECT & SMALLEST CODE FIX

### Exact Defect Identified:
In `scripts/verify_mri_batch.py`, sequence verification relied on:
```python
# DEFECTIVE ORIGINAL LOGIC
if 't1' not in desc and 'mprage' not in desc and 'spgr' not in desc:
    continue
success, vol = reconstruct_volume(slices)
if len(vol.shape) != 3 or vol.shape[2] < 2:
    continue
```
* **Vulnerability**: A 2D multislice scan (e.g. `AX T1 SE` with 20 slices of 5mm thickness) contains `"t1"` in `desc`, and `np.stack(..., axis=-1)` produces a 3D NumPy array `(256, 256, 20)`. The original verifier would have falsely promoted this 2D sequence to `VERIFIED`.

### Code Fix Applied:
`scripts/verify_mri_batch.py` was updated with the following checks:
1. **Mandatory 3D Acquisition Type**:
   ```python
   acq_types = {str(getattr(s, 'MRAcquisitionType', '')).strip().upper() for s in slices}
   if '2D' in acq_types:
       failed_series.append((suid, 'INELIGIBLE_2D_ACQUISITION'))
       continue
   if '3D' not in acq_types:
       ambiguous_series.append((suid, 'AMBIGUOUS_NOT_3D_ACQUISITION'))
       continue
   ```
2. **2D Sequence Keyword Filtering**: Explicitly rejects `' 2d'`, `'_2d'`, `'2-d'`.
3. **Partition & Thickness Constraints**: Rejects series with slice thickness $> 2.0\text{ mm}$ or partition count $< 100$.
4. **Manifest Audit Trail**: Writes `acquisition_type_verified = 'VERIFIED_3D_T1'` and `mr_acquisition_type = '3D'` to the active manifest.

### Regression Tests Added:
In `tests/data/test_ppmi_preprocessing.py`:
* `test_verify_mri_batch_rejects_2d_acquisition`: Proves that a mock 2D series (`MRAcquisitionType = '2D'`, 20 slices, 5mm) is marked `FAILED` with `VERIFIED_2D_NOT_ELIGIBLE` and rejected from preprocessing.
* `test_verify_mri_batch_accepts_genuine_3d_acquisition`: Proves that a genuine 3D series (`MRAcquisitionType = '3D'`, 120 slices, 1mm) is promoted to `PREPROCESS_PENDING` with `VERIFIED_3D_T1`.
* **Test Suite Execution**:
  ```powershell
  python -m pytest tests/data/test_ppmi_preprocessing.py -q
  ```
  **14 passed in 17.18s** (100% pass rate).

---

## 6. VALIDATION MANIFEST AUDIT TRAIL

The active manifest [ppmi_mri_batch_001_download_validation_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_download_validation_v1.csv) was re-evaluated through the updated verifier.
* All 10 participants now have:
  - `status = 'PREPROCESS_PENDING'`
  - `verification_status = 'VERIFIED_3D_T1'`
  - `acquisition_type_verified = 'VERIFIED_3D_T1'`
  - `mr_acquisition_type = '3D'`
* The remaining 14 Batch 001 participants remain `SELECTED` / `PENDING` awaiting download.

