# TADPOLE Public Copy Audit

This audit evaluates the public subset of the TADPOLE dataset retrieved from `https://raw.githubusercontent.com/kramachandrashenoy/TADPOLE/master/TADPOLE_D1_D2.csv`.

**Date of Audit**: 2026-09-10
**Status**: SUITABLE WITH CAVEATS

## 1. File Metadata
- **Source**: `https://github.com/kramachandrashenoy/TADPOLE`
- **File Name**: `TADPOLE_D1_D2.csv`
- **Size**: 1.37 MB (1,431,634 bytes)
- **SHA-256**: `29ad504ded92793e0064b94c871bde24ba7e5f266c4d647aeace3a4c138b3820`

## 2. Row and Column Inventory
- **Rows**: 5,389
- **Columns**: 51 (Truncated compared to official TADPOLE which contains >1900 columns)
- **Date Range**: 2005-04-11 to 2017-10-01
- **Unique Patients (PTIDs / RIDs)**: 759

## 3. D1/D2 Membership
- **D1 rows**: 5,354 (724 unique patients)
- **D2 rows**: 3,159 (344 unique patients)
- **Overlap**: 336 patients appear in both D1 and D2 subsets.
  > [!WARNING]
  > Primary benchmark must ensure only D1 rows are selected to prevent train/test contamination with D2.

## 4. Feature Availability
The dataset includes standard clinical and MRI volumetric features but **completely lacks PET imaging and CSF biomarkers**:

**Available Core Features:**
- `AGE`, `PTGENDER`, `PTEDUCAT`, `APOE4`
- `MMSE`, `CDRSB`, `ADAS11`, `ADAS13`, `RAVLT_immediate`
- `Hippocampus`, `WholeBrain`, `Entorhinal`, `MidTemp`

**Missing Core Features (100% missing):**
- `FDG`, `AV45` (PET)
- `ABETA`, `TAU`, `PTAU` (CSF)

## 5. Follow-Up Statistics
- **Patients with ≥2 visits**: 737
- **Median Follow-up**: 1,069 days (~35 months)
- **Max Follow-up**: 4,213 days (~11 years)
- **Follow-up Thresholds**:
  - ≥ 12M: 86.8%
  - ≥ 24M: 70.0%
  - ≥ 36M: 45.9%
  - ≥ 48M: 26.6%
  - ≥ 60M: 14.9%

## 6. Integration and Compatibility
- **Numeric Transition Codes**: The `DXCHANGE` column uses numeric codes (1-9) rather than pure textual states. The `normalize_tadpole_diagnosis` framework and `tadpole_inspect.py` scripts have been updated to support transitions `4` (MCI to Dementia) through `9` properly.
- **Config Adjustments**: The missing PET and CSF features (`FDG`, `AV45`, `ABETA`, `TAU`, `PTAU`) have been explicitly commented out in `configs/features/tadpole_core_v1.yaml` to prevent the dataset loader's strict validation from failing.
- **Testing**: `pytest -q` succeeds (`66/66` passed). The existing TADPOLE loader handles the dataset smoothly given the config tweaks.

## Final Verdict
**SUITABLE WITH CAVEATS**. This dataset serves as a viable, immediately usable surrogate for development and testing of the core temporal engine. However, the lack of PET/CSF modalities means the "multimodal" capabilities of Nāḍī-Q cannot be fully validated on this specific copy. It should be replaced with the official, full ADNI dataset before generating final scientific claims or testing the full sensor-fusion capability.
