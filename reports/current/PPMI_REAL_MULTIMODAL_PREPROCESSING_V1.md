# NADIQ - PPMI REAL MULTIMODAL PREPROCESSING V1
**Date**: 2026-09-24
**Status**: PREPROCESSED & CONTRACT VERIFIED
**Prepared by**: Antigravity automated pipeline

---

## 1. MRI PREPROCESSING STATISTICS
- **Subjects Processed**: 15
- **Target Tensor Shape**: (128, 128, 128) (Configured for existing ?-VAE)
- **Target Voxel Spacing**: (1.5, 1.5, 1.5)
- **Intensity Normalization**: Min-Max [0, 1] scaling
- **Storage**: Tensors saved as .npy at F:\NADI-Q data\PPMI\MRI_processed\<PATNO>_BL_t1_tensor.npy

All 15 MRI scans were successfully resampled from their native orientations/spacings to the target uniform tensor grid without fabricating missing modalities or inferring substitute visits.

---

## 2. BIOMARKER AGGREGATION & MISSINGNESS
- **Input File**: Current_Biospecimen_Analysis_Results_18Sep2026.csv
- **Aggregation Policy**: Standardized duplicates (e.g. repeated assays) at the BL visit by computing the mean for numerical TESTVALUEs.
- **Missingness Preserved**: Missing values were explicitly preserved as NaN. No imputation was applied to avoid data leakage before the pipeline feature contract.

### Missingness by Priority Marker
- **MTBR_TAU243**: 100.0% missing
- **IU_ABeta40_CSF**: 100.0% missing
- **IU_ABeta42_CSF**: 100.0% missing
- **IU_pTau181_CSF**: 100.0% missing
- **IU_NFL_plasma_mean**: 100.0% missing
- **IU_GFAP_plasma_mean**: 100.0% missing
- **Lilly_ptau217p**: 100.0% missing
- **CSFSAA**: 100.0% missing

---

## 3. GAIT KINEMATICS REPRESENTATION
- **Input File**: Gait_Data___Arm_swing__Opals__18Sep2026.csv
- **Feature Count**: 57 derived kinematic scalars.
- **Temporal Constraint Checked**: As no raw sensor time-series are available locally for this cohort, the kinematics represent a **single-visit derived snapshot**. The CTCLM interface will receive a static feature vector, and must NOT be described as having learned longitudinal temporal dynamics from these 15 subjects.
- **Aggregation Policy**: Any duplicate trials at BL were aggregated using the mean across the 60 features.
- **Overall Missingness**: ~23.2% average feature missingness across the 60 scalars.

---

## 4. MULTIMODAL QUALITY CONTROL & LEAKAGE REPORT
- **Participant Duplication**: PASSED (0 duplicate PATNOs in final manifest)
- **Visit Bleed**: PASSED (100% of rows strictly constrained to BL)
- **Label Leakage**: PASSED (No target/outcome clinical features are present in the modality feature matrices)
- **Modality Substitution**: PASSED (No cross-participant or cross-visit substitution occurred)
- **Synthetic Data**: PASSED (No synthetic or generated records were introduced)
- **Final N**: Exactly 15 traceable subjects.

---

## 5. OUTPUT DATA CONTRACT
The final subject-level multimodal matrix mapping PATNO to clinical labels, MRI tensor paths, and specific biomarker/gait feature columns is written to:
F:\NADI-Q data\PPMI\manifests\ppmi_real_multimodal_preprocessed_v1.csv

*(See 	ests/data/test_ppmi_preprocessing.py for programmatic verification of these contracts).*
