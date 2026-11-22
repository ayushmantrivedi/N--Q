# NADIQ - PPMI PAIRED COHORT GATE V1
**Date**: 2026-09-19
**Status**: COMPLETE
**Inputs**: Real PPMI metadata only. No synthetic data. No model changes.
**Prepared by**: Antigravity automated pipeline

---

## 1. SOURCE TABLES AND JOIN KEYS

| Table | Role | Primary Join Key |
|-------|------|-----------------|
| Participant_Status_18Sep2026.csv | Clinical label | PATNO |
| Magnetic_Resonance_Imaging__MRI__18Sep2026.csv | MRI completion flag | PATNO + EVENT_ID_NORM |
| Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv | T1 confirmation + QC | PATNO + EVENT_ID_NORM |
| MRIQC_18Sep2026.csv | Image quality metrics | PATNO + EVENT_ID_NORM |
| Current_Biospecimen_Analysis_Results_18Sep2026.csv | Biomarker presence | PATNO + EVENT_ID_NORM (via CLINICAL_EVENT) |
| Gait_Data___Arm_swing__Opals__18Sep2026.csv | Gait kinematics | PATNO + EVENT_ID_NORM (via VISNO) |

---

## 2. VISIT NORMALISATION RULE

Raw VISNO codes in gait table -> canonical EVENT_ID_NORM:

  Rule: strip whitespace, uppercase, then zero-pad single digit after V
  BL   -> BL
  V01  -> V01
  V04  -> V04
  V06  -> V06
  V08  -> V08
  V09  -> V09
  V10  -> V10
  V12  -> V12
  V2   -> V02     (non-standard; confirmed equivalent by PPMI visit schedule)
  V8   -> V08     (non-standard; confirmed equivalent)
  v10  -> V10     (case variant)
  v11  -> V11     (case variant)
  v6   -> V06     (case variant)
  v8   -> V08     (case variant)
  NaN  -> excluded (1 row dropped)

Applied identically to: MRI EVENT_ID, Xing EVENT_ID, MRIQC EVENT_ID, Biospecimen CLINICAL_EVENT.
Gait VISNO is the source column; EVENT_ID_NORM is the normalised join key.

---

## 3. FILTERING RULES

### MRI Filter
  REQUIRED: MRICMPLT == 1  (scan actually completed)
  PREFERRED: MRI_SEQ_T1_WEIGHTED == 'Yes' in Xing Core Lab table
  QC (Xing): MRI_IMAGE_ACCEPTABLE == 'Yes'
  QC (MRIQC): subject appears in MRIQC_18Sep2026.csv
  Join key: PATNO + EVENT_ID_NORM (from MRI_Acquisition table)

### Biospecimen Filter
  REQUIRED: PATNO + EVENT_ID_NORM present in Current_Biospecimen_Analysis_Results
  No minimum test count required (minimum observed = 2 assays)
  Specimen type recorded but not filtered (CSF/Plasma/Serum all accepted)
  Join key: PATNO + CLINICAL_EVENT (normalised to EVENT_ID_NORM)

### Gait Filter
  REQUIRED: PATNO + VISNO present in Gait_Data___Arm_swing__Opals table
  REQUIRED: VISNO is not NaN
  VISNO -> EVENT_ID_NORM mapping applied (documented above)
  No minimum feature-non-null requirement applied (SP_U can be null in some rows)
  Join key: PATNO + EVENT_ID_NORM (derived from VISNO)

### Clinical Label
  From Participant_Status: COHORT field
  1=PD, 2=HC, 3=SWEDD, 4=PRODROMAL, 5=GenCohortPD, 6=GenCohortUnaff
  All cohorts retained in manifest; not filtered
  Join key: PATNO

---

## 4. CANDIDATE MANIFEST

File: F:\NADI-Q data\PPMI\manifests\ppmi_same_visit_triple_candidates.csv
Rows: 278 (gait universe = 291 less 13 duplicate VISNO-PATNO after normalisation)
Columns:
  PATNO, EVENT_ID, gait_visit_id_orig
  clinical_state, ENROLL_STATUS
  mri_available, t1_available, mri_qc_xing, mriqc_available, mri_scan_date
  biomarker_available, bio_testname_count, bio_types
  has_csf, has_plasma, has_serum
  gait_available, sp_u, n_gait_features
  eligible_triple, eligible_t1_triple, eligible_qc_triple
  exclusion_reason

---

## 5. EXCLUSION BREAKDOWN (from 291 gait sessions)

| Reason | Sessions excluded |
|--------|-----------------|
| no_mri_completed | 186 |
| no_biospecimen | 84 |
| (may overlap) | - |
| Eligible (MRI+Bio+Gait) | 52 |

The primary bottleneck is MRI: 199 gait subjects, but only 92 of their gait
session visits have a completed MRI at the same PATNO+EVENT_ID.

---

## 6. VERIFIED COHORT COUNTS

### COHORT A: MRI + Biospecimen + Gait (same PATNO + EVENT_ID)
  Sessions: 52
  Subjects: 52  (all subjects have exactly 1 eligible triple session)
  Clinical state:
    PRODROMAL: 37
    PD: 12
    HC: 3
  MRI filter: MRICMPLT=1  (any MRI sequence confirmed)

### COHORT B (strict): T1 + Biospecimen + Gait (same visit)
  Sessions: 48
  Subjects: 48
  Clinical state:
    PRODROMAL: 36
    PD: 10
    HC: 2
  Filter: MRI_SEQ_T1_WEIGHTED=Yes in Xing, same EVENT_ID

### COHORT C (strict + QC): T1 + Bio + Gait + MRIQC evaluated (same visit)
  Sessions: 23
  Subjects: 23
  Clinical state:
    PRODROMAL: 15
    PD: 6
    HC: 2
  Note: MRIQC table only covers BL sessions. This biases COHORT C to BL only.

### PD subset (Cohort A filter):
  Sessions: 12  |  Subjects: 12

### HC subset (Cohort A filter):
  Sessions: 3  |  Subjects: 3

### PRODROMAL subset (Cohort A filter):
  Sessions: 37  |  Subjects: 37

---

## 7. MULTIVISIT ANALYSIS (eligible 52 subjects)

### MRI visits (across all visits, not just triple visit):
  mean=2.29  median=2  max=4  min=1
  >=2 visits: 39/52 subjects
  >=3 visits: 22/52 subjects

### T1-weighted visits (Xing confirmed):
  mean=2.26  median=2  max=4
  >=2 visits: 36/52 subjects

### Biospecimen visits (across all visits):
  mean=1.40  median=1  max=10
  >=2 visits: 8/52 subjects
  >=4 visits: 3/52 subjects
  NOTE: Biospecimen is sparse for this cohort.

### Gait visits (across all visits):
  mean=1.12  median=1  max=3
  >=2 visits: 5/52 subjects

### Same-visit triple sessions per participant:
  Exactly 1: 52/52 subjects
  2 or more: 0 subjects
  FINDING: No participant has 2 or more visits with all 3 modalities simultaneously.

---

## 8. BIOMARKER TYPES AT ELIGIBLE TRIPLE SESSIONS (N=52)

| Specimen | Sessions with data |
|----------|--------------------|
| CSF | 41 / 52 |
| Plasma | 34 / 52 |
| Serum | 4 / 52 |

Bio assay count per session:
  Mean: 13 assays  |  Max: 186  |  Min: 2

Top assays present at eligible triple sessions:
  eMTBR-TAU243 (43 subjects)
  ABeta40 (43)
  ABeta42 (43)
  pTau181 (43)
  NFL (34)
  GFAP (34)
  Ptau217p (25)
  skin_synSAA R&D-v2 10h (13)
  skin_synSAA R&D-v2 24h (13)

These are clinically meaningful quantitative plasma/CSF biomarkers.

---

## 9. COHORT B: PARTICIPANT-LEVEL TRIPLE (any visit per modality)

  MRI + Bio + Gait (any visit): 111 subjects
    PRODROMAL: 66
    PD: 42
    HC: 3

  T1 + Bio + Gait (any visit): 74 subjects
    PRODROMAL: 51
    PD: 21
    HC: 2

  NOTE: Previous audit reported 126; revised count of 111 reflects exact
  PATNO-level intersection after re-computation on same tables.
  Difference due to deduplication of VISNO normalisation (some visits
  appeared twice under different VISNO codes mapping to same EVENT_ID_NORM).

---

## 10. COHORT C: PARTIALLY OBSERVED MRI+BIO LONGITUDINAL

  MRI + Bio (any visit): 2,665 subjects
    PRODROMAL: 1,341
    PD: 1,003
    HC: 256
    SWEDD: 65

  Of these with gait too: 111 subjects (4.2%)
  This is the primary training cohort for missing-modality experiments.

---

## 11. T1 RAW DOWNLOAD MANIFEST

File: F:\NADI-Q data\PPMI\manifests\ppmi_t1_download_manifest.csv
Rows: 52 (one per eligible triple session)
Columns:
  PATNO, EVENT_ID, mri_scan_date, protocol
  t1_confirmed (from Xing), qc_acceptable (Xing), mriqc_evaluated
  dti_seq_flag, rs_fmri_flag
  download_required (all=True), loni_search_key

Statistics:
  t1_confirmed in Xing:    48 / 52 sessions
  qc_acceptable (Xing):    5 / 52 sessions
  mriqc_evaluated:         23 / 52 sessions
  no Xing record:          4 / 52 sessions (MRI completed but Xing coverage gap)

Download workflow for each row:
  1. Go to LONI IDA Advanced Image Search
  2. Filter: Project=PPMI, Subject ID=PATNO, Visit=EVENT_ID, Image Type=T1, Original
  3. Add to collection
  4. Download collection as ZIP
  LONI search key field in manifest encodes: PPMI,PATNO=<id>,visit=<visit>,T1

QC NOTE: Xing 'qc_acceptable=5' is sparse at gait visit timepoints.
MRIQC provides fuller coverage (23/52) and should be used as primary QC signal.
4 sessions have no Xing record; MRIQC confirms 3 of these are at BL with evaluations.

---

## 12. COHORT DEFINITIONS (Objective Report)

### COHORT A - Strict Same-Visit Triple (MRI+Bio+Gait, MRICMPLT=1)
  Participants: 52
  Sessions: 52 (1 per participant)
  PD: 12  |  HC: 3  |  PRODROMAL: 37
  Modality completeness: 100% (by definition)
  Median gait visits: 1  |  Median MRI visits: 2  |  Median bio visits: 1
  Key biomarkers: eMTBR-TAU243, ABeta40/42, pTau181, NFL, GFAP

### COHORT B - Same-Visit Triple with T1 confirmed (Xing)
  Participants: 48
  Sessions: 48 (1 per participant)
  PD: 10  |  HC: 2  |  PRODROMAL: 36
  T1-confirmed from Xing metadata

### COHORT C - Same-Visit Triple with T1 + MRIQC QC
  Participants: 23
  Sessions: 23 (BL-biased, 1 per participant)
  PD: 6  |  HC: 2  |  PRODROMAL: 15
  Strictest available QC; MRIQC coverage constrains this cohort

### COHORT D - Participant-Level Triple (any visit per modality)
  Participants: 111
  PD: 42  |  HC: 3  |  PRODROMAL: 66
  MRI mean 2.29 visits  |  Bio mean 1.40 visits  |  Gait mean 1.12 visits

### COHORT E - MRI+Bio (gait-missing acceptable), largest longitudinal set
  Participants: 2,665
  PD: 1,003  |  HC: 256  |  PRODROMAL: 1,341  |  SWEDD: 65
  Best for missing-modality training and biomarker trajectory learning

---

## 13. SCIENTIFIC GATE ASSESSMENT

### Criterion: MRI representation learning
  Status: READY
  Evidence: 4,789 MRI subjects (T1=3,694), 8,674 completed scans across 22 visit codes
  Blocker: Raw T1 NIfTI/DICOM files not yet downloaded (~15-30 GB needed)

### Criterion: Biomarker trajectory learning (QVBE)
  Status: READY
  Evidence: 2,966 subjects, mean 3.77 visits, 9,094 assays; CSF+Plasma available
  Key markers: eMTBR-TAU243, ABeta42, pTau181, NFL, GFAP confirmed at triple sessions
  Blocker: None for metadata. Raw CSV is the data; already downloaded.

### Criterion: Gait/kinematics (CTCLM)
  Status: READY_BUT_SAMPLE_LIMITED
  Evidence: 199 subjects, 60 derived kinematic features, 278 sessions
  Limiting factor: Only 5/52 triple-eligible subjects have >=2 gait visits
  Mitigation: UPDRS-III (5,423 subjects, 43 visit codes) is usable as motor proxy
  Blocker: Raw Opal sensor archives not available; derived CSV is usable as-is

### Criterion: Same-subject multimodal fusion (QCDF)
  Status: READY_BUT_SAMPLE_LIMITED
  Evidence: 52 same-visit triple sessions confirmed; 48 with T1 confirmed
  Limiting factor: All 52 subjects have exactly 1 triple-modal session (no longitudinal triple)
  Note: Session-level fusion is possible; temporal fusion requires cross-visit approach

### Criterion: Longitudinal prediction
  Status: READY (for MRI+Bio pair); READY_BUT_SAMPLE_LIMITED (for full triple)
  Evidence: 2,665 MRI+Bio subjects, biospecimen up to 15 visits
  Triple longitudinal: 0 participants with 2+ same-visit triple sessions

### Criterion: Missing-modality experiments
  Status: READY
  Evidence: Cohort E (2,665) provides gait-missing training; Cohort A (52) provides
  complete ground truth for evaluation

### Criterion: PD vs HC classification
  Status: READY
  Evidence: PD=1,003 + HC=256 with MRI+Bio (Cohort E). PD=12 + HC=3 (same-visit triple)
  Note: HC count in triple cohort (3) is too small for standalone HC evaluation

### OVERALL GATE DECISION:
  READY_BUT_SAMPLE_LIMITED

  Basis:
  - MRI and biomarker data are READY (large N, confirmed schemas)
  - Same-visit triple is confirmed at N=52 (48 with T1) - sufficient for evaluation
  - Training will rely on larger paired cohorts (Cohort D=111, Cohort E=2,665)
  - Gait is the bottleneck: 199 subjects total, 52 same-visit with MRI+Bio
  - No same-visit triple longitudinal sessions exist in current data
  - Raw T1 image download is the only remaining blocker for full pipeline execution

---

## 14. NEXT REQUIRED DATA

### IMMEDIATE (Critical Path):
  1. Download raw T1 MRI images for 52 eligible sessions (manifest ready):
     F:\NADI-Q data\PPMI\manifests\ppmi_t1_download_manifest.csv
     LONI IDA -> Advanced Image Search -> 52 PATNO+visit combinations -> T1 Original
     Estimated size: 52 sessions x ~5 MB = ~260 MB (minimal), or full cohort = 15-30 GB

  2. Inspect PPMI_Curated_Data_Cut_Public_20260810.xlsx:
     Contains analytic cohort assignments that supersede COHORT field
     May reveal additional cohort structure for the 52 eligible subjects

### IMPORTANT:
  3. SAA Biospecimen table (if not already in Current_Biospecimen):
     skin_synSAA already present for 13/52 triple subjects
     Check if standalone SAA table adds more coverage

  4. Axivity gait table (if available):
     May add subjects to the gait-missing cohort

### VERIFICATION:
  5. Verify that 4 eligible sessions with no Xing T1 record still have T1 images
     via LONI IDA direct search (PATNO=3004, 42012, 42014 confirmed at BL via MRIQC)

---

## 15. FILE OUTPUTS

| File | Path | Rows |
|------|------|------|
| Candidate manifest | F:\NADI-Q data\PPMI\manifests\ppmi_same_visit_triple_candidates.csv | 278 |
| T1 download manifest | F:\NADI-Q data\PPMI\manifests\ppmi_t1_download_manifest.csv | 52 |
| Audit report | reports\current\PPMI_MODALITY_INTERSECTION_AUDIT.md | - |
| Gate report | reports\current\PPMI_PAIRED_COHORT_GATE_V1.md | - |

Generated: 2026-09-19
No participant-level data exposed. Counts only.
