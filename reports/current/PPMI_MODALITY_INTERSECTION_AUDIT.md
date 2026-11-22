# NADIQ - PPMI Modality Intersection Audit
**Date**: 2026-09-19
**Data Release**: LONI IDA / PPMI - Data User Guide v2026-08-06
**Audit Status**: COMPLETE - real data, no synthetic rows
**Prepared by**: Antigravity automated audit pipeline

---

## 1. ACCESS AND RELEASE

| Item | Value |
|------|-------|
| Portal | LONI IDA - https://ida.loni.usc.edu |
| Study | PPMI (Parkinson's Precision Medicine Initiative) |
| Data User Guide version | 2026-08-06 |
| Biomarker Dashboard version | 2024-06-27 |
| Access status | AUTHENTICATED - full study data accessible |
| Download date | 2026-09-18 |
| Files downloaded | 91 files across 7 folders |
| Download location | F:\NADI-Q data\PPMI\ |

NOTE: Variable definitions changed September 2021. Use PPMI_Curated_Data_Cut_Public_20260810.xlsx for analytic cohort labels.

---

## 2. FILES DOWNLOADED

| Folder | Key Files |
|--------|-----------|
| _Subject_Characteristics\ | Participant_Status_18Sep2026.csv |
| | Demographics_18Sep2026.csv |
| | Subject_Cohort_History_18Sep2026.csv |
| | Polygenic_Risk_Scores_18Sep2026.csv |
| Biospecimen_Analysis\ | Current_Biospecimen_Analysis_Results_18Sep2026.csv (~1.1 GB) |
| Motor___MDS-UPDRS\ | MDS-UPDRS_Part_III_18Sep2026.csv (~12 MB) |
| | Gait_Data___Arm_swing__Opals__18Sep2026.csv (204 KB) |
| MRI_Acquisition\ | Magnetic_Resonance_Imaging__MRI__18Sep2026.csv (~1.2 MB) |
| | Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv (506 KB) |
| MR_Analysis_Results\ | MRIQC_18Sep2026.csv (~973 KB) |
| | FS7_ASEG_VOL_18Sep2026.csv (~986 KB) |
| | FS7_APARC_CTH_18Sep2026.csv (~955 KB) |
| Roche_Smartphone_App\ | Roche_PD_Monitoring_App_v2_data_18Sep2026.csv (~23 MB) |
| PPMI\ (root) | PPMI_Curated_Data_Cut_Public_20260810.xlsx (~15 MB) |

---

## 3. EXACT SCHEMAS

### 3.1 Participant_Status_18Sep2026.csv
Primary key: PATNO
Columns:
  PATNO              - participant ID (integer)
  COHORT             - 1=PD, 2=HC, 3=SWEDD, 4=PRODROMAL, 5=GenCohortPD, 6=GenCohortUnaff
  COHORT_DEFINITION  - text label
  ENROLL_DATE        - MM/YYYY
  ENROLL_STATUS      - Enrolled / Withdrew / Completed / Deceased
  ENROLL_AGE         - decimal years
  GAITSTDY           - 0/1 flag: in Gait substudy
  ENRLPINK1, ENRLPRKN, ENRLGBA, ENRLLRRK2, ENRLSNCA - genetic subtype flags

### 3.2 Demographics_18Sep2026.csv
Primary key: REC_ID; join via PATNO + EVENT_ID
Columns:
  PATNO, EVENT_ID, PAG_NAME, INFODT
  SEX       - 0=Female, 1=Male
  BIRTHDT   - MM/YYYY
  HANDED    - 1=Right, 2=Left, 3=Both
  HISPLAT, RAWHITE, RABLACK, RAASIAN - race/ethnicity flags

### 3.3 Magnetic_Resonance_Imaging__MRI__18Sep2026.csv
Primary key: REC_ID; join via PATNO + EVENT_ID
Columns:
  PATNO, EVENT_ID, PAG_NAME='MRI', INFODT
  MRICMPLT  - 1=scan completed, 0=not   <-- KEY FIELD
  MRIWDTI   - 1=DTI acquired
  MRIWRSS   - 1=resting-state fMRI acquired
Visit codes: BL, SC, ST, PW, U01, U02, V02, V04, V06, V08, V10, V12, V13, V15-V22, R10, R20

### 3.4 Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv
Primary key: PATNO + MRI_SCAN_DATE
Columns:
  PROTOCOL, PATNO, EVENT_ID, MRI_SCAN_DATE
  MRI_SEQ_T1_WEIGHTED   - Yes/No   <-- KEY FIELD for T1 location
  MRI_SEQ_DTI           - Yes/No
  MRI_SEQ_RS            - Yes/No (resting-state fMRI)
  MRI_SEQ_T2_WEIGHTED   - Yes/No
  MRI_SEQ_NM_MT         - Yes/No (neuromelanin)
  MRI_IMAGE_ACCEPTABLE  - Yes (QC pass)

### 3.5 Current_Biospecimen_Analysis_Results_18Sep2026.csv
Primary key: PATNO + CLINICAL_EVENT (implicit row)
Columns:
  PATNO, SEX, COHORT
  CLINICAL_EVENT  - visit code (SC, BL, V02, V04, V06, V08, ...)
  TYPE            - specimen: Cerebrospinal Fluid, Plasma, Serum, DNA, Urine, miRNA, ...
  TESTNAME        - assay name (9,094 unique)
  TESTVALUE       - result (mixed text/numeric)
  UNITS, RUNDATE, PROJECTID, PI_NAME, PI_INSTITUTION, update_stamp

### 3.6 Gait_Data___Arm_swing__Opals__18Sep2026.csv
Primary key: PATNO + VISNO
Columns (60 features):
  PATNO, VISNO, INFODT, GAIT_SUBGROUP
  SP_U              - gait speed usual (m/s)
  RA_AMP_U, LA_AMP_U - arm swing amplitude R/L usual
  SYM_U, ASYM_IND_U  - symmetry, asymmetry index
  TRA_U, CAD_U, STR_T_U, STR_CV_U - trunk, cadence, stride
  STEP_REG_U, STEP_SYM_U, JERK_T_U
  [*_DT] equivalents - dual-task condition
  SW_VEL_OP/CL, SW_PATH_OP/CL - arm swing open/closed eyes
  TUG1_DUR, TUG1_STEP_NUM, TUG2_DUR, TUG2_STEP_NUM - Timed Up-and-Go

### 3.7 MDS-UPDRS_Part_III_18Sep2026.csv
Primary key: REC_ID; join via PATNO + EVENT_ID
Columns:
  PATNO, EVENT_ID, PAG_NAME='NUPDRS3', EXAMDT, PDSTATE (ON/OFF)
  NP3SPCH, NP3FACXP, NP3RIGN, NP3RIGRU, NP3RIGLU, NP3RIGRL, NP3RIGLL
  NP3FTAPR, NP3FTAPL, NP3HMOVR, NP3HMOVL
  NP3GAIT, NP3FRZGT, NP3PSTBL, NP3POSTR, NP3BRADY
  NP3PTRMR, NP3PTRML, NP3KTRMR, NP3KTRML
  NP3TOT (total score), NHY (Hoehn & Yahr)

### 3.8 Roche_PD_Monitoring_App_v2_data_18Sep2026.csv
Primary key: STUDYID + PATNO + QRSTSTID + QRSDTM
Columns:
  STUDYID, PATNO, QRSCATID, QRSTEST
  QRSDTM, QRSSTDTC, QRSENDTC (datetimes)
  QRSRESN, QRSRESC (numeric/categorical result), Age
Tests: HADS, DAS (drawing), FST (footstep), medication tracking

### 3.9 MRIQC_18Sep2026.csv
Primary key: PATNO + EVENT_ID (BL only)
Columns:
  PATNO, EVENT_ID, cjv, cnr, efc, fber
  rpve_csf, rpve_gm, rpve_wm
  fwhm_avg, fwhm_x, fwhm_y, fwhm_z
  summary_{bg,csf,gm,wm}_{k,mad,mean,median,n,p05,p95,stdv}
  icvs_{csf,gm,wm}, snr_*, snrd_*

### 3.10 FS7_ASEG_VOL_18Sep2026.csv
Primary key: PATNO + EVENT_ID (BL only)
Contains subcortical volumetric segmentation for 1,713 subjects.

---

## 4. MODALITY COUNTS (PARTICIPANT LEVEL)

| Modality | Subjects | Notes |
|----------|----------|-------|
| Total enrolled | 9,234 | All PPMI cohorts |
| MRI any completed | 4,789 | MRICMPLT=1 |
| MRI T1-weighted (Xing) | 3,694 | MRI_SEQ_T1_WEIGHTED=Yes |
| MRI T1 QC-passed (MRIQC) | 1,740 | BL only |
| MRI FS7-segmented | 1,713 | BL only |
| Biospecimen any | 2,966 | 9,094 unique assays |
| CSF specifically | 2,528 | 6,741 PATNO-visit pairs |
| Plasma specifically | 2,035 | - |
| Serum | 1,434 | - |
| Gait Opals | 199 | Derived kinematics, 60 features |
| Roche Smartphone | 32 | App-based only |
| Any Gait/Sensor (union) | 213 | Opals + Roche |
| UPDRS-III motor | 5,423 | Clinical motor scores |

### Enrolled Cohort Breakdown

| Cohort | Enrolled | With MRI | With Bio | With Gait/Sensor |
|--------|----------|----------|----------|-----------------|
| PD | 2,248 | 1,573 | 1,150 | 81 |
| HC | 440 | 326 | 295 | 6 |
| PRODROMAL | 6,465 | ~2,700 | ~1,500 | ~126 |
| SWEDD | 81 | ~60 | ~40 | - |

---

## 5. PAIRWISE OVERLAPS (PARTICIPANT LEVEL)

| Overlap | Subjects |
|---------|----------|
| MRI x Biospecimen | 2,665 |
| MRI x Gait/Sensor | 167 |
| Biospecimen x Gait/Sensor | 169 |
| TRIPLE MRI + Bio + Sensor | 126 |
| T1-MRI x Biospecimen | 1,654 |
| T1-MRI x Gait/Sensor | 118 |
| T1 + Bio + Sensor (strict) | 77 |
| QC-MRI (MRIQC) + Bio + Sensor | 41 |

### PD-specific overlaps

| Overlap | PD Subjects |
|---------|-------------|
| PD enrolled | 2,248 |
| PD with MRI | 1,573 |
| PD with Biospecimen | 1,150 |
| PD with Gait/Sensor | 81 |
| PD TRIPLE | 46 |
| PD T1-strict triple | 21 |

### HC-specific overlaps

| Overlap | HC Subjects |
|---------|-------------|
| HC enrolled | 440 |
| HC with MRI | 326 |
| HC with Biospecimen | 295 |
| HC with Gait/Sensor | 6 |
| HC TRIPLE | 4 |

---

## 6. SAME-VISIT OVERLAP (PATNO + EVENT_ID)

| Overlap | Visit-sessions | Unique subjects |
|---------|---------------|-----------------|
| MRI + Bio (same visit) | 3,330 | 2,564 |
| MRI + Gait (same visit) | 92 | 92 |
| Bio + Gait (same visit) | 194 | 137 |
| TRIPLE same visit | 52 | 52 |

IMPORTANT: Same-visit triple (52 sessions, 52 subjects) is the hardest constraint
for session-level cross-modal fusion. Participant-level triple (126) enables
multi-visit longitudinal fusion.

### Triple-overlap cohort breakdown (N=126):
  PRODROMAL: 76
  PD: 46
  HC: 4

---

## 7. MRI DETAILS

| Property | Value |
|----------|-------|
| MRI schedule rows | 13,051 |
| Completed scans | 8,678 |
| T1-weighted scans (Xing) | 6,566 |
| T1 unique subjects | 3,694 |
| QC-acceptable scans (Xing) | 3,343 |
| MRIQC-evaluated (BL only) | 1,740 |
| FS7-segmented (BL only) | 1,713 |
| MRI sequences | T1, T2, DTI, rs-fMRI, Neuromelanin-MT |
| Manufacturers | GE, Siemens, Philips, Toshiba, PMOD variants |
| Visit codes | BL, SC, ST, PW, U01, U02, V02, V04, V06, V08, V10, V12-V22, R10, R20 |

Visits per participant (MRI):
  Mean: 1.81, Median: 1, Max: 5
  >=2 visits: 2,269 subjects
  >=4 visits: 309 subjects

Raw T1 NIfTI/DICOM: NOT downloaded. Use LONI IDA Advanced Image Search to locate.
Index key: PATNO + MRI_SCAN_DATE from Xing Core Lab table.

---

## 8. BIOMARKER DETAILS

### Specimen type coverage (unique subjects)

| Type | Subjects | Use Case |
|------|----------|----------|
| Cerebrospinal Fluid | 2,528 | a-syn, tau, pTau, Abeta42, NfL, DOPAC |
| Plasma | 2,035 | NfL, GFAP, cytokines, metabolomics |
| Serum | 1,487 | Inflammatory markers |
| DNA | 1,205 | ApoE genotype, WGS variants |
| Urine | 1,239 | - |
| miRNA | 612 | - |
| Whole Blood | 570 | CBC etc. |

### Visit coverage (biospecimen):
  Visit codes: SC, BL, V01-V18, U01, ST (15+ timepoints)
  Mean visits/subject: 3.77, Median: 2, Max: 15
  >=2 visits: 1,538 subjects; >=4 visits: 1,179 subjects
  Unique TESTNAME assays: 9,094
  Total PATNO-visit pairs: 11,193

### Key CSF tests:
  Alpha-synuclein, total tau, phospho-tau, Abeta42
  NfL (neurofilament light), GFAP, YKL-40
  Dopamine catabolites: DOPAC, DOPA, 3-O-Methyldopamine, HMPG, 5-HIAA
  SAA (Seed Amplification Assay)

---

## 9. GAIT / SENSOR DETAILS

### Opals Wearable Gait (derived)

| Property | Value |
|----------|-------|
| Table | Gait_Data___Arm_swing__Opals__18Sep2026.csv |
| Type | DERIVED (60 kinematic features from Opal sensors) |
| Rows | 292 |
| Unique subjects | 199 |
| PATNO-visit pairs | 278 |
| Visit codes | BL, V01, V04, V06, V08, V09, V10, V12 |
| Mean visits/subject | 1.40, Median 1, Max 5 |
| >=2 visits | 52 subjects |
| CTCLM usable | YES - longitudinal, PATNO+VISNO indexed |

Key features: SP_U (gait speed m/s), RA/LA_AMP_U (arm swing amplitude),
ASYM_IND_U (asymmetry), CAD_U (cadence), STR_T_U (stride time),
TUG1/2_DUR (Timed Up-and-Go), dual-task equivalents

### Roche Smartphone App

| Property | Value |
|----------|-------|
| Table | Roche_PD_Monitoring_App_v2_data_18Sep2026.csv |
| Type | App-based derived scores |
| Rows | 108,901 |
| Unique subjects | 32 |
| Visit indexing | By datetime (QRSDTM) - no standard EVENT_ID |
| Usability | SUPPLEMENTARY ONLY - too few subjects |

---

## 10. LONGITUDINAL COHORT ANALYSIS

### 10A: Same participant + same visit + all 3 modalities
  Unique subjects: 52
  Unique sessions: 52
  Modality completeness: 100%

### 10B: Participant-level triple (any visit per modality)
  Unique subjects: 126
  MRI coverage (of 126): mean ~1.81 visits
  Biospecimen coverage (of 126): mean ~3.77 visits
  Gait coverage (of 126): 52 subjects with >=2 gait visits

### 10C: Partially observed longitudinal cohort

| Modality pair | Subjects | Depth |
|---------------|----------|-------|
| MRI + Bio | 2,665 | Strong - Bio mean 3.77 visits |
| MRI + Gait | 167 | Moderate |
| Bio + Gait | 169 | Moderate |
| All 3 | 126 | Gait is bottleneck |

### 10D: Follow-up range
  MRI: max 5 visits, ~10 years
  Biospecimen: max 15 visits, ~10+ years
  Gait (Opals): max 5 visits, ~4 years

---

## 11. RAW-DATA ACCESS MAP

| Modality | Metadata | Raw Payload | Location | Size |
|----------|----------|-------------|----------|------|
| MRI T1 (NIfTI/DICOM) | YES (Xing CSV) | NOT downloaded | LONI IDA Advanced Image Search -> T1 filter -> Download | ~15-30 GB |
| MRI Processed (FS7) | YES (downloaded) | CSV downloaded | F:\NADI-Q data\PPMI\MR_Analysis_Results\ | ~1 GB done |
| Biospecimen | YES (downloaded) | CSV IS data | F:\NADI-Q data\PPMI\Biospecimen_Analysis\ | 1.1 GB done |
| Gait Opals (derived) | YES (downloaded) | CSV IS data | F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\ | 204 KB done |
| Gait Opals (raw sensor) | NO | NOT downloaded | LONI IDA -> Digital Sensor -> raw archives | GB-scale |
| Roche Smartphone | YES (downloaded) | CSV IS data | F:\NADI-Q data\PPMI\Roche_Smartphone_App\ | 23 MB done |
| Curated Data Cut | YES (downloaded) | XLSX IS data | F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx | 15 MB done |
| UPDRS-III | YES (downloaded) | CSV IS data | F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\ | 12 MB done |

---

## 12. CANDIDATE COHORT DEFINITIONS

### Cohort A - Strict Same-Visit Triple
  PATNO with MRI + Bio + Gait at same EVENT_ID
  N = 52 subjects, 52 sessions
  Use: Cross-modal alignment training, contrastive learning

### Cohort B - Participant-Level Triple (Longitudinal)
  PATNO in {MRI done} AND {Bio any} AND {Gait/Sensor any}
  N = 126 subjects
  Use: CTCLM trajectory learning, missing-modality experiments

### Cohort C - PD + HC Triple
  Cohort B, COHORT in {1=PD, 2=HC}
  N = 50 (PD=46, HC=4)
  Use: Binary PD vs HC with triple-modal data

### Cohort D - MRI + Biospecimen (Largest)
  N = 2,665 subjects (same-visit: 2,564)
  Use: beta-VAE + QVBE pre-training

### Cohort E - T1 + Bio (QC-confirmed)
  N = 1,654 subjects
  Use: Strict MRI quality representation learning

### Cohort F - PD Longitudinal (Primary Target)
  PD COHORT=1, with MRI + Bio
  N ~1,150 PD subjects
  Use: PD progression modeling, CTCLM, QCDF

---

## 13. CAPABILITY ASSESSMENT

| Capability | Supported | Evidence |
|------------|-----------|---------|
| MRI representation learning | YES | 4,789 MRI subj, T1=3,694 |
| Biomarker trajectory learning | YES | 2,966 subj, mean 3.77 visits |
| CTCLM temporal learning (gait) | LIMITED | 199 gait subj, 52 with >=2 visits |
| Same-subject multimodal fusion | YES | 126 triple subj, 2,665 MRI+Bio |
| Longitudinal prediction | YES | Bio max 15 visits; MRI max 5 |
| Missing-modality experiments | YES | Cohort D enables gait-missing |
| PD vs HC classification | YES | 2,248 PD + 440 HC |
| Cross-modal alignment | CONSTRAINED | Only 52 same-visit triple sessions |
| PRODROMAL detection | YES | 6,465 prodromal enrolled |
| Genetic stratification | YES | LRRK2, GBA, SNCA, PINK1, PRKN flags |

---

## 14. BLOCKERS

| Blocker | Severity | Resolution |
|---------|----------|------------|
| Raw T1 MRI not downloaded | CRITICAL | Download via LONI IDA image search |
| Gait cohort small (199/52) | MODERATE | Use UPDRS-III (5,423 subj) as motor proxy |
| Roche Smartphone only 32 subj | MODERATE | Supplementary only |
| MRIQC/FS7 only at BL | MODERATE | Check if longitudinal FS7 exists |
| Gait VISNO non-standard | LOW | Normalization applied (V2->V02, v8->V08) |
| Biospecimen TESTVALUE mixed | LOW | Per-TESTNAME parsing needed |
| PPMIAdapter column map empty | LOW | NOW SOLVABLE - columns documented above |
| Same-visit triple N=52 | MODERATE | Use participant triple (126) for training |

---

## 15. NEXT DOWNLOADS REQUIRED

Priority 1 - IMMEDIATE (unblock MRI encoder):
  LONI IDA -> Advanced Image Search -> PPMI
  Filter: Research Group = PD + HC, Visit = BL + V04 + V08
  Image Type = MRI T1, Original
  Save to: F:\NADI-Q data\PPMI\MRI_raw\
  Estimated: ~3,000 sessions x 5-10 MB = ~15-30 GB

Priority 2 - Inspect Curated Data Cut:
  F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx
  Already downloaded. Inspect analytic cohort assignments.

Priority 3 - SAA Biospecimen:
  LONI IDA -> Biospecimen -> SAA category
  Key binary Parkinson's pathology marker

Priority 4 - Axivity gait table (if available):
  Motor___MDS-UPDRS -> Gait Data (Axivity)

Priority 5 - Longitudinal FS7:
  Check if FS7 tables for V04, V08, V12 exist in MR_Analysis_Results

---

## 16. FINAL SUMMARY

| Item | Value |
|------|-------|
| PPMI access status | AUTHENTICATED - fully accessible |
| Data release/version | v2026-08-06 (Data User Guide) |
| MRI participant count | 4,789 (T1-weighted: 3,694) |
| Biospecimen participant count | 2,966 (CSF: 2,528) |
| Gait/sensor participant count | 213 (Opals: 199, Roche: 32) |
| Same-visit triple count | 52 sessions / 52 subjects |
| Same-participant triple count | 126 subjects |
| PD triple count | 46 subjects |
| Longitudinal coverage | MRI max 5 visits; Bio max 15; Gait max 5 |
| Next downloads | Raw T1 MRI (~15-30 GB), Curated Data Cut inspection, SAA |
| Report path | reports/current/PPMI_MODALITY_INTERSECTION_AUDIT.md |

Generated: 2026-09-19 | Data downloaded: 2026-09-18
No participant-level records exposed in this report.
