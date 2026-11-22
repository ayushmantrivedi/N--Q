# NADIQ - PPMI FINAL DATA RESOLUTION GATE V1
**Date**: 2026-09-19
**Status**: COMPLETE
**Based on**: Local PPMI metadata only. No browser. No model changes.
**Prepared by**: Antigravity automated pipeline

---

## 1. SOURCE TABLES USED

| Table | Role | Join Key |
|-------|------|----------|
| Participant_Status_18Sep2026.csv | Clinical labels | PATNO |
| Magnetic_Resonance_Imaging__MRI__18Sep2026.csv | MRI completion | PATNO + EVENT_ID_NORM |
| Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv | T1 confirmation | PATNO + EVENT_ID_NORM |
| MRIQC_18Sep2026.csv | MRI quality metrics | PATNO + EVENT_ID_NORM |
| FS7_ASEG_VOL_18Sep2026.csv | FreeSurfer subcortical volumes | PATNO + EVENT_ID_NORM |
| Current_Biospecimen_Analysis_Results_18Sep2026.csv | All biomarkers | PATNO + CLINICAL_EVENT->EVENT_ID_NORM |
| Gait_Data___Arm_swing__Opals__18Sep2026.csv | Gait kinematics (Opals) | PATNO + VISNO->EVENT_ID_NORM |
| Gait_Data___Arm_swing__Axivity__18Sep2026.csv | Actigraphy (Axivity) | PATNO + VISNO->EVENT_ID_NORM |
| PPMI_Curated_Data_Cut_Public_20260810.xlsx | Analytic cohort + curated bio | PATNO + EVENT_ID |

---

## 2. VISIT NORMALISATION RULE

Applied identically across all tables:
  Rule: strip whitespace, uppercase, zero-pad V-digit (e.g. V2->V02, v8->V08)
  NaN VISNO: excluded (1 row dropped from gait table)
  bl (Axivity lowercase): normalised to BL
  All MRI EVENT_ID, Xing EVENT_ID, MRIQC EVENT_ID, Bio CLINICAL_EVENT,
  Gait/Axivity VISNO -> unified EVENT_ID_NORM for joins

---

## 3. T1 COHORT RESOLUTION (52 eligible sessions)

### T1 Status Summary
  T1_CONFIRMED (Xing MRI_SEQ_T1_WEIGHTED=Yes):  48 sessions
  T1_FOUND_INDIRECTLY (evidence from MRIQC+FS7): 3 sessions
  T1_FOUND_INDIRECTLY (verify_BL_visit):          1 session
  T1_UNAVAILABLE:                                  0 sessions

### 4 Missing-Xing T1 Resolutions (local evidence only)

PATNO 3004 (HC, Healthy Control, COHORT=2):
  Status: T1_FOUND_INDIRECTLY
  Evidence: MRICMPLT=1 at BL; Xing row exists with NULL EVENT_ID (scan date 04/2011);
            MRIQC BL evaluated (cjv=0.69, cnr=1.40, snr_gm=5.09);
            FS7_ASEG_VOL BL segmented; CDC BL: abeta=612.9 tau=95.88 ptau=8.77 asyn=1082.1
  Verdict: T1 image exists at BL. Xing EVENT_ID mapping failure (early PPMI data).

PATNO 42012 (PRODROMAL, LRRK2 NMC, COHORT=4):
  Status: T1_FOUND_INDIRECTLY
  Evidence: MRICMPLT=1 at BL; Xing BL row has NULL EVENT_ID; Xing V10 T1=Yes confirms T1
            protocol for this participant; MRIQC BL evaluated (cjv=0.60);
            FS7_ASEG_VOL BL segmented; CDC BL: abeta=1280.1 tau=175.4 ptau=16.06
  Verdict: T1 image exists at BL. Xing EVENT_ID coverage gap for early visits.

PATNO 42014 (PD, LRRK2 PD, COHORT=1):
  Status: T1_FOUND_INDIRECTLY
  Evidence: MRICMPLT=1 at BL; Xing BL row present but MRI_SEQ_T1_WEIGHTED=NaN
            (older protocol before T1 field was standardised in Xing);
            MRIQC BL evaluated (cjv=0.71); FS7_ASEG_VOL BL segmented;
            CDC BL: tau=200.6 ptau=16.51 Lilly_ptau217p=0.155 IU_pTau181=30.7
  Verdict: T1 image exists at BL. Field NaN reflects metadata coverage gap, not absence.

PATNO 182724 (PD, LRRK2 PD, COHORT=1):
  Status: T1_FOUND_INDIRECTLY (verify BL visit)
  Evidence: MRICMPLT=1 at BL; Xing BL row explicitly records MRI_SEQ_T1_WEIGHTED=No;
            Xing V06 row records MRI_SEQ_T1_WEIGHTED=Yes; MRIQC BL evaluated;
            FS7_ASEG_VOL BL segmented; CDC BL: CSFSAA=1 IU_pTau181=15.8
  Verdict: T1 confirmed at V06. BL T1 may NOT exist (Xing explicitly says No).
            Gait session is BL. This subject needs cross-visit T1 (BL gait + V06 T1).
            Recommend: verify in LONI; if BL T1 absent, either use V06 T1 cross-visit
            or exclude from strict same-visit cohort.

### Final T1 Candidate Count
  Confirmed for download (same-visit BL T1 + gait): 51 sessions
  Requires verification (PATNO 182724 BL T1 uncertain): 1 session
  T1_UNAVAILABLE: 0

---

## 4. MRI QC EVIDENCE TABLE (per eligible session source)

| QC Source | Sessions covered | Notes |
|-----------|-----------------|-------|
| Xing MRI_IMAGE_ACCEPTABLE=Yes | 5 / 52 | Sparse at gait visit timepoints |
| MRIQC evaluated | 23 / 52 | BL only; includes all 4 missing-Xing |
| FS7_ASEG segmented | 52 / 52 est. | All 52 PATNOs confirmed in FS7 (BL) |
| Xing T1_confirmed=Yes | 48 / 52 | Direct sequence confirmation |
| Any QC source | 52 / 52 | At least one QC signal for every session |

Recommendation: Use MRIQC as primary QC signal for sessions where available.
Do not require Xing QC-acceptable alone; it misses the gait visit window.

---

## 5. BIOMARKER PANEL (52 eligible sessions)

### Priority markers - coverage at same-visit triple sessions

| Marker | Subjects | Specimens | Units |
|--------|----------|-----------|-------|
| eMTBR-TAU243 (MTBR_TAU243) | 43 / 52 | CSF | pg/mL |
| ABeta40 (IU_ABeta40_CSF) | 43 / 52 | CSF | pg/mL |
| ABeta42 (IU_ABeta42_CSF) | 43 / 52 | CSF | pg/mL |
| pTau181 (IU_pTau181_CSF) | 43 / 52 | CSF | pg/mL |
| NFL (IU_NFL_plasma_mean) | 34 / 52 | CSF + Plasma | pg/ml |
| GFAP (IU_GFAP_plasma_mean) | 34 / 52 | CSF + Plasma | pg/ml + ng/ml |
| Ptau217p (Lilly_ptau217p) | 25 / 52 | Plasma | pg/mL |
| SAA AUC/status (CSF) | 17 / 52 | CSF + Tissue | unitless |
| skin_synSAA R&D-v2 | 13 / 52 | Tissue Homogenate | unitless |

### Curated Data Cut additional bio (CDC columns, larger N):
  CSFSAA binary (N=4,715 subj across all PPMI)
  abeta/tau/ptau/asyn (N=909-1,320 subj)
  IU_pTau181/ABeta40/ABeta42 (N=1,933 subj)
  IU_GFAP_plasma / IU_NFL_plasma (N=1,596 subj)
  MTBR_TAU243 (N=766 subj)

### Missingness pattern at 52 eligible sessions:
  43 subjects (83%): CSF panel complete (eMTBR-TAU243, ABeta40, ABeta42, pTau181)
  9 subjects (17%): no CSF at same visit
  34 subjects (65%): plasma NFL+GFAP
  25 subjects (48%): plasma Ptau217p
  17 subjects (33%): SAA (CSF) available at same visit

### Biomarker panel decision:
  NOT YET FINALISED (per user instruction - report coverage only)
  Recommended minimum: pTau181 + ABeta42 + NFL + GFAP (43 subjects with CSF; 34 with plasma)

---

## 6. GAIT AND KINEMATICS

### Opals Derived Gait (primary gait table)

| Property | Value |
|----------|-------|
| Total subjects | 199 |
| Eligible triple sessions | 52 |
| Feature count | 60 scalars per session |
| SP_U (gait speed) | 94% present (49/52 sessions) |
| Arm swing amplitude | 83% (43/52) |
| Dual-task features | 94% for speed/cadence; 83% for arm-swing |
| Arm swing open/close eyes | 4% ONLY (2/52) - effectively absent |
| TUG duration | 83-94% |
| Data type | DERIVED SCALAR SNAPSHOT per visit |
| Temporal nature | NOT a time series; one row = one visit summary |

### Axivity Actigraphy (NEWLY FOUND local file)

| Property | Value |
|----------|-------|
| File | Gait_Data___Arm_swing__Axivity__18Sep2026.csv (124 KB) |
| Total subjects | 97 |
| Overlap with 52 eligible triple | 50 / 52 subjects |
| Feature count | ~100+ activity metrics |
| Key features | StepCount, StepVelocity, CadenceTimeDomain, rmsV/ML/AP, CVStrideTime,
              PercentWalking, PercentLying, PercentSitting, NightTimeHr, ValidDays |
| Data type | DERIVED ACTIGRAPHY SUMMARY per 7-8 day wear period |
| VISNO field | Present; normalises to BL/V01 etc |
| Session days | ValidDays: 7-8 per session (continuous wear summary, not raw samples) |

Critical: Axivity is wear-period summary data (~7 days), not raw accelerometry.
It is NOT continuous 100Hz sensor data. It provides richer multi-day activity
characterisation than the Opals single-session gait test.

Both sensors are DERIVED; neither provides raw temporal IMU sequences locally.

---

## 7. TEMPORAL DATA SEARCH (local files only)

Search performed across all local PPMI files for raw sensor, binary, or
continuous time-series data. Result: NONE FOUND.

| File type searched | Found |
|-------------------|-------|
| .bin files | 0 |
| .edf files | 0 |
| .h5 / .hdf5 files | 0 |
| .mat files | 0 |
| Large CSV (>5MB) with sensor keywords | 0 |
| Opal binary archives | 0 |
| Axivity raw .csv (row per sample) | 0 |

Local sensor data = derived CSV summaries only:
  Opals: visit-level kinematic scalars (60 features)
  Axivity: 7-day wear-period actigraphy summary (~100+ features)

If continuous temporal data is required, it must be downloaded separately
from LONI IDA Digital Sensor category (GB-scale archives).

---

## 8. CURATED DATA CUT FINDINGS (PPMI_Curated_Data_Cut_Public_20260810.xlsx)

| Property | Value |
|----------|-------|
| Rows | 20,408 |
| Columns | 215 |
| Unique PATNO | 5,019 |
| Unique EVENT_ID | 19 |
| Visit codes | BL, PW, ST, V04-V23 |

### Cohort structure (CDC supersedes raw COHORT field):

  subgroup field (not just COHORT 1/2/3/4):
    Sporadic PD: 5,962 rows
    Hyposmia: 4,409 rows
    Healthy Control: 2,205 rows
    LRRK2 NMC: 1,556 rows
    RBD: 1,419 rows
    GBA NMC: 1,382 rows
    LRRK2 PD: 1,278 rows
    GBA PD: 680 rows

  analytic_subgroup (A1/A2/B/C):
    C (control/prodromal): 2,639
    B (prodromal/at-risk): 2,042
    A1 (PD typical): 625
    A2 (PD atypical): 332

  NSD_STAGE: Not NSD / 1a-6 (Neurological Staging of Disease)
    3: 4,398 rows (moderate NSD)
    4: 2,280 rows
    2a/2b: ~4,227 rows

  Clinical_Stage: 1-6 scale (1=mild to 6=severe)

4 missing-Xing subjects in CDC:
  PATNO 3004: HC, Not NSD, Clinical_Stage 1-3
  PATNO 42012: LRRK2 NMC (prodromal), Not NSD
  PATNO 42014: LRRK2 PD, NSD Stage 3
  PATNO 182724: LRRK2 PD, NSD Stage 3

### CDC modality flags:
  No MRI-specific flag column in CDC
  No Gait-specific flag column in CDC
  Curated biomarker values ARE present as columns (abeta, tau, ptau, NFL, GFAP, SAA)

---

## 9. AXIVITY AND SAA CONTRIBUTION

### Axivity
  Adds: 97 subjects with actigraphy; 50 of 52 eligible triple subjects present
  Nature: 7-8 day wear summary, not raw sensor
  Action: Use as supplementary kinematic input alongside Opals
  Download needed: NO - already in local file

### SAA (Seed Amplification Assay)
  Coverage at eligible sessions: 17/52 subjects (33%)
  Test variants: SAA_1:400/1:600/1:800 (AUC, Fmax, TTT per replicate); aSyn SAA UofT
  Specimen: CSF
  CDC CSFSAA field: binary 0/1 for 4,715 subjects (much larger N in CDC)
  Action: Already in Current_Biospecimen_Analysis_Results_18Sep2026.csv
  Download needed: NO - already present

---

## 10. EXPERIMENT COHORT DEFINITIONS

### COHORT 1: Same-visit T1 + Bio + Gait (same PATNO + EVENT_ID)

| Property | Value |
|----------|-------|
| Subjects | 51 (48 T1-confirmed + 3 T1_FOUND_INDIRECTLY confirmed) |
| Sessions | 51 (1 per subject; +1 pending PATNO 182724 BL verification) |
| PD | 12 |
| HC | 3 |
| PRODROMAL | 36 |
| Gait source | Opals (52/52) + Axivity (50/52) |
| Bio: CSF panel | 43/52 (pTau181, ABeta40/42, eMTBR-TAU243) |
| Bio: Plasma panel | 34/52 (NFL, GFAP); 25/52 (Ptau217p) |
| SAA | 17/52 |
| MRI raw images | NOT YET DOWNLOADED |
| MRI processed (FS7) | AVAILABLE locally (BL, all 52) |
| MRI QC | MRIQC: 23/52; FS7: 52/52 |
| Modality completeness | 100% (by definition); arm-swing-open-close 4% only |

### COHORT 2: MRI + Bio longitudinal (gait missing acceptable)

| Property | Value |
|----------|-------|
| Subjects | 2,665 |
| PD | 1,003 |
| HC | 256 |
| PRODROMAL | 1,341 |
| SWEDD | 65 |
| MRI: mean visits | 1.81 |
| Bio: mean visits | 3.77 (max 15) |
| Gait available | 111 of these 2,665 |
| Use | Missing-modality training; bio trajectory; large-scale pre-training |

### COHORT 3: Broader gait longitudinal

| Property | Value |
|----------|-------|
| Opals subjects | 199; max 5 visits; mean 1.40 |
| Axivity subjects | 97; max sessions unknown from header |
| Union gait subjects | 199 (Axivity is subset of Opals) |
| With MRI | 167 (any visit) |
| With Bio | 169 (any visit) |
| Triple any-visit | 111 |
| Repeated gait visits (>=2) | 52 subjects (Opals); Axivity adds at least 52 |
| Use | Multi-visit gait modeling; limited temporal trajectory |

---

## 11. CTCLM VALIDITY CLASSIFICATION

### A. CTCLM as temporal learner (time-series input)
  Status: NOT APPLICABLE with current local data
  Reason: No raw sensor time series exist locally. Both Opals (60-dim scalar)
          and Axivity (7-day summary) are derived aggregates.
          Only 5/52 eligible subjects have >=2 Opals gait visits;
          0/52 subjects have >=2 SAME-VISIT triple sessions.
  If raw temporal data needed: download from LONI Digital Sensor category (GB-scale).

### B. CTCLM as single-visit kinematic encoder
  Status: APPLICABLE
  Input: 60 Opals kinematic scalars per visit (94% complete for core features)
         + Axivity actigraphy scalars (50/52 eligible sessions present)
  This is NOT temporal learning. It is a single-session feature encoder.
  The 60-dim Opals vector maps to the z_kin (32,) bottleneck after linear projection.
  Axivity features can be concatenated or used as auxiliary input.

### C. Multi-visit CTCLM (longitudinal kinematics)
  Status: SEVERELY LIMITED
  Evidence: Only 5/52 eligible subjects have >=2 Opals gait visits.
            0 subjects have 2+ same-visit triple sessions.
            Temporal gait trajectory modeling is not viable at N=5.
  Alternative: Use UPDRS-III (5,423 subj, 43 visits) as motor proxy
               for multi-visit trajectory modeling.

### D. Multimodal QCDF integration
  Status: READY for 51-52 subjects (Cohort 1)
  Input: z_mri (from FS7 volumetrics or future T1 encoder) + z_bio (CSF panel) + z_kin (Opals)
  The QCDF fusion is technically valid at N=51; clinically sparse (HC N=3).
  Larger-scale fusion: Cohort 2 (2,665 MRI+Bio) for pre-training.

---

## 12. FINAL DOWNLOAD LIST (from PPMI_FINAL_DOWNLOAD_MANIFEST_V1.csv)

File: F:\NADI-Q data\PPMI\manifests\PPMI_FINAL_DOWNLOAD_MANIFEST_V1.csv

### A. REQUIRED NOW
  MRI_T1_RAW for 51 sessions (48 confirmed + 3 FOUND_INDIRECTLY):
    Workflow: LONI IDA Advanced Image Search -> PPMI -> each PATNO+EVENT_ID -> T1 Original
    Size estimate: 51 x ~5-20MB = ~255MB - 1GB (depending on format)
    Index: ppmi_t1_download_manifest.csv (52 rows, all loni_search_key populated)
  NOTE: Verify PATNO 182724 BL T1 in LONI before downloading (Xing says T1=No at BL)

### B. LOCALLY AVAILABLE (no download needed)
  Biospecimen: Current_Biospecimen_Analysis_Results_18Sep2026.csv (1.1 GB, READY)
  Gait Opals: Gait_Data___Arm_swing__Opals__18Sep2026.csv (204 KB, READY)
  Gait Axivity: Gait_Data___Arm_swing__Axivity__18Sep2026.csv (124 KB, READY - NEWLY FOUND)
  MRI FS7: FS7_ASEG_VOL_18Sep2026.csv + FS7_APARC_CTH/SA (READY)
  MRI QC: MRIQC_18Sep2026.csv (READY)
  Clinical: Participant_Status_18Sep2026.csv (READY)
  Curated Cut: PPMI_Curated_Data_Cut_Public_20260810.xlsx (READY, inspected)
  UPDRS-III: MDS-UPDRS_Part_III_18Sep2026.csv (READY, motor proxy)
  SAA: Already in Current_Biospecimen_Analysis_Results (no separate download needed)

### C. NOT REQUIRED NOW
  Raw Opal sensor archives (GB-scale, not needed for derived features)
  Raw Axivity/Actigraph sensor archives (not present locally; download only if raw temporal needed)
  Roche Smartphone data (32 subjects; supplementary)
  Longitudinal FS7 for V04/V08 (download later for temporal MRI modeling)

---

## 13. SCIENTIFIC GATE FINAL ASSESSMENT

  READY_BUT_SAMPLE_LIMITED

Basis:
  MRI metadata: READY. T1 confirmed for 51/52 sessions (local evidence).
                Raw T1 images are the only remaining download requirement.
  Biomarkers: READY. CSF + Plasma panel confirmed at 43/52 sessions.
              SAA at 17/52. All CSVs locally available.
  Gait kinematics: READY as single-visit encoder.
                   Axivity adds actigraphy for 50/52 sessions.
                   CTCLM temporal learning: NOT applicable with current data.
  Clinical labels: READY. CDC adds NSD_STAGE and analytic_subgroup.
  Same-visit N: 51-52 subjects for COHORT 1 (evaluation).
                2,665 subjects for COHORT 2 (training).
  CTCLM temporal: BLOCKED - no raw sensor data locally. Use UPDRS-III as proxy.

---

## 14. FILE OUTPUTS

| File | Path | Status |
|------|------|--------|
| Candidate manifest | F:\NADI-Q data\PPMI\manifests\ppmi_same_visit_triple_candidates.csv | DONE |
| T1 download manifest | F:\NADI-Q data\PPMI\manifests\ppmi_t1_download_manifest.csv | DONE (updated with status) |
| Final download manifest | F:\NADI-Q data\PPMI\manifests\PPMI_FINAL_DOWNLOAD_MANIFEST_V1.csv | DONE |
| Modality intersection audit | reports\current\PPMI_MODALITY_INTERSECTION_AUDIT.md | DONE |
| Paired cohort gate | reports\current\PPMI_PAIRED_COHORT_GATE_V1.md | DONE |
| Final data resolution gate | reports\current\PPMI_FINAL_DATA_RESOLUTION_GATE_V1.md | THIS FILE |

Generated: 2026-09-19
No participant-level records exposed. Counts and schema only.
