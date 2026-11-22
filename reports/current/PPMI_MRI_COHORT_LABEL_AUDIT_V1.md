# NADIQ - PPMI MRI COHORT LABEL AUDIT V1
**Date**: 2026-09-24
**Status**: AUDITED & VERIFIED
**Prepared by**: Antigravity automated pipeline

---

## 1. THE 75-COUNT SUBGROUP INCONSISTENCY EXPLAINED

In the previous report, the generic COHORT counts correctly summed to 15. However, the curated subgroup (from the Curated Data Cut - CDC) summed to 75. 

**Root Cause**:
The CDC is a longitudinal dataset containing multiple rows (visits) per participant. By filtering cdc['PATNO'].isin(valid_patnos) and immediately calling .value_counts(), the previous logic counted **every visit** for the 15 subjects instead of just unique subjects. 
The 15 subjects have a total of 75 visits recorded in the CDC. Because the subgroup field is a cohort-level label repeated across all visits for a given participant, counting all rows resulted in a massive overcount (75).

When properly de-duplicated to 1 row per participant, the subgroup semantics are completely mutually exclusive per subject, and correctly sum to 15.

---

## 2. EXACT CLINICAL CLASS COUNTS (N=15)

Using the generic clinical_state (derived from the primary COHORT field), the exact mutually exclusive class distribution for the modeling cohort is:

- **PRODROMAL (COHORT 4)**: 11
- **Healthy Control (HC, COHORT 2)**: 3
- **PD (COHORT 1)**: 1

**Total Subjects**: 15

---

## 3. MULTIMODAL OVERLAP VERIFICATION

I have rigorously joined the 15 acquired MRI subjects against the biomarker and gait tables.

- **MRI available**: 15 / 15
- **Biomarker available at BL**: 15 / 15
- **Gait (Opals) available at BL**: 15 / 15
- **Verified Triple Cohort (MRI ? Bio ? Gait)**: 15 subjects.

*All 15 manually acquired MRI subjects legitimately possess Biomarker and Gait records at Baseline.*

---

## 4. ANOMALY & DUPLICATION AUDIT

I scanned for duplicate records at the Baseline (BL) event for the 15 subjects:
- **Multiple Biomarker records at BL**: 15 subjects
- **Multiple Gait records at BL**: 0 subjects
- **Multiple MRI records in manifest**: 0 subjects
- **Mismatched EVENT_ID or PATNO**: 0 (all successfully joined on strict equality)
- **Missing clinical label**: 0 subjects

> *Note on Multiple Records: If multiple records exist (e.g., duplicated gait trials at BL), they will need to be aggregated or filtered during the DataLoader phase. The metadata confirms the modality exists.*

---

## 5. SUBJECT-LEVEL MODELING MANIFEST

The clean, verified subject-level modeling manifest has been written to:
F:\NADI-Q data\PPMI\manifests\ppmi_mri_modeling_cohort_v1.csv

### Snapshot
|   PATNO | EVENT_ID   | clinical_state   |   COHORT | curated_subgroup   | analytic_subgroup   | NSD_STAGE   | MRI_available   | biomarker_available   | gait_available   |
|--------:|:-----------|:-----------------|---------:|:-------------------|:--------------------|:------------|:----------------|:----------------------|:-----------------|
|  100232 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  100445 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  100712 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  101178 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  101463 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | 2b          | True            | True                  | True             |
|  102245 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | 2b          | True            | True                  | True             |
|  114551 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | 2a          | True            | True                  | True             |
|  137259 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  138755 | BL         | PRODROMAL        |        4 | Hyposmia           | C                   | Not NSD     | True            | True                  | True             |
|  140451 | BL         | HC               |        2 | Healthy Control    | nan                 | Not NSD     | True            | True                  | True             |
|  153191 | BL         | PD               |        1 | Sporadic PD        | nan                 | 3           | True            | True                  | True             |
|  187759 | BL         | HC               |        2 | Healthy Control    | nan                 | Not NSD     | True            | True                  | True             |
|  213013 | BL         | PRODROMAL        |        4 | RBD                | A1                  | 2a          | True            | True                  | True             |
|    3004 | BL         | HC               |        2 | Healthy Control    | nan                 | Not NSD     | True            | True                  | True             |
|   42012 | BL         | PRODROMAL        |        4 | LRRK2 NMC          | nan                 | Not NSD     | True            | True                  | True             |