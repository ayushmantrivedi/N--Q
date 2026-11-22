# NADIQ - PPMI FINAL DATA RESOLUTION V1
**Date**: 2026-09-19
**Status**: COMPLETE
**Inputs**: Local PPMI metadata only (Browser automation disabled per user constraints).
**Prepared by**: Antigravity automated pipeline

---

## 1. T1 COHORT RESOLUTION

Based on the 52 eligible triple sessions (MRI + Bio + Gait), the T1 availability is as follows:

- **T1_CONFIRMED (Xing metadata)**: 48 sessions
- **T1_FOUND_INDIRECTLY**: 3 sessions (Resolved via local MRIQC and FS7 metadata proxy for LONI search)
- **UNRESOLVED / VERIFY_MANUALLY**: 1 session

### Resolution for 4 Missing-Xing Sessions:
1. **PATNO 3004 (BL)**: T1_FOUND_INDIRECTLY - Xing row exists with NULL EVENT_ID. MRIQC and FS7 confirm a BL scan was processed. T1 image is highly likely present in LONI.
2. **PATNO 42012 (BL)**: T1_FOUND_INDIRECTLY - Xing BL row has NULL EVENT_ID. Xing V10 T1=Yes confirms T1 protocol. MRIQC and FS7 confirm BL T1 processed.
3. **PATNO 42014 (BL)**: T1_FOUND_INDIRECTLY - Xing BL T1 field is NaN (older protocol). MRIQC and FS7 confirm BL T1 processed.
4. **PATNO 182724 (BL)**: UNRESOLVED / VERIFY_MANUALLY - Xing explicitly records T1=No at BL. T1 is confirmed at V06. Gait session is BL. **Manual LONI search required to verify if BL T1 exists.**

---

## 2. BIOMARKER COVERAGE

For the 52 eligible sessions:
- **CSF Panel (eMTBR-TAU243, ABeta40, ABeta42, pTau181)**: 43/52 subjects (83%)
- **Plasma Panel (NFL, GFAP)**: 34/52 subjects (65%)
- **Plasma Ptau217p**: 25/52 subjects (48%)
- **SAA (CSF AUC/status)**: 17/52 subjects (33%)

*All biomarker data is present in the local Current_Biospecimen_Analysis_Results_18Sep2026.csv.*

---

## 3. GAIT / SENSOR AVAILABILITY

- **Opals (Kinematics)**: Derived 60 scalar features available for 52/52 sessions (94% completeness for key features like speed/cadence).
- **Axivity (Actigraphy)**: Derived 7-day wear summaries available for 50/52 sessions.
- **Raw Sensor Archives**: NOT available in local metadata. Only derived CSVs are present.

---

## 4. CURATED COHORT MAPPING

Based on PPMI_Curated_Data_Cut_Public_20260810.xlsx:
- The curated data cut supersedes the generic COHORT field with nalytic_subgroup, subgroup (e.g., LRRK2 PD, Sporadic PD), and NSD_STAGE.
- Contains curated bio columns (CSFSAA, abeta, tau, ptau, NFL, GFAP).
- The 52 sessions map cleanly to these curated definitions (e.g., PATNO 42012 is LRRK2 NMC, PATNO 42014 is LRRK2 PD).

---

## 5. CTCLM VALIDITY CLASSIFICATION

- **A. CTCLM Temporal Learning**: NOT APPLICABLE with current local data. No raw sensor time series exist locally. Both Opals and Axivity are derived aggregates.
- **B. CTCLM Single-Visit Representation**: APPLICABLE. The 60-dim Opals scalar features (plus Axivity) can be used as a single-visit kinematic encoder.
- **C. Multimodal QCDF Integration**: READY for 51-52 subjects. Can fuse z_mri (from FS7 or T1) + z_bio + z_kin.
- **Longitudinal Claims**: CANNOT BE CLAIMED on the triple cohort because only 5 subjects have >=2 gait visits, and 0 subjects have >=2 same-visit triple sessions. Broader gait longitudinal data (without MRI/Bio constraints) would be needed for temporal kinematics alone.

---

## 6. FINAL 52-SESSION TABLE
|   PATNO | EVENT_ID   | clinical_state   | t1_status                    | mriqc_evaluated   | gait_visit_id_orig   |
|--------:|:-----------|:-----------------|:-----------------------------|:------------------|:---------------------|
|    3004 | BL         | HC               | T1_FOUND_INDIRECTLY          | True              | BL                   |
|   42012 | BL         | PRODROMAL        | T1_FOUND_INDIRECTLY          | True              | BL                   |
|   42014 | BL         | PD               | T1_FOUND_INDIRECTLY          | True              | BL                   |
|  100232 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  100445 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  100712 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  101178 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  101463 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  102245 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  112557 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  114551 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  129515 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  137259 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  138755 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  139471 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  140451 | BL         | HC               | T1_CONFIRMED                 | True              | BL                   |
|  145950 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  153191 | BL         | PD               | T1_CONFIRMED                 | True              | BL                   |
|  165676 | BL         | PD               | T1_CONFIRMED                 | True              | BL                   |
|  168195 | BL         | PD               | T1_CONFIRMED                 | True              | BL                   |
|  175342 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  182724 | BL         | PD               | UNRESOLVED / VERIFY_MANUALLY | True              | BL                   |
|  182848 | BL         | PD               | T1_CONFIRMED                 | True              | BL                   |
|  187759 | BL         | HC               | T1_CONFIRMED                 | False             | BL                   |
|  200627 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  204745 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  211366 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  213013 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  213954 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  217792 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  219562 | BL         | PRODROMAL        | T1_CONFIRMED                 | True              | BL                   |
|  222906 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  230613 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  231122 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  233881 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  239437 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  240559 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  245341 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  245474 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  250052 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  250868 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  252794 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  254231 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  266887 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  292996 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  292998 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  300026 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  301697 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  305079 | BL         | PD               | T1_CONFIRMED                 | False             | BL                   |
|  305772 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  306977 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |
|  307259 | BL         | PRODROMAL        | T1_CONFIRMED                 | False             | BL                   |

---

## 7. EXACT FILES TO DOWNLOAD

See manifests/ppmi_final_download_manifest.csv for the exact required files.
Only the **raw T1 MRI images** for the 51 confirmed sessions (plus manual verification for PATNO 182724) are required. All other metadata (Bio, Gait, FS7) is already locally available.

