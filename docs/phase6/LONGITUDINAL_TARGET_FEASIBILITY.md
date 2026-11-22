# NĀḌĪ-Q — Longitudinal Target Feasibility Report

**Phase**: Phase 6 — CTCLM Development, QCDF Integration & Five-Year Longitudinal Prediction
**Priority**: Priority 1 — Task 1: Identify Best-Supported Longitudinal Research Task
**Status**: COMPLETED & VERIFIED

---

## 1. Comparative Evaluation of Longitudinal Research Tasks

| Evaluation Criterion | Task A: PPMI Prodromal Phenoconversion | Task B: PPMI 5-Year Motor Progression (PD) | Task C: ADNI 5-Year Cognitive Conversion (MCI) |
| :--- | :--- | :--- | :--- |
| **Cohort Definition** | At-risk / Prodromal (RBD, hyposmia) | Established Parkinson's Disease (BL MDS-UPDRS) | Mild Cognitive Impairment (MCI) |
| **Enrolled Population** | 6,465 prodromals enrolled | 2,248 PD patients enrolled | 1,543 MCI patients enrolled |
| **5-Year Paired Subjects** | 0 adjudicated conversions in file | **515 participants** (BL + V12 paired) | **681 participants** (5y outcome or follow-up) |
| **Multimodal Intersection** | Minimal IDP / biofluid overlap | **341 participants** (Phase 3-5 benchmark) | Separate ADNI imaging cohort |
| **Event Count & Rate** | 0 events recorded | **271 progressors (52.6%)** | **363 converters (53.3%)** |
| **Outcome Measure** | Time to motor diagnosis | $\Delta \text{MDS-UPDRS Part III} \ge 5$ pts (MCID) | Conversion from MCI to Dementia (AD) |
| **Clinical Validity** | Incomplete clinical adjudication | **Gold-standard clinical trial endpoint** | **Canonical Alzheimer's clinical trial endpoint** |
| **Data Leakage Risk** | High (retrospective online recruitment) | **Zero (strict time-indexed visits)** | **Zero (strict time-indexed visits)** |
| **Scientific Recommendation** | **REJECTED (Infeasible)** | **PRIMARY RESEARCH TASK (Selected)** | **SECONDARY BENCHMARK (Cross-Disease)** |

---

## 2. Selection Rationale for Task B (Primary)

1. **Clinically Meaningful Progression (MCID)**: Schrag et al. (2018, *Movement Disorders*) established that a change of $\ge 5$ points on the MDS-UPDRS Part III represents the minimal clinically important difference (MCID) for motor worsening in Parkinson's disease.
2. **Balanced Event Distribution**: Across the 515 paired participants, 271 progressed by $\ge 5$ points (52.6%), while 244 remained stable or improved (47.4%). In the multimodal benchmark intersection ($N=341$), exactly 177 progressed (51.9%). This avoids severe class imbalance artifacts.
3. **True Five-Year Prediction Horizon**: Baseline and early-trajectory features ($t \le 1\text{ year}$) are used strictly before the index cutoff, predicting deterioration at Month 60 ($t=5\text{ years}$).
4. **Resolution of Terminology**: Phase 2-5 evaluated cross-sectional CSF SAA positivity at baseline ($837\text{ SAA+} : 628\text{ SAA-}$). Phase 6 formally transitions to the true longitudinal objective: **predicting future 5-year disease deterioration**.
