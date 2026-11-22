# NĀḌĪ-Q — Temporal Leakage and Partition Integrity Audit

**Phase**: Phase 6 — CTCLM Development, QCDF Integration & Five-Year Longitudinal Prediction
**Audit Status**: 100% PASSED (Zero Data Leakage Detected)

---

## 1. Systematic Leakage Verification Matrix

| Audit Check | Description | Rule Enforced | Verification Status |
| :--- | :--- | :--- | :---: |
| **Observation Window Cutoff** | All predictor features measured at t <= Month 12 (V04) | Deterministic Gate | **PASSED** |
| **Target Separation** | NP3TOT_V12 measured at Month 60 (t=5 years) strictly excluded from input X | Deterministic Gate | **PASSED** |
| **Intermediate Visit Filtering** | Visits V06 (Yr 2), V08 (Yr 3), V10 (Yr 4) strictly excluded from feature space | Deterministic Gate | **PASSED** |
| **Patient Split Disjointness** | Train and test participant sets across 5 folds share zero PATNO overlap | Deterministic Gate | **PASSED** |
| **Normalization Integrity** | Imputers and StandardScaler fitted strictly on training partition folds | Deterministic Gate | **PASSED** |
| **Outcome Derivation Check** | No administrative or study conclusion columns included in X | Deterministic Gate | **PASSED** |
| **Censoring Bias Prevention** | Censored subjects without 5-year follow-up excluded from binary evaluation | Deterministic Gate | **PASSED** |

---

## 2. Stratified Partition Integrity

- **Grouping**: Strictly clustered by `PATNO`. No participant appears in both training and test partitions of any fold.
- **Preprocessing Isolation**: Imputers, StandardScalers, and dimension reduction projections are fitted solely on $X_{\text{train}}$ and transformed on $X_{\text{test}}$.
- **Target Isolation**: The 5-year progression outcome ($Y$) and Year 5 score (`NP3TOT_V12`) are strictly quarantined and never exposed to the feature pipeline.
