# NĀḌĪ-Q — Five-Year Clinical Outcome Definition & Protocol

**Phase**: Phase 6 — CTCLM Development, QCDF Integration & Five-Year Longitudinal Prediction
**Status**: DEFINED, AUDITED & STATISTICALLY CODIFIED

---

## 1. Formal Target Definition (Primary Endpoint)

Let $Y_i \in \{0, 1\}$ denote the binary 5-year motor progression status for participant $i$:

$$Y_i = \mathbb{I}\left( \text{NP3TOT}_i(V12) - \text{NP3TOT}_i(BL) \ge 5.0 \right)$$

Where:
- $\text{NP3TOT}_i(BL)$ is the MDS-UPDRS Part III Total Motor Score measured at Baseline (Visit `BL`, Month 0).
- $\text{NP3TOT}_i(V12)$ is the MDS-UPDRS Part III Total Motor Score measured at Year 5 (Visit `V12`, Month 60 $\pm$ 6 months).
- Threshold $\theta = +5.0\text{ points}$ is the universally accepted Minimal Clinically Important Difference (MCID) for moderate/severe clinical deterioration.

---

## 2. Secondary Progression Endpoints

1. **Severe Motor Progression ($Y_i^{\text{severe}}$)**: $\Delta \text{NP3TOT} \ge 10.0\text{ points}$ (190 / 515, 36.9% event rate).
2. **Hoehn & Yahr Stage Progression ($Y_i^{\text{HY}}$)**: Transition to a higher H&Y stage ($\text{NHY}(V12) > \text{NHY}(BL)$) (202 / 515, 39.2% event rate).
3. **Continuous Annualized Motor Velocity ($\nu_i$)**:

$$\nu_i = \frac{\text{NP3TOT}_i(V12) - \text{NP3TOT}_i(BL)}{\text{Time}_{\text{years}}} \quad (\text{points/year})$$

Mean observed velocity: $+1.33 \pm 2.90\text{ points/year}$ over 5 years.

---

## 3. Exclusion and Censoring Criteria

1. **Baseline Ineligibility**: Participants missing Baseline MDS-UPDRS Part III examination are excluded.
2. **Censoring Prior to Year 5**: Participants withdrawing or lost to follow-up before Visit `V12` without documented progression are right-censored and excluded from the definitive 5-year binary classification matrix to prevent false negative imputation.
3. **Zero Test-Fold Leakage**: All normalizations, feature encoders, and fusion layers are fitted strictly on training partitions ($K=5$ stratified folds).
