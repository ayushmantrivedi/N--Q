# TADPOLE Classical Baseline Experiment

**Objective**: Establish a scientifically clean classical baseline for MCI baseline snapshot → future DEMENTIA transition risk.

## 1. Methodology
- **Leakage checks pass**: True
- **Feature set**: AGE_AT_SNAPSHOT, PTGENDER, PTEDUCAT, APOE4, MMSE, CDRSB, ADAS11, ADAS13, RAVLT_immediate, Hippocampus, WholeBrain, Entorhinal, MidTemp
- **Split methodology**: Patient-level disjoint train/val/test splits (15% val, 15% test)
- **Random seed**: 42
- **Preprocessing**: Median Imputation → Variance Filter → Z-Score Normalization (fitted on train partition ONLY)
- **Models Run**: LogisticRegression, HistGradientBoosting, MLP

## 2. Results by Horizon
### Horizon 12 Months

- **Eligible Patients (MCI)**: 67
- **Positive Cases**: 7
- **Negative Cases**: 44
- **Censored/Unknown Cases**: 16
- **Train/Val/Test Counts**: 37 / 7 / 7

#### LogisticRegression
- **AUROC**: 1.0
- **AUPRC**: 1.0
- **F1**: 1.0
- **Sensitivity (Recall)**: 1.0
- **Specificity**: 1.0
- **Brier Score**: 0.051308985443717696
- **95% Bootstrap CI (AUROC)**: [1.000, 1.000]

#### HistGradientBoosting
- **AUROC**: 0.5
- **AUPRC**: 0.14285714285714285
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.1228216633622039
- **95% Bootstrap CI (AUROC)**: [0.500, 0.500]

#### MLP
- **AUROC**: 0.0
- **AUPRC**: 0.14285714285714285
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.2125280726782692
- **95% Bootstrap CI (AUROC)**: [0.000, 0.000]

### Horizon 24 Months

- **Eligible Patients (MCI)**: 67
- **Positive Cases**: 12
- **Negative Cases**: 26
- **Censored/Unknown Cases**: 29
- **Train/Val/Test Counts**: 28 / 5 / 5

#### LogisticRegression
- **AUROC**: 1.0
- **AUPRC**: 1.0
- **F1**: 0.6666666666666666
- **Sensitivity (Recall)**: 0.5
- **Specificity**: 1.0
- **Brier Score**: 0.29842064431857496
- **95% Bootstrap CI (AUROC)**: [1.000, 1.000]

#### HistGradientBoosting
- **AUROC**: 0.5
- **AUPRC**: 0.8
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.503061224489796
- **95% Bootstrap CI (AUROC)**: [0.500, 0.500]

#### MLP
- **AUROC**: 0.0
- **AUPRC**: 0.6791666666666667
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.31918420322668306
- **95% Bootstrap CI (AUROC)**: [0.000, 0.000]

### Horizon 36 Months

- **Eligible Patients (MCI)**: 67
- **Positive Cases**: 16
- **Negative Cases**: 16
- **Censored/Unknown Cases**: 35
- **Train/Val/Test Counts**: 24 / 4 / 4

#### LogisticRegression
- **AUROC**: 1.0
- **AUPRC**: 1.0
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.3662421942491742
- **95% Bootstrap CI (AUROC)**: [1.000, 1.000]

#### HistGradientBoosting
- **AUROC**: 0.5
- **AUPRC**: 0.75
- **F1**: 0.8571428571428571
- **Sensitivity (Recall)**: 1.0
- **Specificity**: 0.0
- **Brier Score**: 0.25
- **95% Bootstrap CI (AUROC)**: [0.500, 0.500]

#### MLP
- **AUROC**: 0.6666666666666666
- **AUPRC**: 0.9166666666666665
- **F1**: 0.0
- **Sensitivity (Recall)**: 0.0
- **Specificity**: 1.0
- **Brier Score**: 0.3209831561483696
- **95% Bootstrap CI (AUROC)**: [0.000, 1.000]

### Horizon 48 Months

- **Eligible Patients (MCI)**: 67
- **Positive Cases**: 20
- **Negative Cases**: 5
- **Censored/Unknown Cases**: 42
- **Train/Val/Test Counts**: 19 / 3 / 3

#### LogisticRegression
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

#### HistGradientBoosting
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

#### MLP
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

### Horizon 60 Months

- **Eligible Patients (MCI)**: 67
- **Positive Cases**: 21
- **Negative Cases**: 3
- **Censored/Unknown Cases**: 43
- **Train/Val/Test Counts**: 18 / 3 / 3

#### LogisticRegression
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

#### HistGradientBoosting
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

#### MLP
- **AUROC**: N/A
- **AUPRC**: N/A
- **F1**: N/A
- **Sensitivity (Recall)**: N/A
- **Specificity**: N/A
- **Brier Score**: N/A
- **95% Bootstrap CI (AUROC)**: N/A

---
**Disclaimer**: Do not optimize aggressively for maximum accuracy. Do not perform test-set hyperparameter tuning. Do not claim clinical validity. Do not claim quantum advantage. The purpose of this experiment is to establish the REAL DATA BASELINE that future Nāḍī-Q experiments must beat.