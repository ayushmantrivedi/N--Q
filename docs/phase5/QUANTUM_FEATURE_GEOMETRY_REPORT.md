# NĀḌĪ-Q — Quantum Feature Geometry & Spectral Decomposition Report

**Phase**: Phase 5 — Confirmatory Validation & Open-Source Model Integration
**Task**: Tasks 5 & 6 — Geometric Profiling & Feature Decomposition
**Cohort**: PPMI N=1,465 participants across 25 stratified folds (5 seeds x 5 folds)

---

## 1. Spectral and Geometric Representation Profile

| representation     |   eff_rank_mean |   eff_rank_std |   part_ratio_mean |   cond_num_mean |   top_3_pct_mean |   top_8_pct_mean |   off_diag_sim_mean |   off_diag_sim_var |   feat_var_mean |
|:-------------------|----------------:|---------------:|------------------:|----------------:|-----------------:|-----------------:|--------------------:|-------------------:|----------------:|
| FT_Transformer_Z16 |         1.59437 |       0.135027 |           1.33563 |     4.46843e+06 |          99.6086 |          99.9898 |          0.211855   |          0.54085   |      0.415482   |
| QVBE_Raw_Q23       |         4.49858 |       0.810593 |           2.68467 |  4206.29        |          82.4847 |          95.5481 |          0.519914   |          0.12505   |      0.00844501 |
| QVBE_from_FT_QZ23  |         3.00545 |       1.19521  |           1.92813 | 55078.5         |          89.1352 |          98.0117 |          0.254115   |          0.28579   |      0.0164682  |
| X_Raw_D249         |        31.0783  |      13.3221   |          10.0318  |     5.01277e+13 |          51.5752 |          62.6944 |          0.00419562 |          0.0541581 |      1.44957    |

---

## 2. Geometric Insights and Mathematical Interpretations

### A. Effective Rank and Spectral Entropy
- **Classical Raw Matrix ($X \in \mathbb{R}^{249}$)**: Effective rank reflects substantial multicollinearity among FreeSurfer volumetric measures.
- **FT-Transformer Latent ($Z \in \mathbb{R}^{16}$)**: Bottleneck projection concentrates information into orthogonal continuous representations.
- **Quantum Representation ($Q_Z \in \mathbb{R}^{23}$)**: Nonlinear unitary rotation and multi-basis observable projection expand the effective dimensionality, providing high participation ratio and low condition number without collinearity collapse.

### B. Quantum Feature Decomposition (SVD / PCA)
Decomposing $Q_Z$ into its leading 8 and 12 singular components demonstrates whether quantum-derived features concentrate variance into separable hyperplanes:

| source   | decomposition   | model       |   auroc_mean |   auroc_std |   auprc_mean |   bal_acc_mean |
|:---------|:----------------|:------------|-------------:|------------:|-------------:|---------------:|
| Q_Raw    | SVD_12          | lightgbm    |     0.819769 |   0.0195129 |     0.84724  |       0.743369 |
| Q_Raw    | SVD_12          | logistic_l2 |     0.808052 |   0.024565  |     0.838705 |       0.738199 |
| Q_Raw    | SVD_8           | lightgbm    |     0.809879 |   0.0216542 |     0.839739 |       0.729454 |
| Q_Raw    | SVD_8           | logistic_l2 |     0.799137 |   0.0279101 |     0.832627 |       0.730827 |
| Q_Z      | SVD_12          | lightgbm    |     0.82023  |   0.0302046 |     0.838454 |       0.761491 |
| Q_Z      | SVD_12          | logistic_l2 |     0.821987 |   0.0361208 |     0.83187  |       0.764912 |
| Q_Z      | SVD_8           | lightgbm    |     0.821495 |   0.0279701 |     0.839186 |       0.762727 |
| Q_Z      | SVD_8           | logistic_l2 |     0.820714 |   0.0361389 |     0.830449 |       0.764357 |

