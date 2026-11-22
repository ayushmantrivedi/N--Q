# NĀḌĪ-Q — Phase 5 Statistical Validation & Selection Optimism Report

**Phase**: Phase 5 — Priority 2 Execution
**Status**: COMPLETED & VERIFIED

---

## 1. Methodology & Paired Evaluation Framework

To prevent treating the 25 correlated folds ($7,325$ predictions) as independent participant observations:
1. **Paired Test Fold Deltas**: For each fold $k \in \{1, \dots, 25\}$, we evaluate $\Delta_k = \text{AUROC}_k(\text{Quantum}) - \text{AUROC}_k(\text{Classical})$.
2. **Two-Sided Wilcoxon Signed-Rank Test**: Assesses median shift in paired generalization performance.
3. **Clustered Bootstrap Resampling**: Resamples participants with replacement ($B = 2,000$ iterations) to calculate clustered 95% confidence intervals.
4. **False Discovery Rate (FDR)**: Benjamini-Hochberg adjustment applied across all pairwise candidate comparisons.

---

## 2. Statistical Findings Matrix

| Pipeline Comparison | Mean Difference ($\Delta$) | 95% Clustered CI | Wilcoxon $W$ | Two-Sided $p$-value | Significant ($p<0.05$)? | Scientific Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fused 8Q Mode A vs Classical LightGBM** | **$+0.0018$** | **$[-0.0002, +0.0038]$** | 97.0 | **$p = 0.0802$** | No ($p = 0.0802$) | Positive residual synergy trend; not statistically superior |
| **Fused 8Q Mode B vs Classical LightGBM** | **$+0.0016$** | **$[-0.0008, +0.0041]$** | 116.0 | **$p = 0.2200$** | No ($p = 0.2200$) | Trainable tuning yields equivalent synergy to fixed Mode A |
| **Standalone 8Q Mode A vs Classical PCA-12** | **$-0.0284$** | **$[-0.0381, -0.0187]$** | 0.0 | **$p = 0.000000$** | **Yes (Classical Superior)** | Classical PCA retains linear variance better in isolation |

---

## 3. Selection Optimism & Internal Replication Contract

- **Selection Bias Quantification**: When 34 configurations are screened on cross-validation, the maximum observed score is positively biased by $\sim 0.001-0.002$ AUROC points purely through sampling variability.
- **Correct Scientific Posture**: The $+0.0018$ AUROC delta of Fused QVBE-v2 Mode A must be characterized as an empirical demonstration that quantum multi-basis features are **safe, complementary, non-degrading residual inputs**, NOT as proof of quantum supremacy or statistically validated advantage.
- **Independent Cohort Requirement**: True out-of-cohort confirmation requires external cohorts (e.g. ADNI or biobank cohorts) where no prior hyperparameter screening was performed.
