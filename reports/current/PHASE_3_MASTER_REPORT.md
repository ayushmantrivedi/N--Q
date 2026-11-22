# NĀḌĪ-Q — PHASE 3: MASTER BENCHMARK EVALUATION & QUANTUM-CLASSICAL FAIR COMPARISON REPORT
**Document ID**: `NADIQ-PHASE3-MASTER-001`  
**Location**: `F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\PHASE_3_MASTER_REPORT.md`  
**Execution Date**: 2026-11-03  
**Status**: OFFICIALLY COMPLETED & SCIENTIFICALLY VERIFIED  

---

## 1. PRIMARY RESEARCH QUESTION & SCIENTIFIC VERDICT

> **Primary Research Question**:  
> *Can a projected quantum kernel provide measurable predictive value over strong classical machine-learning models when predicting CSF Alpha-Synuclein SAA positivity from baseline non-CSF features?*

### The Definitive Empirical Verdict: **NO.**
Across an exhaustive, leakage-free benchmark spanning **$N = 1,465$ real PPMI participants**, **25 identical stratified evaluation folds** (5 seeds $\times$ 5 folds), **73,250 quantum circuit simulations**, and **200 classical model training cycles**:
1. **Classical Machine Learning Strongly Outperforms Quantum Kernels**:
   - The best classical model (**LightGBM**) achieved a participant-level out-of-fold AUROC of **0.8664** (95% CI: $[0.8459, 0.8841]$), followed closely by **XGBoost** (AUROC **0.8649**) and **L1-regularized Logistic Regression** (AUROC **0.8639**).
   - In contrast, the best quantum architecture (**12-Qubit Projected Quantum Kernel / PQK**) achieved an AUROC of **0.7817** (95% CI: $[0.7563, 0.8041]$), and the **8-Qubit PQK** achieved **0.7727** (95% CI: $[0.7462, 0.7946]$).
2. **Statistically Significant Classical Superiority ($p < 0.001$)**:
   - The performance deficit between 12-qubit PQK and LightGBM is **$\Delta = -0.0849$ AUROC points** (95% CI: $[-0.1024, -0.0667]$, $p < 0.001$).
   - Even when comparing on **strictly matched low-dimensional representations**, a standard Classical RBF Kernel fit on the exact same 12 principal components achieved AUROC **0.8460**, outperforming 12-qubit PQK by **$\Delta = +0.0645$ AUROC points** ($p < 0.001$).
3. **No Quantum Advantage on Tabular/IDP Neurodegenerative Data**:
   - The non-linear rotation and entanglement dynamics of PQK compress patient representations into a low-variance spherical cap (kernel effective rank $\approx 1.31$, condition number $> 10^{14}$), degrading decision boundary fidelity relative to standard classical kernels.

---

## 2. COMPREHENSIVE BENCHMARK PERFORMANCE MATRIX

All models were evaluated on the **exact same 25 cross-validation folds** using identical training/testing splits from [PARTICIPANT_SPLIT_MANIFEST.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/PARTICIPANT_SPLIT_MANIFEST.csv). Point estimates represent pooled participant-level predictions ($N=1,465$), with 95% Confidence Intervals computed via 1,000 stratified bootstrap resamples:

| Model Rank & Name | Model Paradigm | Input Features | AUROC [95% CI] | AUPRC [95% CI] | Balanced Acc [95% CI] | Brier Score | ECE | Per-Fold Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. LightGBM** | Classical GBDT | Full Raw ($D=249$) | **0.8664** [0.8459, 0.8841] | **0.8888** [0.8650, 0.9081] | **0.8043** [0.7827, 0.8252] | **0.1495** | 0.0633 | **0.954 s** |
| **2. XGBoost** | Classical GBDT | Full Raw ($D=249$) | **0.8649** [0.8443, 0.8828] | **0.8843** [0.8593, 0.9051] | **0.8047** [0.7831, 0.8241] | **0.1488** | 0.0653 | 2.833 s |
| **3. Logistic Regression (L1)** | Linear Sparse | Full Raw ($D=249$) | **0.8639** [0.8427, 0.8814] | **0.8869** [0.8630, 0.9054] | **0.7943** [0.7723, 0.8145] | **0.1484** | 0.0647 | 1.734 s |
| **4. Random Forest (300 trees)** | Ensemble Trees | Full Raw ($D=249$) | **0.8589** [0.8369, 0.8770] | **0.8834** [0.8608, 0.9028] | **0.7955** [0.7737, 0.8169] | 0.1511 | 0.0692 | 3.361 s |
| **5. Classical RBF (Raw)** | Kernel SVM | Full Raw ($D=249$) | **0.8551** [0.8335, 0.8728] | **0.8773** [0.8531, 0.8984] | **0.7876** [0.7649, 0.8086] | 0.1524 | 0.0591 | 0.629 s |
| **6. Classical RBF (PCA-12)** | Kernel SVM | 12 Components | **0.8460** [0.8230, 0.8664] | **0.8526** [0.8222, 0.8796] | **0.7808** [0.7582, 0.8019] | 0.1545 | 0.0569 | **0.188 s** |
| **7. Classical RBF (PCA-8)** | Kernel SVM | 8 Components | **0.8414** [0.8190, 0.8617] | **0.8525** [0.8233, 0.8785] | **0.7792** [0.7569, 0.8016] | 0.1562 | **0.0536** | **0.174 s** |
| **8. Logistic Regression (L2)** | Linear Ridge | Full Raw ($D=249$) | **0.8410** [0.8176, 0.8599] | **0.8662** [0.8409, 0.8896] | **0.7647** [0.7414, 0.7865] | 0.1674 | 0.0834 | **0.062 s** |
| **9. PQK (12-Qubit)** | Quantum Hybrid | 12 Components | **0.7817** [0.7563, 0.8041] | **0.8158** [0.7836, 0.8435] | **0.7318** [0.7076, 0.7551] | 0.1891 | 0.0669 | 69.526 s |
| **10. PQK (8-Qubit)** | Quantum Hybrid | 8 Components | **0.7727** [0.7462, 0.7946] | **0.8120** [0.7803, 0.8405] | **0.7203** [0.6969, 0.7437] | 0.1919 | 0.0616 | 43.402 s |

---

## 3. STATISTICAL HYPOTHESIS TESTING & PAIRED DIFFERENCES

To rigorously avoid treating the 25 correlated folds as independent measurements, paired statistical hypothesis testing was performed on participant-level out-of-fold predictions:

$$\Delta \text{AUROC} = \text{AUROC}(\text{Quantum}) - \text{AUROC}(\text{Classical})$$

| Quantum Architecture | Classical Benchmark | Representation Parity | $\Delta \text{AUROC}$ [95% CI] | Empirical $p$-value | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PQK (12-Qubit)** | **Classical RBF (PCA-12)** | **Identical 12 PCs** | **$-0.0645$** [$-0.0801, -0.0490$] | **$p < 0.001$** | **Classical RBF Statistically Superior** |
| **PQK (8-Qubit)** | **Classical RBF (PCA-8)** | **Identical 8 PCs** | **$-0.0688$** [$-0.0850, -0.0529$] | **$p < 0.001$** | **Classical RBF Statistically Superior** |
| **PQK (12-Qubit)** | **LightGBM** | Classical Raw vs Quantum | **$-0.0849$** [$-0.1024, -0.0667$] | **$p < 0.001$** | **LightGBM Statistically Superior** |
| **PQK (8-Qubit)** | **LightGBM** | Classical Raw vs Quantum | **$-0.0938$** [$-0.1131, -0.0748$] | **$p < 0.001$** | **LightGBM Statistically Superior** |
| **PQK (12-Qubit)** | **Logistic Regression (L1)** | Sparse Linear vs Quantum | **$-0.0823$** [$-0.0998, -0.0641$] | **$p < 0.001$** | **L1-LogReg Statistically Superior** |
| **PQK (12-Qubit)** | **Classical RBF (Raw)** | Full Kernel vs Quantum | **$-0.0735$** [$-0.0891, -0.0568$] | **$p < 0.001$** | **Classical RBF Statistically Superior** |

> **Key Statistical Finding**: In every single pairwise comparison, the 95% bootstrap confidence interval of $\Delta \text{AUROC}$ is strictly negative and bounded away from zero. There is zero evidence of quantum advantage.

---

## 4. MODALITY ABLATION EXPERIMENTS (TASK 7)

To determine which biological modalities contribute predictive signal toward CSF Alpha-Synuclein SAA positivity, systematic modality ablations were executed across all 25 folds:

```mermaid
flowchart TD
    M1["Morphometry Only (204 IDPs)<br/>AUROC = 0.6435"] --> M12["Morphometry + Motor (242 Features)<br/>AUROC = 0.8514 (+0.208)"]
    M2["Clinical Motor Only (38 Features)<br/>AUROC = 0.8366"] --> M12
    M3["Genetics & PRS Only (7 Features)<br/>AUROC = 0.6994"] --> M123["Morphometry + Motor + Genetics (249 Core)<br/>AUROC = 0.8612 (+0.010)"]
    M12 --> M123
    M123 --> M_FULL["Full Suite with Demographics (252 Features)<br/>AUROC = 0.8629 (+0.002)"]
```

### Empirical Ablation Breakdown (AUROC Mean ± Std Across 25 Folds)

| Modality Combination | Feature Count ($D$) | LightGBM | Random Forest | Logistic Regression (L2) | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Morphometry Only** | 204 FreeSurfer IDPs | $0.6211 \pm 0.0307$ | $0.6435 \pm 0.0328$ | $0.6399 \pm 0.0250$ | Modest standalone signal; brain morphometry alone cannot reliably infer SAA status. |
| **Genetics & PRS Only** | 7 Gen/PRS Features | $0.6698 \pm 0.0344$ | $0.6699 \pm 0.0298$ | $0.6994 \pm 0.0275$ | Meaningful genetic susceptibility signal (LRRK2, GBA, APOE, and Polygenic Scores). |
| **Clinical Motor Only** | 38 MDS-UPDRS Items | $0.8342 \pm 0.0197$ | $0.8366 \pm 0.0196$ | $0.8272 \pm 0.0193$ | **Dominant clinical driver**: Motor impairment in unmedicated OFF state is strongly predictive. |
| **Morphometry + Motor** | 242 Features | $0.8482 \pm 0.0227$ | $0.8514 \pm 0.0211$ | $0.8217 \pm 0.0251$ | **True Multimodal Synergy**: Adding structural MRI to motor scores provides a **+0.015 AUROC boost**! |
| **Morphometry + Motor + Genetics**| **249 Features (Core)** | **$0.8612 \pm 0.0214$** | **$0.8585 \pm 0.0199$** | **$0.8341 \pm 0.0230$** | **Optimal Biological Core**: Tri-modal integration yields maximal predictive discrimination. |
| **Full Approved Suite** | 252 Features (+Demo) | **$0.8629 \pm 0.0194$** | **$0.8601 \pm 0.0208$** | $0.8344 \pm 0.0222$ | Demographics add negligible marginal discrimination (+0.0017 AUROC). |

---

## 5. REPRESENTATION STABILITY & VARIANCE RETENTION (TASK 4)

Across the 25 training folds, PCA was fit **strictly inside each training partition** ($N = 1,172$):
- **PCA-8 Variance Retention**: Mean **$54.4\% \pm 0.3\%$** (range: $53.9\%$ to $55.0\%$).
- **PCA-12 Variance Retention**: Mean **$59.5\% \pm 0.3\%$** (range: $59.0\%$ to $59.9\%$).
- **Effective Rank of Representations**:
  - Full Standardized Multimodal Space ($D=249$): **$48.2 \pm 0.6$**
  - PCA-8 Space: **$5.07 \pm 0.03$**
  - PCA-12 Space: **$6.63 \pm 0.05$**
- **Takeaway**: While PCA variance retention is highly stable across random fold partitions ($\sigma < 0.003$), compressing 249 diverse multimodal features down to 8 or 12 dimensions discards $40-45\%$ of variance. This explains why full-dimensional classical models (LightGBM, XGBoost, LR-L1) outperform PCA-based models.

---

## 6. FINAL DELIVERABLES MANIFEST

All Phase 3 output files are saved in `F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\`:

1. [FINAL_FEATURE_TABLE_SCHEMA.md](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/FINAL_FEATURE_TABLE_SCHEMA.md) — Unified feature schema and provenance documentation.
2. [PARTICIPANT_SPLIT_MANIFEST.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/PARTICIPANT_SPLIT_MANIFEST.csv) — Exact 25-fold participant-level assignment manifest ($7,325$ rows).
3. [CLASSICAL_BENCHMARK_RESULTS.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/CLASSICAL_BENCHMARK_RESULTS.csv) — Per-fold performance metrics for all 8 classical model configurations.
4. [QUANTUM_BENCHMARK_RESULTS.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/QUANTUM_BENCHMARK_RESULTS.csv) — Per-fold performance and resource metrics for 8-qubit and 12-qubit PQK.
5. [OUT_OF_FOLD_PREDICTIONS.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/OUT_OF_FOLD_PREDICTIONS.csv) — Complete participant-level test predictions ($73,250$ rows).
6. [KERNEL_DIAGNOSTICS.md](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/KERNEL_DIAGNOSTICS.md) — Mathematical audit of kernel concentration, effective rank, and condition numbers.
7. [ABLATION_RESULTS.csv](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/ABLATION_RESULTS.csv) — 25-fold results across all 6 modality combinations.
8. [RUNTIME_AND_MEMORY_REPORT.md](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/RUNTIME_AND_MEMORY_REPORT.md) — Computational cost, memory usage, and quantum simulation budgets.
9. [PHASE_3_MASTER_REPORT.md](file:///F:/NADI-Q_AUDIT_OUTPUT/PHASE_3/PHASE_3_MASTER_REPORT.md) — This master synthesis document (also mirrored in local repo [reports/current](file:///C:/Users/ayush/OneDrive/Desktop/Nadi-Q/reports/current/PHASE_3_MASTER_REPORT.md)).

---

## 7. STRATEGIC RECOMMENDATIONS FOR NĀḌĪ-Q

1. **Adopt Classical Gradient Boosted Trees as the Production Backbone**: LightGBM and XGBoost deliver AUROC $> 0.866$, AUPRC $> 0.888$, and Balanced Accuracy $> 0.804$ at sub-second training latencies.
2. **Halt Unconstrained Quantum Kernel Scaling**: Quantum kernels (even projected formulations) do not exhibit inductive biases suitable for continuous, heterogeneous clinical tables and brain IDPs. Further hyperparameter search on standard PQK is not scientifically justified.
3. **Focus Quantum Research on Genuine Discrete Geometric Problems**: If quantum representations are pursued in later phases, they should target discrete combinatorial structures (e.g. molecular drug docking or graph-theoretic connectome graphs) rather than continuous tabular clinical records.
