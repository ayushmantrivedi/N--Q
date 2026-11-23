# NĀḌĪ-Q Forensic Validation: Clinical Metric Reconciliation
## Verification of Phase 14A Baseline vs Architecture A Clinical Inference Across Seeds and Ensembles

**Date**: November 24, 2026  
**Auditor**: Senior Biostatistician & Clinical AI Systems Auditor  
**Audited Artifacts**:
- `experiments/phase14a/PHASE14A_T2_RESULTS.csv`
- `src/nadiq/experiments/phase14a/run_phase14a.py`
- `research/pre_implementation_benchmark/scripts/harness_clinical_isolation.py`
- `research/pre_implementation_benchmark/results/clinical_isolation_results.json`

---

## 1. Executive Summary of the Discrepancy

In the benchmark audit, a small numerical variance was noted between the reported Phase 14A classical baseline metrics and the numbers reported by `harness_clinical_isolation.py`:

| Metric | Phase 14A Reference Table (`PHASE14A_T2_RESULTS.csv`) | Harness Clinical Isolation Table (`clinical_isolation_results.json`) | Absolute Variance ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Out-of-Fold AUROC** | **0.6366** | **0.6353** | $-0.0013$ |
| **Out-of-Fold AUPRC** | **0.5879** | **0.5928** | $+0.0049$ |
| **Brier Score Loss** | **0.2355** | **0.2359** | $+0.0004$ |
| **Expected Calibration Error (ECE)** | **0.0327** | **0.0615** | $+0.0288$ |

### Forensic Root Cause
**The underlying cohort ($N=341$), clinical target (Task T2: Month-12-to-Month-60 future progression $\ge 5.0$), model architecture (`LogisticRegression(C=1.0)`), and fold-contained preprocessing are 100% identical.**

The variance is entirely explained by the aggregation method:
- **Phase 14A** reported metrics on a **3-Seed Ensemble** (seeds $42, 123, 456$, where out-of-fold predicted probabilities $\hat{p}_i^{(s)}$ are averaged across the 3 seeds before metric computation).
- **`harness_clinical_isolation.py`** executed a **Single Seed (Seed 42)** 5-fold cross-validation to isolate and verify the quantum decoupling invariance.

When the live forensic validation script evaluated the 3-seed ensemble using the identical data protocol, **it reproduced the Phase 14A numbers exactly to four decimal places**.

---

## 2. Independent Empirical Verification

To verify this reconciliation, an independent validation script was executed directly on the local PPMI cohort (`ppmi_longitudinal_multimodal_biofluid_n341.parquet`, Task T2, $N=341$, 159 events):

### 2.1 Seed-by-Seed and Ensemble Metrics

```
Seed 42  (5 folds): AUROC = 0.6353 | AUPRC = 0.5928 | Brier = 0.2359 | ECE = 0.0615
Seed 123 (5 folds): AUROC = 0.6399 | AUPRC = 0.5905 | Brier = 0.2355 | ECE = 0.0302
Seed 456 (5 folds): AUROC = 0.6283 | AUPRC = 0.5791 | Brier = 0.2372 | ECE = 0.0618

3-Seed Ensemble   : AUROC = 0.6366 | AUPRC = 0.5879 | Brier = 0.2355 | ECE = 0.0327
```

### 2.2 Direct Cross-Verification Table

| Evaluation Setting | Out-of-Fold AUROC | Out-of-Fold AUPRC | Brier Score | ECE | Matches Phase 14A? | Matches Harness? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Harness Run (Seed 42)** | **0.6353** | **0.5928** | **0.2359** | **0.0615** | Single-seed split | **EXACT MATCH** |
| **Forensic Run (Seed 123)**| 0.6399 | 0.5905 | 0.2355 | 0.0302 | Single-seed split | N/A |
| **Forensic Run (Seed 456)**| 0.6283 | 0.5791 | 0.2372 | 0.0618 | Single-seed split | N/A |
| **Phase 14A Master (3 Seeds)**| **0.6366** | **0.5879** | **0.2355** | **0.0327** | **EXACT MATCH** | N/A (3-seed ens.) |
| **Forensic Run (3 Seeds)** | **0.6366** | **0.5879** | **0.2355** | **0.0327** | **EXACT MATCH** | Ensemble of 3 |

### 2.3 Why ECE Decreases from 0.0615 to 0.0327 in the Ensemble
The Expected Calibration Error (ECE) measures how closely predicted probabilities match observed event rates across 10 probability bins. 
- In a single random 5-fold split (Seed 42), partition variance creates small local bin shifts, yielding an ECE of **0.0615**.
- In the 3-seed ensemble, averaging out-of-fold predictions across 15 total folds smooths probability peaks and reduces predictive variance, cutting calibration error by nearly half to **0.0327**.

---

## 3. Quantum Prediction Invariance Verification

The central architectural claim of Architecture A is that patient clinical risk predictions are strictly decoupled from quantum optimization.

We evaluated prediction invariance across all $N=341$ participants under both single-seed and multi-seed settings:
$$\Delta_i = |P_{\text{enabled}}(y_i = 1) - P_{\text{disabled}}(y_i = 1)|$$

| Setting | Max Absolute Difference ($\max_i \Delta_i$) | Mean Absolute Difference | AUROC Delta | Status |
|:---|:---:|:---:|:---:|:---:|
| **Seed 42 Split** | **0.00e+00** | **0.00e+00** | **0.0000** | **PERFECT INVARIANCE** |
| **Seed 123 Split** | **0.00e+00** | **0.00e+00** | **0.0000** | **PERFECT INVARIANCE** |
| **Seed 456 Split** | **0.00e+00** | **0.00e+00** | **0.0000** | **PERFECT INVARIANCE** |
| **3-Seed Ensemble** | **0.00e+00** | **0.00e+00** | **0.0000** | **PERFECT INVARIANCE** |

---

## 4. Historical Contrast: Coupled Hybrid Failure vs Architecture A

| Model Configuration | AUROC | AUROC Delta vs Classical | Paired Permutation $p$-value | Clinical Viability |
|:---|:---:|:---:|:---:|:---|
| **M0 Logistic Regression Baseline** | **0.6366** | Baseline ($0.0000$) | N/A | Fully Viable |
| **M6 CTCLM Temporal Baseline** | **0.6131** | $-0.0235$ | $p = 0.284$ | Fully Viable |
| **Phase 14A M8 (CTCLM + BRQK Hybrid)** | **0.5027** | **$-0.1104$** | **$p = 0.0010$** | **REJECTED (Severe Harm)** |
| **Phase 14A M5 (Standalone BRQK)** | **0.4158** | **$-0.2208$** | **$p < 0.0001$** | **REJECTED (Collapsed)** |
| **Architecture A Decoupled Prediction** | **0.6366** | **$0.0000$** | **$p = 1.0000$** | **APPROVED (100% Protected)** |

---

## 5. Final Reconciliation Conclusion

1. **The reported clinical results are 100% genuine and fully reproduced.**
2. The number **0.6353** is the verified single-seed result (Seed 42).
3. The number **0.6366** is the verified 3-seed ensemble result (Seeds 42, 123, 456).
4. In all settings, Architecture A achieves **$\Delta_{\max} = 0.00\text{e}+00$**, completely eliminating the quantum degradation of Phase 14A.
