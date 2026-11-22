# Evaluation Protocol — Nāḍī-Q

> **Status markers:**  
> **[A]** Established · **[B]** Hypothesis · **[C]** Experimental Claim · **[D]** Future Architecture

---

## 1. Purpose

This document defines the metrics, statistical tests, and reporting standards for all experiments in Nāḍī-Q. It is designed to prevent a common failure mode in ML research: reporting best-case metrics without uncertainty quantification, on cherry-picked seeds, with inappropriate comparison.

---

## 2. Task Definitions

### 2.1 Primary Task: Binary Risk Stratification [C]

The initial experimental task is **binary disease risk stratification**: given a patient's feature vector, predict whether the patient belongs to a high-risk or low-risk group within a defined time horizon.

**This is a proxy for clinical prediction, not a clinical prediction.** Results on synthetic or benchmark datasets do not imply clinical utility.

### 2.2 Label Definitions

- **Positive class:** High-risk (disease event within prediction window)
- **Negative class:** Low-risk
- **Label source:** Must be explicitly documented per dataset
- **Prediction window:** Must be explicitly documented per experiment (e.g., 5-year risk)

---

## 3. Primary Metrics [A]

For binary risk stratification:

| Metric | Symbol | Description | Justification |
|--------|--------|-------------|---------------|
| Area Under ROC Curve | AUROC | Threshold-independent discrimination | Standard for risk models |
| Area Under PR Curve | AUPRC | Accounts for class imbalance | Required for imbalanced datasets |
| Brier Score | BS | Proper scoring rule for calibration | Calibration quality |
| Expected Calibration Error | ECE | Calibration in probability bins | Clinical interpretability |
| F1 Score (at threshold) | F1 | Harmonic mean of precision/recall | Threshold-specific summary |

**Rule:** AUROC alone is insufficient. Brier Score and ECE must always be reported to assess calibration, not just discrimination.

### 3.1 Threshold Selection

If a hard decision threshold is needed (for F1, precision, recall):
- Threshold must be selected on the **validation set only**.
- The selected threshold must be fixed before evaluating on the test set.
- The threshold selection strategy (Youden's J, F1-optimal, fixed prevalence prior) must be documented.

---

## 4. Uncertainty Quantification [A]

### 4.1 Bootstrap Confidence Intervals

For all reported metrics:
- Compute B = 1000 bootstrap samples on the test set predictions.
- Report 95% CI using the percentile method.
- Report as: `metric ± CI_half_width [CI_lower, CI_upper]`

Example: `AUROC = 0.812 [0.781, 0.843]`

### 4.2 Multiple Runs

- For any stochastic component (neural network, quantum circuit initialization), run with at least 5 different seeds.
- Report mean ± std across seeds alongside the CI.
- Report the best seed and the worst seed to show variance.

---

## 5. Comparison Tests [A]

### 5.1 AUROC Comparison

Use the **DeLong test** for comparing two AUROC values from the same test set. This accounts for the correlation between predictions from models evaluated on the same samples.

```python
# Reference: delong_roc_variance (Sun & Xu, 2014)
from scipy import stats
# p-value from DeLong test; report two-sided
```

### 5.2 Other Metric Comparisons

For Brier Score, F1, or other metrics:
- Use the **Wilcoxon signed-rank test** on per-sample scores if predictions are paired.
- Use **McNemar's test** for paired binary predictions.
- Report both uncorrected and Bonferroni-corrected p-values when comparing more than two models.

### 5.3 Significance Threshold

- α = 0.05 (two-sided) for primary comparisons.
- α = 0.01 for claims of strong significance.
- All p-values reported to 3 significant figures; p < 0.001 reported as "< 0.001".

---

## 6. Model Comparison Framework

### 6.1 Comparison Table Format

Every experiment report must include a table in this format:

```
| Model                    | AUROC [95% CI]      | AUPRC [95% CI]      | Brier Score | p-value vs. best classical |
|--------------------------|---------------------|---------------------|-------------|---------------------------|
| Logistic Regression (L2) | 0.XXX [0.XXX–0.XXX] | 0.XXX [0.XXX–0.XXX] | 0.XXX       | (baseline)                |
| GBT (XGBoost)            | 0.XXX [0.XXX–0.XXX] | ...                 | ...         | DeLong p = 0.XXX          |
| MLP (matched params)     | ...                 | ...                 | ...         | ...                       |
| QVBE + Classical head    | ...                 | ...                 | ...         | DeLong p = 0.XXX          |
| QCDF (full hybrid)       | ...                 | ...                 | ...         | DeLong p = 0.XXX          |
```

### 6.2 Fairness Reporting [D]

Once real data is available, the following subgroup analyses must be conducted:
- Performance stratified by age group, sex, and (where available) ethnicity.
- Disparate impact analysis between demographic groups.

**Status:** [D] Future — not applicable without real patient data.

---

## 7. Ablation Reporting

For each ablation experiment:
1. Identify the component being ablated.
2. Define the baseline (full model) and ablated variant (component removed or replaced).
3. Report the delta metric: `Δ = metric_full − metric_ablated`.
4. Report statistical significance of the delta.
5. Interpret the delta: positive Δ means the component helps; negative Δ means it hurts.

---

## 8. Quantum-Specific Evaluation

### 8.1 Circuit Resource Reporting [A]

For every evaluated quantum model, report:
- Number of qubits (n)
- Circuit depth (L)
- Total gate count (single-qubit and two-qubit)
- Number of trainable parameters (|θ|)
- Total shots per forward pass
- Estimated wall-clock time per epoch (simulation and extrapolated real-device)

### 8.2 Noise Sensitivity Analysis [B]

For noise experiments:
- Report metric vs. noise level (depolarizing probability p from 0.0 to 0.1).
- Identify the noise threshold at which performance degrades below the classical baseline.
- Compare noise robustness across encoding strategies.

### 8.3 Barren Plateau Diagnostics [A]

For circuit training:
- Plot gradient variance vs. circuit depth for each ansatz.
- Flag circuits where gradient variance < 1e-6 as potentially suffering from barren plateaus.
- Do not train circuits exhibiting barren plateau behaviour without mitigation strategy.

---

## 9. Result Storage and Archiving

All numerical results must be stored in machine-readable format (JSON) alongside human-readable reports:

```json
{
  "experiment_id": "qvbe_rq1b_angle_encoding_20241115",
  "model": "qvbe_v1_angle",
  "dataset": "synthetic_tabular_v1",
  "seed": 42,
  "test_metrics": {
    "auroc": 0.812,
    "auroc_ci_lower": 0.781,
    "auroc_ci_upper": 0.843,
    "auprc": 0.744,
    "brier_score": 0.182,
    "ece": 0.031
  },
  "vs_best_classical": {
    "baseline_model": "xgboost_v1",
    "delong_p_value": 0.143,
    "significant_at_0.05": false
  }
}
```

---

## 10. Publication Checklist

Before any result is written up or shared:

- [ ] Test set evaluated exactly once
- [ ] All 4 classical baselines included
- [ ] Bootstrap CIs reported for all metrics
- [ ] Statistical significance tests applied
- [ ] Multiple seed variance reported
- [ ] Config file and requirements_lock.txt committed
- [ ] Null results (where H₀ not rejected) reported fully
- [ ] No cherry-picked seeds
- [ ] Simulation vs. hardware clearly distinguished
