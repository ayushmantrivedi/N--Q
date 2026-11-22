# Experimental Protocol — Nāḍī-Q

> **All experiments conducted using this platform must adhere to this protocol.**  
> Deviating from this protocol without documentation invalidates comparability between runs.

---

## 1. Reproducibility Requirements

### 1.1 Random Seed Policy

Every experiment must specify a global random seed in its configuration. The seed must be applied to:

- Python's `random` module
- NumPy's random generator (`numpy.random.default_rng(seed)`)
- PyTorch (if used): `torch.manual_seed(seed)`
- PennyLane (if used): via device and circuit initialization
- Qiskit (if used): `algorithm_globals.random_seed = seed`
- Scikit-learn (if used): all estimators with `random_state=seed`

**Rule:** A single `seed` field in the config must flow through to all stochastic operations. No stochastic operation may use an uncontrolled source of randomness.

### 1.2 Configuration Versioning

- Every experiment run must log its full resolved configuration (after OmegaConf interpolation) to the run artifact directory.
- Configuration files are version-controlled. Experiments reference a specific config file; they do not accept ad-hoc command-line hyperparameter overrides except via explicit `--override` flags that are also logged.

### 1.3 Environment Locking

Before publishing any result:
- Export `pip freeze > requirements_lock.txt` and commit alongside the result.
- Record Python version, OS, and hardware (CPU/GPU/QPU) in the run metadata.

---

## 2. Experiment Structure

### 2.1 Every Experiment Must Produce

```
experiments/runs/<experiment_name>_<timestamp>/
├── config_resolved.yaml      # Full resolved config
├── metadata.json             # Environment, hardware, runtime
├── metrics/
│   ├── train_metrics.json
│   ├── val_metrics.json
│   └── test_metrics.json     # Only computed ONCE at the end
├── checkpoints/              # Model weights (if applicable)
├── logs/
│   └── run.log
└── figures/                  # Generated evaluation plots
```

### 2.2 Test Set Usage Policy

**Critical rule:** The test set is touched exactly **once** — at final evaluation after all hyperparameter selection is complete. Any use of the test set during development (including visual inspection of predictions) invalidates the experiment.

Procedure:
1. All hyperparameter selection and model development on training + validation sets only.
2. Final architecture and hyperparameters frozen.
3. Model retrained on train + validation (optionally).
4. Test set evaluated exactly once.
5. Results reported with confidence intervals.

---

## 3. Baseline Requirements

**No result from a quantum model is meaningful without a properly tuned classical baseline.**

For every experiment involving a quantum or hybrid model, the following baselines must be trained and reported:

| Baseline | Rationale |
|----------|-----------|
| Logistic Regression (L2) | Linear baseline; defines a minimum bar |
| Gradient-Boosted Trees | Strong tabular baseline (XGBoost or LightGBM) |
| MLP (matched parameter count) | Controls for parameter count vs. quantum circuit |
| Classical kernel SVM (RBF) | Controls for kernel-like non-linearity |

Any claim of "quantum model performs better than classical" requires all four baselines to be beaten with statistical significance.

---

## 4. Statistical Reporting

### 4.1 Required Statistics

- Point estimate of each metric (AUROC, AUPRC, F1, calibration error)
- 95% bootstrap confidence interval (B ≥ 1000 bootstrap samples)
- p-value from an appropriate two-sample test (DeLong test for AUROC; paired t-test or Wilcoxon for others)
- Effect size (Cohen's d or equivalent)

### 4.2 Multiple Comparisons

If comparing across more than two models:
- Apply Bonferroni or Holm-Bonferroni correction.
- Report uncorrected p-values alongside corrected ones.

### 4.3 Reporting Standards

- Never report a metric without its confidence interval.
- Never report a comparison without a significance test.
- Never claim "improvement" if the confidence intervals overlap substantially.
- Null results (H₀ not rejected) must be reported fully, not omitted.

---

## 5. Ablation Study Design

Every major model component must have a corresponding ablation experiment that isolates its contribution.

Example ablations for QVBE (once implemented):
- Full model vs. model with quantum layer replaced by equivalent-dimensional linear projection
- Full model vs. model with quantum layer replaced by random Fourier features (RFF) kernel
- Varying circuit depth (L = 1, 2, 4, 8 layers)
- Varying qubit count (n = 4, 8, 16)
- Varying encoding scheme (angle, amplitude, re-uploading)
- Noise-free simulation vs. noise-injected simulation

---

## 6. Experiment Naming Convention

```
<component>_<hypothesis_id>_<variant>_<date>

Examples:
  qvbe_rq1b_angle_encoding_20241115
  classical_baseline_logreg_tabular_20241115
  ablation_encoding_sweep_20241115
```

---

## 7. What Must Never Be Done

- Do not train on the test set.
- Do not tune hyperparameters on the test set.
- Do not report test results before finalizing the model.
- Do not cherry-pick random seeds.
- Do not run an experiment and report only the best seed.
- Do not fabricate, smooth, or selectively omit metrics.
- Do not claim a result is "better" without statistical significance.
- Do not compare models trained on different data splits.
- Do not present simulation results as real-device results.
