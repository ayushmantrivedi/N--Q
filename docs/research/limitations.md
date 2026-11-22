# Limitations — Nāḍī-Q

> This document is a first-class research artifact. Limitations are not an afterthought.  
> Honest documentation of limitations is a prerequisite for scientific credibility.

---

## 1. Fundamental Quantum Computing Limitations

### 1.1 NISQ-Era Hardware Constraints [A]

Current quantum hardware (as of 2024–2025) is **not mature enough** for reliable clinical inference at scale. Specific constraints:

- **Gate errors:** Two-qubit gate fidelities of 98–99.5% compound rapidly. A circuit with 50 two-qubit gates accumulates ~10–40% cumulative error probability.
- **Coherence times:** Circuits must complete within T1/T2 times (microseconds to milliseconds). Deep circuits decohere before completion.
- **Limited connectivity:** Real devices have sparse qubit connectivity graphs, requiring SWAP gates that add overhead and error.
- **Measurement shot noise:** Expectation values estimated from finite shots have standard error ∝ 1/√shots. High-precision outputs require many shots, which are slow on real hardware.

**Implication:** Any experiment on real hardware today will be significantly noisier than simulation suggests. Results from noiseless or lightly-noisy simulation should not be extrapolated to current real hardware without explicit noise benchmarking.

### 1.2 Barren Plateaus [A]

Parameterized quantum circuits of depth O(n) or greater, with random initialization and global observables, are known to suffer from **exponentially vanishing gradients** (barren plateaus; McClean et al., 2018). This means:

- Training may fail entirely on deep circuits.
- This is not a bug in the software; it is a fundamental property of the loss landscape.
- Mitigation strategies exist (layer-by-layer training, local observables, structured initialization) but add complexity.

### 1.3 Classical Simulability [A]

For small qubit counts (n ≤ 40–50), quantum circuits can be exactly simulated classically. Any advantage we observe on a simulator could in principle be replicated by a classical algorithm. This does not invalidate the research but means:

- **Simulation results do not prove quantum advantage.**
- Demonstrating classical indistinguishability (dequantization) is an important null check.

---

## 2. Data Limitations

### 2.1 No Real Clinical Data [A]

This platform currently has no access to real patient data. All initial experiments will use:
- Synthetic data (generated from known distributions)
- Publicly available benchmark datasets (once appropriate access is established)

**Results on synthetic data may not generalize to real clinical distributions.** Synthetic data generation involves assumptions about feature correlation structure, label mechanisms, and class balance that are unlikely to match real-world data.

### 2.2 Label Quality [A]

Clinical labels (disease risk within a time window) are inherently noisy:
- Ground truth may be based on administrative codes, not direct diagnosis.
- Time-to-event information may be incomplete (censoring).
- Label definitions vary across datasets and institutions.

We do not have a solution to this; it is an acknowledged limitation of clinical ML in general.

### 2.3 Distribution Shift [A]

Models trained on one dataset (even a real one) are known to degrade when deployed in a different clinical environment. This platform makes no claims about generalizability.

---

## 3. Modeling Limitations

### 3.1 Quantum Circuit Expressibility vs. Trainability [A]

There is a known tension (Cerezo et al., 2021) between:
- **Expressibility:** The ability of a circuit family to represent arbitrary unitaries.
- **Trainability:** The ability to optimize the circuit parameters using gradient descent.

Highly expressive circuits often suffer from barren plateaus. We do not yet know where the optimal design point is for clinical tabular data. This is part of the research.

### 3.2 Hyperparameter Sensitivity [B]

PQC performance is likely sensitive to:
- Encoding strategy choice
- Ansatz depth and structure
- Optimizer (Adam, SPSA, QNG)
- Learning rate schedule
- Number of shots

We do not have evidence yet that any of these choices is robust across datasets. Extensive ablation is required before drawing any conclusion.

### 3.3 Feature Encoding Is Lossy [A]

Most practical encoding strategies for tabular data (angle encoding) encode at most n features into n qubits. For D ≫ n, a classical dimensionality reduction step is required before quantum encoding. This means the quantum circuit sees a projected (potentially information-losing) representation of the data, not the raw features.

---

## 4. Evaluation Limitations

### 4.1 AUROC Is Not a Clinical Metric [A]

AUROC measures discrimination but not calibration. A model with AUROC = 0.85 may assign probabilities that are systematically too high or too low. Clinical decision-making requires calibration (ECE, Brier Score) alongside discrimination. This protocol enforces this, but the fundamental limitation remains.

### 4.2 Small Sample Sizes [C]

Initial experiments on synthetic data with N = 1000–10000 samples. Confidence intervals will be wide. Statistical significance of any difference between quantum and classical models is uncertain at this scale. We make no claims based on experiments conducted below N = 5000 unless explicitly noted.

### 4.3 Multiple Comparisons Risk [A]

Searching across many hyperparameter combinations, encoding strategies, and ansatz designs increases the risk of false positives even with Bonferroni correction. Any result that emerges from a hyperparameter sweep must be validated on a fresh hold-out set not used during the sweep.

---

## 5. Scope Limitations

### 5.1 Not a Clinical Tool [A]

This platform is explicitly not a clinical decision-support system. It has not been:
- Evaluated on real patient cohorts
- Reviewed by clinical domain experts
- Validated against clinical guidelines
- Approved by any regulatory body

### 5.2 No Fairness Analysis Yet [A]

Without real patient data with demographic information, fairness analysis is not possible. This is a critical gap: quantum ML models may exhibit biases inherited from their training data. This must be addressed before any real-data experiments.

### 5.3 Explainability Is an Open Problem [B]

The proposed Q-XAI component (quantum explainability) is a research direction. There is no established consensus on how to produce clinically meaningful attributions from quantum circuits. Classical attribution methods (SHAP, integrated gradients) apply to the classical components only. Quantum-specific attribution methods are the subject of active research with no settled answer.

---

## 6. Summary Table

| Limitation | Category | Severity | Mitigation |
|------------|----------|----------|-----------|
| NISQ noise degrades circuits | Hardware | High | Noise simulation before hardware |
| Barren plateaus in deep circuits | Algorithm | High | Limit depth; monitor gradient variance |
| No real clinical data | Data | High | Use real public data when ethically available |
| Synthetic data generalization | Data | High | Explicit statement in all results |
| Classical simulability of small circuits | Theory | Medium | Dequantization baseline |
| Hyperparameter sensitivity | Modeling | Medium | Full ablation required |
| AUROC not sufficient alone | Evaluation | Medium | Always report Brier + ECE |
| Multiple comparisons | Evaluation | Medium | Bonferroni correction |
| No fairness analysis | Ethics | High (for real data) | Required before real-data experiments |
| No clinical validation | Regulatory | Critical | Explicit disclaimer always present |
