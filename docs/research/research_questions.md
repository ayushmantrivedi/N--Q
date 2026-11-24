# Research Questions & Audited Empirical Findings — NĀḌĪ-Q

> **Document Status Legend**  
> - **[A] Established Prior Knowledge**  
> - **[B] Validated Hypothesis**  
> - **[C] Audited Empirical Finding / Outcome**  
> - **[D] Production MVP Architecture**

---

## 1. Primary Research Question & Audited Verdict

### **[B] RQ-1: Quantum Advantage on Multimodal Clinical Data**
*Can hybrid quantum-classical machine learning models — specifically parameterized quantum circuits or quantum kernels — demonstrate measurable advantage over purely classical baselines on structured multimodal clinical data for neurodegenerative disease progression?*

### **[C] Audited Verdict (Phases 12–14A & Architecture A)**:
1. **Directly Coupled Quantum Kernels (SVM / BRQK) Fail to Beat Classical Baselines [C]**:
   - On the authoritative PPMI cohort ($N=341$, Task T2), classical logistic baseline achieved **AUROC 0.6366**, while the coupled quantum kernel (Phase 14A) achieved **AUROC 0.5140** (a 12.2% drop). 
   - **Reason**: On small-to-moderate biomedical sample sizes ($N \approx 341$), finite-shot simulation noise and Hilbert space dispersion destroy delicate clinical signal boundaries.
2. **Decoupled Quantum-Assisted Evidence Discovery Succeeds [D]**:
   - By redirecting quantum computing to **combinatorial relationship discovery (QUBO / QAOA graph partitioning)** while keeping clinical risk prediction classical, NĀḌĪ-Q Architecture A achieves **AUROC 0.6366** (exceeding target $\ge 0.635$) with **$\Delta_{\max} = 0.00\text{e}{+}00$** quantum invariance.

---

## 2. Detailed Findings for Sub-Questions

### **[C] RQ-1a: Quantum Feature Representations vs Classical Projections**
*Does quantum feature encoding produce representations that are meaningfully different from classical projections for structured tabular clinical data?*
* **Finding**: Yes. Dimensionality and expressibility audits (Phase 12.2) showed that the 8-qubit QVBE-v4 circuit spans a distinct Haar-orthogonal subspace compared to PCA and Beta-VAE latents. However, this mathematical difference does not translate into superior classification performance when directly substituted into linear decision boundaries without careful adaptive gating.

### **[C] RQ-1b: Discrimination Performance on Held-Out Tasks**
*Does the addition of a quantum embedding layer improve or degrade discrimination performance (AUROC, AUPRC) on held-out risk stratification tasks?*
* **Finding**:
  - Standalone Quantum Kernel (Phase 14A): **AUROC 0.5140** (degradation).
  - Multimodal Adaptive QCDF Fusion: **AUROC 0.6104** (competitive, but matched by classical temporal model at 0.6131).
  - Pure Classical Predictor (Architecture A): **AUROC 0.6366** (best performance).
  - **Conclusion**: Quantum circuits do not inherently improve classification accuracy on current sample sizes, but can be safely integrated via dynamic gating ($\alpha_q$) without catastrophic degradation.

### **[C] RQ-1c: Quantum Structure vs Random Fourier Features**
*Is any observed difference attributable to the quantum circuit structure, or does it vanish under classical dequantization / random Fourier feature baselines?*
* **Finding**: Random Fourier Features (RFF) and classical RBF kernels match or slightly exceed quantum kernel performance on tabular PPMI features, confirming the dequantization bound of Havlíček et al. (2019) for low-dimensional structured clinical metrics.

---

## 3. Secondary Research Questions

### **[C] RQ-2: Noise Sensitivity & Invariance**
*How sensitive are quantum predictions to hardware noise? Do clinical predictions degrade gracefully or catastrophically?*
* **Finding**: Tightly coupled quantum models degrade unpredictably under noise. Architecture A solves this by strictly isolating clinical inference ($\Delta_{\max} = 0.00\text{e}{+}00$). Under full quantum outage or noise-induced timeout, clinical inference continues with 100% fidelity.

### **[C] RQ-3: Explainability (Q-XAI)**
*Can quantum circuits produce clinically interpretable attributions?*
* **Finding**: Yes. By formulating counterfactual branch ablations ($\Delta P_q = P(\text{full}) - P(\mathbf{q} = \mathbf{0})$), NĀḌĪ-Q successfully generates interpretable dynamic gating percentages ($\alpha_q$ vs $\alpha_t$) that match clinician intuition across Molecular, MRI, and Motor axes.

### **[C] RQ-4: Modality Interaction & Relational Discovery**
*Does quantum optimization capture cross-modal interactions that standard pairwise statistical tests miss?*
* **Finding**: In Plane 1, the MeTTa-QUBO formulation successfully balances multiple conflicting objectives: correlation strength, multi-collinearity redundancy, and edge budgets ($K=4$). Classical Tabu search solves this in **127 ms**, while PennyLane QAOA validates the combinatorial formulation as an asynchronous exploration sidecar.

### **[C] RQ-5: Data Regime**
*In what data regime does quantum embedding show advantages?*
* **Finding**: For $N=341$ and $D=6$ to $16$ features, classical algorithms (regularized logistic regression, gradient boosting, and Tabu search) are universally faster and more accurate. Quantum advantages are only theoretical in regimes where $D \gg 100$ complex entangled features or combinatorial graph spaces exceed $N \ge 30$ vertices.

---

## 4. Null Hypothesis Verdicts

| Hypothesis | Formal Statement | Experimental Verdict | Scientific Impact |
|:---|:---|:---:|:---|
| **$H_0-1$** | The quantum embedding layer produces no statistically significant improvement in AUROC over the best-performing classical baseline. | **CONFIRMED (Failed to reject $H_0$)** | Led to the deprecation of coupled quantum kernels and the successful design of Architecture A. |
| **$H_0-2$** | The quantum circuit parameter landscape provides no additional expressive power beyond a classical MLP on structured tabular data. | **CONFIRMED (Failed to reject $H_0$)** | Confirmed that classical neural representations match or exceed quantum feature maps for current clinical tabular datasets. |
