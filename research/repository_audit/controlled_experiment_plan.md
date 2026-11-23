# NĀḌĪ-Q Controlled Proof-of-Concept Experiment Plans

**Audit Date**: November 23, 2026  
**Auditor**: Principal AI Research Architect & Quantum ML Benchmark Engineer  
**Framework**: Strict Multi-Seed Outer 5-Fold Stratified Cross-Validation on PPMI ($N=341$)  
**Guiding Principle**: No changes to production clinical prediction paths until controlled experiments demonstrate statistically significant, reproducible superiority over audited baselines.

---

## 1. Baseline Performance Standards to Beat

All candidate experiments must be evaluated against our verified Phase 14A benchmark standards on the primary clinical endpoint (**Task T2**: Month-12-to-Month-60 landmark motor progression, 159 events / 46.6%):

| Reference Model | Primary Metric: AUROC | AUPRC | Brier Score | ECE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M0: Logistic Regression (Linear)** | **0.6366** | **0.5879** | **0.2355** | 0.0327 | **Current Top Predictor** |
| **M6: CTCLM (Temporal Motor Baseline)** | **0.6131** | **0.5589** | **0.2387** | **0.0192** | **Top Temporal Continuous Model** |
| **M4: Classical Relational Classifier** | **0.5935** | 0.5313 | 0.2479 | 0.0589 | Relational Feature Control |
| **M8: CTCLM + BRQK Quantum Hybrid** | 0.5027 | 0.4871 | 0.3270 | 0.2450 | Degraded by Quantum Branch |
| **M5: BRQK Quantum Kernel Standalone** | 0.4158 | 0.4157 | 0.2513 | 0.0059 | Collapsed (Worse than chance) |

---

## 2. Experiment 1: Longitudinal Clinical Prediction — Standalone GRU-D vs CTCLM

### 2.1 Research Objective
Determine whether adapting Che et al.'s **GRU-D** temporal decay mechanism (isolated from YAIB) outperforms or complements our continuous-time Neural ODE model (**CTCLM**) for irregular clinical visit intervals in Parkinson's progression.

### 2.2 Falsifiable Hypothesis
$$\mathcal{H}_1: \text{AUROC}(\text{GRU-D}) - \text{AUROC}(\text{CTCLM}) > 0 \quad \text{with 95% Bootstrap CI strictly }> 0$$
*Null Hypothesis $\mathcal{H}_0$*: Parameterized exponential feature and hidden-state decay does not produce superior out-of-fold generalization compared to continuous temporal latent flow (CTCLM) on outpatient visits sampled 6–12 months apart.

### 2.3 Experimental Specification
- **Component Evaluated**: Temporal sequence encoder in `src/nadiq/models/temporal/`.
- **Implementation**: Adapt `GRUDCell` as a lightweight module (~150 lines of pure PyTorch):
  $$\gamma_x = \exp\left(-\text{ReLU}(W_\gamma \delta_t + b_\gamma)\right), \quad \gamma_h = \exp\left(-\text{ReLU}(W_h \delta_t + b_h)\right)$$
  Input: Longitudinal UPDRS Part III scores at Baseline and Month 12, with observation mask $m$ and time lag $\delta_t$ in years.
- **Matched Controls**:
  1. `M6_CTCLM_Baseline` (Continuous latent ODE flow).
  2. `Standard_GRU` (Ablation with zero temporal decay).
  3. `M0_Logistic_Regression` (Linear delta baseline).
- **Cross-Validation Protocol**:
  - Exact 3 seeds $[42, 123, 456] \times 5$ folds outer Stratified CV (15 splits).
  - Identical participant test folds matching Phase 14A.
  - Zero test leakage: feature standardizers and imputation medians fitted strictly on $X_{\text{train}}$.
- **Metrics Recorded**:
  - AUROC, AUPRC, Brier Score, Expected Calibration Error (ECE).
  - Paired clustered bootstrap 95% CI (2,000 resamples) for $\Delta = \text{AUROC}(\text{GRU-D}) - \text{AUROC}(\text{CTCLM})$.
- **Explicit Stop Condition**:
  - If $\text{AUROC}(\text{GRU-D}) < 0.6131$ or if the 95% CI for difference spans zero, **REJECT** GRU-D and retain CTCLM as the official temporal motor module.

---

## 3. Experiment 2: Biological Hypergraph Representation — DHG-based HNHN vs Classical Relational Baselines

### 3.1 Research Objective
Determine whether hypergraph convolution networks (HNHN / UniGCN from `DeepHypergraph`) operating on evidence-grown MeTTa biological hyperedges improve multimodal progression prediction over flat classical relational features ($M_4$) and linear baselines ($M_0$).

### 3.2 Falsifiable Hypothesis
$$\mathcal{H}_2: \text{AUROC}(\text{HNHN}_{\text{bio}}) - \text{AUROC}(M_0) > 0 \quad \text{with 95% Bootstrap CI strictly }> 0$$
*Null Hypothesis $\mathcal{H}_0$*: On a low-dimensional 6-node multimodal graph ($|V|=6$, molecular, structural, motor), deep hypergraph message passing overfits small cohorts ($N=341$) and fails to outperform regularized linear or shallow relational models.

### 3.3 Experimental Specification
- **Component Evaluated**: Hypergraph representation branch in `src/nadiq/models/hypergraph/`.
- **Implementation**:
  - Use `dhg.Hypergraph` incidence matrix structure $H \in \mathbb{R}^{6 \times 8}$.
  - Incidence edges derived strictly inside each training fold using MeTTa correlation filters ($r \ge 0.05, p \le 0.05$).
  - Two-layer HNHN model (`dhg.models.HNHN`):
    $$x_e = \sigma(W_e \cdot \text{Aggr}_{v \in e}(x_v)), \quad x_v = \sigma(W_v \cdot \text{Aggr}_{e \ni v}(x_e))$$
    Readout: Readout pooling across node embeddings $\to$ classification head.
- **Matched Controls**:
  1. `M0_Logistic_Regression` (AUROC 0.6366).
  2. `M4_Classical_Relational_Classifier` (Linear model on explicit $[\boldsymbol{\theta}, \mathbf{r}]$ features, AUROC 0.5935).
  3. `M1_RBF_SVM` (Classical RBF-SVM, AUROC 0.5241).
  4. `HNHN_Shuffled_Topology` (Negative control with permuted incidence matrix $H$).
- **Cross-Validation Protocol**:
  - Strict fold-contained graph discovery: MeTTa correlation discovery and edge promotion executed strictly on the training partition of each fold.
  - 3 seeds $\times$ 5 folds outer Stratified CV.
- **Metrics Recorded**:
  - AUROC, AUPRC, Brier Score, ECE, runtime, and GPU/CPU memory footprint.
  - Paired clustered bootstrap 95% CI vs $M_0$ and vs $M_4$.
- **Explicit Stop Condition**:
  - If $\text{AUROC}(\text{HNHN}_{\text{bio}}) \le 0.6366$ (fails to beat simple Logistic Regression), **STOP** hypergraph neural network expansion. Retain classical linear/logistic relational integration for production.

---

## 4. Experiment 3: Decoupled Quantum Simulation Benchmark — Geometric Manifold Kernel (Isolated Research Sandbox)

### 4.1 Research Objective
Provide a scientifically rigorous explanation for *why* quantum relational kernels collapsed on PPMI clinical progression (Phase 14A AUROC 0.4158), and empirically delineate the geometric boundaries where quantum kernels can vs cannot preserve distance metrics.

### 4.2 Falsifiable Hypothesis
$$\mathcal{H}_3: \text{On continuous monotonic clinical trajectories, quantum kernel fidelity } |\langle \psi(x_i) | \psi(x_j) \rangle|^2 \text{ decays exponentially with phase term variance, whereas on synthetic discrete cyclic manifolds it preserves margin separation.}$$

### 4.3 Experimental Specification
- **Component Evaluated**: Quantum feature map and fidelity kernel in `src/nadiq/quantum/`.
- **Strict Constraint**: **Never executed on restricted PPMI patient data.** Purely synthetic geometric benchmark.
- **Synthetic Datasets**:
  1. **Continuous Monotonic Drift** (mimicking clinical disease progression): 1D continuous latent trajectory + Gaussian noise.
  2. **Discrete Cyclic Manifold**: Discrete $n$-dimensional hypercube with $\mathbb{Z}_2^{\otimes n}$ parity symmetry (where quantum phase operations match group structure).
- **Implementation**:
  - PennyLane `default.qubit` simulator.
  - Measure kernel Gram matrix condition number, off-diagonal fidelity distribution, and eigenvalue decay spectrum across varying phase scales $s \in [0.1, 1.0, \pi, 2\pi]$.
- **Expected Scientific Deliverable**:
  - Formal mathematical proof and empirical demonstration that clinical biomarker trajectories lack the algebraic group structure required for quantum kernel advantage.
  - Publishable negative-result paper artifact supporting our architectural pivot.

---

## 5. Experiment Execution Governance

```
                    +--------------------------------+
                    | Controlled Experiment Roadmap  |
                    +--------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
[ Experiment 1: GRU-D vs CTCLM ]            [ Experiment 2: DHG HNHN vs Linear ]
- Task T2 landmark progression              - Fold-contained MeTTa discovery
- Paired bootstrap vs CTCLM & M0            - Paired bootstrap vs M0 (0.6366)
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v
             [ Gate Check: Superiority Threshold ]
             - Must exceed AUROC 0.6366 with 95% CI > 0
             - If NO: Freeze M0 + CTCLM as Production
             - If YES: Promote to Production Branch
                                   |
                                   v
             [ Experiment 3: Decoupled QML Sandbox ]
             - Isolated synthetic manifold analysis
             - Formalize negative quantum findings for publication
```
