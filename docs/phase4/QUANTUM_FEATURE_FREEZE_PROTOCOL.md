# NĀḌĪ-Q — Quantum Feature Freeze Protocol & Fold Isolation Contract
**Document Version**: 2.0.0  
**Phase**: Phase 4 — Quantum Variational Feature Extraction & Representation Learning  
**Status**: APPROVED & MANDATORY ENFORCEMENT  

---

## 1. Scientific Rationale & Imperative

In hybrid quantum-classical machine learning, subtle information leakage can easily produce spurious "quantum advantage" if quantum transformations inadvertently access global statistics, test partition distributions, or target labels. 

To eliminate all forms of data contamination, NĀḌĪ-Q establishes the **Quantum Feature Freeze Protocol**:
1. All feature scaling, dimensionality projection, variational angle calibration, and quantum parameter optimizations occur strictly and solely on designated training partitions.
2. Once fitted on training data, the quantum feature extractor is explicitly **frozen**.
3. Feature matrices are extracted **offline** prior to downstream machine learning model training.
4. Downstream classical models are trained strictly on frozen representations, guaranteeing zero gradient backpropagation into the quantum circuit across partition boundaries.

---

## 2. Modes of Operation: Mode A vs. Mode B

QVBE-v2 defines two distinct operational modes:

### Mode A: Fixed Multi-Basis Quantum Feature Extractor
- **Variational Parameters**: $\boldsymbol{\theta}$ initialized deterministically from seed (or two-stage convex warm-start via linear discriminant alignment on $X_{\text{train}}$).
- **Gradient Tuning**: None ($\eta = 0$, $N_{\text{iter}} = 0$).
- **Role**: Acts as a fixed, non-linear geometric Hilbert space projection. Evaluates whether the intrinsic topological entanglement and multi-basis observables ($\langle Z \rangle, \langle X \rangle, \langle ZZ \rangle$) provide useful inductive biases for downstream tabular learners.
- **Computational Cost**: $O(N \cdot M)$ quantum circuit evaluations per fold; zero gradient overhead.

### Mode B: Supervised Trainable Representation Learning Engine
- **Initialization**: Stage 1 Convex warm-start initializes Layer 0 $U_3$ rotation angles proportional to Ridge discriminant coefficients computed on $X_{\text{train}}$:
$$\theta_{0, i, 0} = \text{clip}\left(\beta_i \cdot \pi, -\pi, \pi\right)$$
- **Optimization**: Stage 2 Parameter-shift optimization tunes rotation angles on mini-batches from $X_{\text{train}}$ using a contrastive separation loss:
$$\mathcal{L}_{\text{contrastive}} = \mathbb{E}_{y_a = y_b} [\|Q_a - Q_b\|_2^2] - \mathbb{E}_{y_a \neq y_b} [\|Q_a - Q_b\|_2^2]$$
- **Freezing**: At the conclusion of training epochs, the optimized parameter tensors $\boldsymbol{\theta}^*$ and all preprocessing scalers are irreversibly frozen:
$$\text{QVBE}.\text{freeze}() \implies \text{is\_frozen} = \text{True}$$
- **Role**: Explores whether task-adapted quantum representations discover non-linear separability unattainable by linear projections.

---

## 3. Strict Fold-Isolation Contract (5 Folds $\times$ 5 Seeds = 25 Replicates)

For each cross-validation fold $k \in \{1, \dots, 25\}$:

```
[ Full PPMI Cohort (N = 1,465) ]
              │
              ├── Stratified Partitioning (Identical Phase 3 Splits)
              ▼
   ┌──────────────────────────────────────────────┬───────────────────────────────┐
   │ Training Fold k (N_train = 1,172)            │ Test Fold k (N_test = 293)    │
   └──────────────────────────────────────────────┴───────────────────────────────┘
              │                                                   │
   [ Fit Scaler & PCA ]                                           │ (Untouched)
   [ Fit Stage 1 Warm-Start ]                                     │
   [ Fit Stage 2 Contrastive (Mode B) ]                           │
              │                                                   │
   [ Execute QVBE.freeze() ] ───────────────────────────┐         │
              │                                         │         │
   [ Transform X_train -> Q_train ]                     │ [ Transform X_test -> Q_test ]
              │                                         │         │
   [ Train Downstream ML Models ]                       │         │
     - LightGBM / XGBoost / LogReg                      │         │
              │                                         │         │
              └────────────────────► [ Evaluate on Q_test / (X_test + Q_test) ]
```

### Inviolable Rules:
1. **Zero Out-of-Fold Contamination**: Neither $X_{\text{test}}$ nor $y_{\text{test}}$ may be inspected by `StandardScaler`, `MinMaxScaler`, PCA, or quantum parameter optimization.
2. **Offline Representation Export**: Quantum features $Q_{\text{train}}$ and $Q_{\text{test}}$ are materialized to disk or memory arrays before initializing any downstream classifier.
3. **Frozen Immutability Enforcement**: Any call to `fit()` on an instance where `is_frozen == True` immediately terminates execution with a `RuntimeError`.
4. **Exact Determinism**: Given the seed and training partition, parameter initialization, gradient updates, and feature extraction must reproduce identically down to machine precision.

---

## 4. Architectural Equivalence with Pretrained Classical Freezing

This freeze protocol exactly mirrors the standard machine learning paradigm used for foundation models and deep self-supervised encoders:

| Model Family | Pretraining / Fitting Phase | Freezing Point | Downstream Phase |
| :--- | :--- | :--- | :--- |
| **TabNet / FT-Transformer** | Self-supervised masked feature reconstruction on $X_{\text{train}}$ | Encoder weights frozen | Linear probe or GBDT on extracted latents |
| **3D ResNet-50 / $\beta$-VAE** | Supervised or VAE loss on 3D MRI scans | Backbone weights frozen | Downstream classifier on bottleneck code $z$ |
| **NĀḌĪ-Q QVBE-v2** | Warm-start & contrastive parameter-shift on $X_{\text{train}}$ | Quantum circuit parameters $\boldsymbol{\theta}$ frozen | Downstream GBDT / LogReg on expectation matrix $Q$ |

By decoupling representation learning from downstream classification, NĀḌĪ-Q provides a clean, rigorous, and leakage-safe testbed for evaluating the genuine utility of quantum variational embeddings.
