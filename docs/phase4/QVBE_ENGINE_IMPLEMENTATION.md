# NĀḌĪ-Q — QVBE v2 Engine Implementation & Architecture Specification
**Document Version**: 2.0.0  
**Phase**: Phase 4 — Quantum Variational Feature Extraction & Representation Learning  
**Module Path**: `src/nadiq/models/quantum/qvbe_v2.py`  
**Execution Environment**: Python 3.12.4, PennyLane 0.41.1, PyTorch, Scikit-Learn 1.6.1  
**Status**: APPROVED & VERIFIED  

---

## 1. Executive Summary & Paradigm Realignment

The **Quantum Variational Biomarker Encoder v2 (QVBE-v2)** serves as NĀḌĪ-Q's central representation-learning engine. Departing fundamentally from the Phase 3 black-box projected quantum kernel (PQK) architecture—which suffered from severe Gram matrix ill-conditioning (mean off-diagonal similarity $\approx 0.957$, condition number $> 10^{14}$)—QVBE-v2 operates as an explicit, high-dimensional non-linear feature extractor.

QVBE-v2 transforms continuous classical representations (tabular clinical/demographic indices, FreeSurfer morphometry, genetic PRSs, and pretrained latent embeddings) into an explicit, well-conditioned tabular feature matrix:
$$Q \in \mathbb{R}^{N \times M}$$
where $M$ denotes the number of physical multi-basis quantum expectation observables ($M = 23$ for 8-qubit circuits; $M = 35$ for 12-qubit circuits).

Downstream machine learning models (LightGBM, XGBoost, ElasticNet Logistic Regression) operate directly on $Q$ or fused representations $[X \parallel Q]$, decoupling quantum state preparation from downstream classification dynamics.

---

## 2. Mathematical Circuit Architecture

The parameterized quantum circuit $\mathcal{U}(\mathbf{x}; \boldsymbol{\theta})$ comprises five consecutive physical stages:

```
|0⟩ ─── RY(x_0) ─── H ─── U3(θ_0) ─── ● ─────────╭─ / ─╮ ─── ControlledPhase(α_0) ─── [Z, X, ZZ Readout]
                                      │         │     │
|0⟩ ─── RY(x_1) ─── H ─── U3(θ_1) ─── X ─── ● ───│  Q  │ ─── ControlledPhase(α_1) ─── [Z, X, ZZ Readout]
                                            │   │  A  │
|0⟩ ─── RY(x_2) ─── H ─── U3(θ_2) ───────── X ───│  F  │ ─── ControlledPhase(α_2) ─── [Z, X, ZZ Readout]
                                                ╰─ / ─╯
```

### Stage 1: Input Scaling & Non-Linear Angle Encoding
For input vector $\mathbf{x} \in \mathbb{R}^D$:
1. If $D > n$, a deterministic projection (PCA fitted on training folds) reduces dimensionality to $n$ qubits.
2. Standard scaling normalizes features to zero-mean and unit-variance.
3. MinMax scaling maps coordinates strictly into rotation angles $\phi_i \in [0, \pi]$:
$$\phi_i = \pi \cdot \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$$
4. Rotation operators prepare the initial product state:
$$|\psi_0(\mathbf{x})\rangle = \bigotimes_{i=0}^{n-1} R_y(\phi_i) |0\rangle^{\otimes n} = \bigotimes_{i=0}^{n-1} \left( \cos\frac{\phi_i}{2}|0\rangle + \sin\frac{\phi_i}{2}|1\rangle \right)$$

### Stage 2: Superpositional Initialization
To prevent localized product state traps and ensure broad Hilbert space exploration:
$$|\psi_1(\mathbf{x})\rangle = H^{\otimes n} |\psi_0(\mathbf{x})\rangle$$

### Stage 3: Parameterized Cyclic $U_3$ Unitaries
Each qubit undergoes arbitrary $SU(2)$ single-qubit rotations via 3-parameter unitaries across $L$ layers:
$$U_3(\theta, \phi, \lambda) = \begin{pmatrix} \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2} \end{pmatrix}$$
This contributes $3 \times n \times L$ variational degrees of freedom.

### Stage 4: Cyclic CNOT Entanglement Ladder
Multi-qubit non-local correlations are generated through a cyclic entanglement ring:
$$U_{\text{ent}} = \prod_{i=0}^{n-1} \text{CNOT}_{(i, (i+1) \bmod n)}$$
This cyclic topology ensures boundary-free periodic connectivity, enabling non-local feature interactions across all input dimensions.

### Stage 5: Hierarchical Quantum Attention-Fourier (QAF) Layer
To capture multi-scale cross-frequency dependencies (as specified in the SIH 2026 SRS), controlled phase rotations are applied across dyadic strides $s \in \{0, 1, \dots, \lfloor \log_2 n \rfloor - 1\}$:
$$U_{\text{QAF}} = \prod_{s=0}^{S-1} \prod_{i=0}^{n-1} \text{CPHASE}\left(\alpha_{s, i}, \text{wires}=[i, (i + 2^s) \bmod n]\right)$$
where:
$$\text{CPHASE}(\alpha) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & e^{i\alpha} \end{pmatrix}$$

---

## 3. Configuration Class Specification (`QVBEV2Config`)

The engine configuration is fully declared as a Python dataclass:

```python
@dataclass
class QVBEV2Config:
    input_dim: int
    n_qubits: int = 8
    n_layers: int = 2
    use_u3: bool = True
    use_hadamard: bool = True
    use_qaf: bool = True
    measurement_basis: str = "full"  # 'z_only', 'dual_basis', 'full'
    shots: Optional[int] = None      # None = analytic statevector; int = finite-shot
    backend: str = "default.qubit"
    seed: int = 42
    mode: str = "fixed"              # 'fixed' (Mode A) or 'trainable' (Mode B)
    learning_rate: float = 0.05
    n_iter: int = 20
```

### Parameter Budgets

| Configuration | Qubits ($n$) | Layers ($L$) | $U_3$ Parameters ($3nL$) | QAF Parameters ($n \cdot \lfloor\log_2 n\rfloor$) | Total Circuit Parameters | Output Dimension ($M$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **8-Qubit Z-only** | 8 | 2 | 48 | 24 | 72 | 8 |
| **8-Qubit Dual-Basis** | 8 | 2 | 48 | 24 | 72 | 16 |
| **8-Qubit Full (Standard)** | 8 | 2 | 48 | 24 | 72 | **23** |
| **12-Qubit Full (Expanded)** | 12 | 2 | 72 | 36 | 108 | **35** |

---

## 4. Class Interface & Lifecycle (`QuantumVariationalBiomarkerEncoderV2`)

The implementation strictly satisfies Scikit-Learn's `BaseEstimator` and `TransformerMixin` protocols:

### Primary Methods
1. `fit(X, y=None)`:
   - Validates that the instance is not frozen.
   - Fits `StandardScaler` and `MinMaxScaler` on training features $X_{\text{train}}$.
   - Fits PCA projection if $D > n$.
   - **Mode A (Fixed)**: Computes two-stage convex warm-start on Layer 0 using linear discriminant weights:
     $$\theta_{0, i, 0} \leftarrow \text{clip}\left(\beta_i^{\text{Ridge}} \cdot \pi, -\pi, \pi\right)$$
   - **Mode B (Trainable)**: Refines parameters using contrastive loss via the parameter-shift rule over mini-batches.
   - Marks `is_fitted = True`.

2. `transform(X)`:
   - Applies fitted scalers and projections to input data.
   - Executes the quantum circuit for each sample.
   - Returns feature matrix $Q \in \mathbb{R}^{N \times M}$ where all values are physical expectation values in $[-1.0, 1.0]$.

3. `fit_transform(X, y=None)`:
   - Sequential invocation of `fit` followed by `transform`.

4. `freeze()` / `unfreeze()`:
   - Sets `is_frozen = True`. Prevents accidental parameter re-fitting on validation or test partitions.
   - Calling `fit()` on a frozen instance raises a strict `RuntimeError`.

5. `save(filepath)` / `load(filepath)`:
   - Serializes complete configuration, variational parameter tensors, scalers, and PCA projection matrices to deterministic JSON/NPZ.
   - Load restores identical mathematical states with zero floating-point drift ($\Delta < 10^{-14}$).

6. `get_diagnostics(Q)`:
   - Computes covariance matrix $\Sigma_Q$, eigenvalues, effective spectral rank:
     $$\text{eff\_rank}(Q) = \exp\left(-\sum p_i \ln p_i\right), \quad p_i = \frac{\lambda_i}{\sum \lambda}$$
   - Condition number $\kappa = \lambda_{\max} / \lambda_{\min}$.
   - Pairwise off-diagonal cosine similarities to monitor expressivity vs. dimensional collapse.

7. `get_state_entropy(x)`:
   - Intrinsic Quantum Explainable AI (Q-XAI).
   - Reconstructs 1-qubit reduced density matrix $\rho_0 = \text{Tr}_{\bar{0}}(|\psi\rangle\langle\psi|)$.
   - Computes subsystem von Neumann entropy:
     $$S(\rho_0) = -\text{Tr}(\rho_0 \ln \rho_0) = -\sum_{k} \lambda_k \ln \lambda_k$$
   - Measures the degree of quantum entanglement shared between the target feature and the rest of the biomarker register ($S \in [0, \ln 2]$).

---

## 5. Verification & Test Coverage

The engine is rigorously validated by the dedicated scientific test suite `tests/scientific/test_qvbe_v2.py`:
- 11 formal unit test cases verifying shape correctness, parameter bounds, determinism, freeze semantics, serialization fidelity, contrastive tuning, Q-XAI entropy extraction, finite-shot simulation, and diagnostic profiling.
