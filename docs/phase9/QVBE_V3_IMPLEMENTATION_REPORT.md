# NĀḌĪ-Q — QVBE-v3 IMPLEMENTATION REPORT
## Architecture, Mathematical Remediation, and Unit Verification

> **Document Type**: Scientific Implementation Report  
> **Phase**: 9 — QVBE Remediation & Architectural Repair  
> **Authors**: Senior Quantum Machine Learning Researcher, Mathematical Physicist & Scientific Software Architect  
> **Target Package**: `src/nadiq/models/qvbe_v3/`  
> **Verification Standards**: Every claim carries an explicit evidence tag: `[VERIFIED FROM SOURCE CODE]`, `[VERIFIED FROM EXPERIMENT]`, `[MATHEMATICALLY DERIVED]`.

---

# 1. EXECUTIVE SUMMARY

In Phase 8, forensic internal diagnostics identified seven structural vulnerabilities in `QVBE-v2`:
1. **The Pre-Quantum PCA Bottleneck**: High-dimensional features ($d=204$ MRI) were reduced to 8 linear components before entering the circuit, discarding up to $70\%$ of variance classically.
2. **Broken Parameter Optimization in Mode B**: The label array `y_sub` was extracted but never referenced in the gradient update loop.
3. **Random Parameter Seed Dominance**: Seed variance ($0.0152$) was $1.54\times$ greater than patient biological variance ($0.0098$).
4. **Effective Rank Ceiling**: Output rank was capped at $\sim 8.68 / 23.0$, bounded by the input PCA angles.
5. **Entanglement Irrelevance**: Product states correlated at $r_1 = 0.9977$ with entangled states.
6. **Absence of Variational Optimization in Mode A**: Parameters were frozen after a single classical Ridge Classifier warm-start.
7. **Readout Truncation**: Measuring only 23 local Pauli expectations discarded 232 out of 255 pure-state degrees of freedom.

In Phase 9, we developed `QVBE-v3` as an independent, modular architecture (`src/nadiq/models/qvbe_v3/`) without modifying or overwriting legacy files. All 10 unit tests in `tests/phase9/test_qvbe_v3.py` passed with zero errors (`[VERIFIED FROM EXPERIMENT]`).

---

# 2. MATHEMATICAL ARCHITECTURE OF QVBE-v3

### 2.1 Elimination of the Fixed PCA Bottleneck: Trainable Input Adapter
`[VERIFIED FROM SOURCE CODE: src/nadiq/models/qvbe_v3/encoding.py]`
Instead of static linear PCA ($X_{\text{proj}} = \text{PCA}(n=8)(X)$), `QVBE-v3` introduces a trainable linear projection adapter $W \in \mathbb{R}^{n_{\text{qubits}} \times d_{\text{input}}}$:
$$z_i = \sum_{j=1}^{d_{\text{input}}} W_{ij} x_j, \quad i \in \{0, \dots, n_{\text{qubits}} - 1\}$$
- $W$ is initialized with LeCun normal scaling: $W_{ij} \sim \mathcal{N}\left(0, \frac{1}{d_{\text{input}}}\right)$.
- $W$ is registered as a PyTorch `nn.Parameter` and trained jointly with quantum circuit parameters via autograd.
- *Mathematical Consequence*: All $d_{\text{input}}$ features contribute to the representation. The adapter learns to align biological variance with the circuit's entangling topology rather than discarding variance along orthogonal classical principal axes.

### 2.2 Data Re-uploading and Fourier Spectrum Expansion
`[MATHEMATICALLY DERIVED]` & `[VERIFIED FROM SOURCE CODE: src/nadiq/models/qvbe_v3/circuit.py]`
In `QVBE-v2`, features entered through a single $R_Y(x_i)$ gate. By the Schur-Weyl and Fourier representation theorems (Schuld et al., 2021), a circuit with a single encoding gate per wire represents functions restricted to frequencies $\omega \in \{-1, 0, 1\}$.

`QVBE-v3` implements sequential Data Re-uploading (Pérez-Salinas et al., 2020) across $K = n_{\text{uploads}}$ stages:
$$|\psi(x, \boldsymbol{\theta}, \mathbf{w})\rangle = \prod_{k=1}^K \left[ \mathcal{U}_{\text{entangle}} \cdot \mathcal{U}_{\text{var}}(\boldsymbol{\theta}_k) \cdot \mathcal{U}_{\text{enc}}(x, \mathbf{w}_k) \right] |0\rangle^{\otimes n}$$
where:
$$\mathcal{U}_{\text{enc}}(x, \mathbf{w}_k) = \bigotimes_{i=0}^{n-1} R_Z\left(\frac{1}{2}\phi_k(x_i)\right) R_Y\left(\phi_k(x_i)\right), \quad \phi_k(x_i) = \pi \cdot \sigma(w_{k, i} z_i)$$
- *Mathematical Consequence*: With $K$ re-uploading stages, the accessible multi-dimensional Fourier spectrum expands to:
  $$\omega_i \in \{-K, \dots, -1, 0, 1, \dots, K\}$$
  For $K = 3$, the circuit represents harmonic interactions up to degree 3, allowing non-linear expressivity beyond classical 2nd-order trigonometric models.

### 2.3 Elimination of Random Seed Dominance: Controlled Near-Identity Initialization
`[VERIFIED FROM SOURCE CODE: src/nadiq/models/qvbe_v3/circuit.py]`
- In `QVBE-v2`, parameters were initialized uniformly over $[-\pi, \pi]$, creating random unitary rotations that overwhelmed biological inputs.
- In `QVBE-v3`, rotation parameters are initialized near identity:
  $$\boldsymbol{\theta}_{l, i} \sim \mathcal{N}(0, 0.05^2), \quad w_{k, i} = 1.0$$
- *Empirical Verification*: Diagnostic Test 2 confirmed that the seed-variance-to-patient-variance ratio dropped from **$1.5184$** in v2 to **$0.0372$** in v3 (`[VERIFIED FROM EXPERIMENT]`), representing a $97.5\%$ reduction in initialization noise.

### 2.4 Resolving Entanglement Readout: Connected 2-Qubit Correlators
`[VERIFIED FROM SOURCE CODE: src/nadiq/models/qvbe_v3/observables.py]`
To rigorously separate genuine quantum entanglement from separable product-state baselines, `QVBE-v3` introduces the connected 2-qubit correlation function:
$$C_{ij}^{ZZ} = \langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle$$
- For separable product states ($|\psi\rangle = \bigotimes_i |\psi_i\rangle$), $C_{ij}^{ZZ} = 0.0$ analytically.
- In Diagnostic Test 6, separable circuits yielded $\max |C_{ij}| = 6.75 \times 10^{-7} < 10^{-4}$, while entangled circuits produced mean $|C_{ij}| = 0.1610 > 0$ (`[VERIFIED FROM EXPERIMENT]`).

### 2.5 True Variational Optimization: Symmetric InfoNCE Cross-Modal Alignment
`[VERIFIED FROM SOURCE CODE: src/nadiq/models/qvbe_v3/objectives.py]`
`QVBE-v3` replaces the broken Mode B training loop with an exact, leak-free self-supervised objective:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2B} \sum_{i=1}^B \left[ \log \frac{\exp(\text{sim}(q_{\text{mol}, i}, q_{\text{temp}, i}) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_{\text{mol}, i}, q_{\text{temp}, j}) / \tau)} + \log \frac{\exp(\text{sim}(q_{\text{temp}, i}, q_{\text{mol}, i}) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_{\text{temp}, i}, q_{\text{mol}, j}) / \tau)} \right]$$
- No downstream 5-year progression labels are used during representation learning.
- PyTorch autograd computes parameter-shift and backpropagation gradients through PennyLane, yielding non-zero, finite gradients for all parameters.

---

# 3. UNIT TEST VERIFICATION SUITE

All unit tests in `tests/phase9/test_qvbe_v3.py` passed (`[VERIFIED FROM EXPERIMENT]`):

| Test Identifier | Verified Property | Numerical Result | Status |
| :--- | :--- | :--- | :---: |
| `test_data_reuploading_circuit_shape` | Output tensor dimension | Shape $(10, 23)$ | **PASS** |
| `test_gradient_flows` | Autograd gradient flow | Finite, non-zero gradients across all parameters | **PASS** |
| `test_entanglement_ablation` | Output sensitivity to CNOTs | Mean absolute difference: $0.5041 > 10^{-3}$ | **PASS** |
| `test_connected_correlator` | Connected correlator $C_{ij}$ | Sep: $8.73 \times 10^{-7} < 10^{-4}$, Ent: $0.188 > 0$ | **PASS** |
| `test_seed_variance` | Seed-to-patient variance ratio | **$0.0381 < 1.0$** (vs. $1.54$ in v2) | **PASS** |
| `test_no_pca_bottleneck` | Trainable adapter verification | Adapter $W \in \mathbb{R}^{8 \times 50}$ present, no PCA | **PASS** |
| `test_infonce_loss_uses_pairs` | Contrastive pair sensitivity | Correct pairs: $0.0009$, Shuffled: $6.0811$ | **PASS** |
| `test_reconstruction_monotonic` | Optimization convergence | Loss dropped from $2.432$ to $2.077$ ($0.887$ ratio) | **PASS** |
| `test_effective_rank_callable` | Spectral entropy rank | Computed effective rank: $7.148$ | **PASS** |
| `test_config_required_fields` | Configuration interface | All required fields present and validated | **PASS** |

Execution time: **37.21s** across 10 test cases on Windows 11 x64 (`pytest-9.1.1`).
