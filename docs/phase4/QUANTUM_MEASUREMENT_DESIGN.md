# NĀḌĪ-Q — Quantum Measurement Design Specification
**Document Version**: 2.0.0  
**Phase**: Phase 4 — Quantum Variational Feature Extraction & Representation Learning  
**Authoritative Reference**: SIH 2026 SRS & Phase 4 Architecture  
**Status**: APPROVED & ACTIVE  

---

## 1. Executive Overview

A critical defect identified in the Phase 3 Quantum Projected Kernel (PQK) benchmark was the collapse of continuous representations into an ill-conditioned Gram matrix, where pairwise inner products concentrated around $0.957 \pm 0.02$ with condition numbers exceeding $10^{14}$.

Phase 4 redesigns quantum measurement completely: rather than generating an $N \times N$ inner product matrix, the **Quantum Variational Biomarker Encoder v2 (QVBE-v2)** outputs an explicit tabular feature matrix:
$$Q \in \mathbb{R}^{N \times M}$$
where each column represents the expectation value of a physical, Hermitian observable evaluated over the parameterized variational state $|\psi(\mathbf{x}; \boldsymbol{\theta})\rangle$.

---

## 2. Multi-Basis Observable System

To maximize representational expressivity and extract both single-qubit polarizations and multi-qubit spatial correlations, QVBE-v2 implements a multi-basis physical readout across three distinct operator sets:

### 2.1. Longitudinal Pauli-$Z$ Observables (Computational Basis)
For each qubit $i \in \{0, 1, \dots, n - 1\}$:
$$O_i^{(Z)} = Z_i = I^{\otimes i} \otimes \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \otimes I^{\otimes (n - i - 1)}$$
$$\langle Z_i \rangle = \langle \psi(\mathbf{x}; \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}; \boldsymbol{\theta}) \rangle = P(q_i = 0) - P(q_i = 1) \in [-1, +1]$$
- **Physical Meaning**: Quantifies the computational basis polarization (population inversion) of qubit $i$.
- **Count**: $n$ features.

### 2.2. Transverse Pauli-$X$ Observables (Dual-Basis Superposition)
For each qubit $i \in \{0, 1, \dots, n - 1\}$:
$$O_i^{(X)} = X_i = I^{\otimes i} \otimes \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \otimes I^{\otimes (n - i - 1)}$$
$$\langle X_i \rangle = \langle \psi(\mathbf{x}; \boldsymbol{\theta}) | X_i | \psi(\mathbf{x}; \boldsymbol{\theta}) \rangle = P(q_i = |+\rangle) - P(q_i = |-\rangle) \in [-1, +1]$$
- **Physical Meaning**: Quantifies transverse quantum coherence and superposition phase alignment. Under dual-basis readout, single-qubit states are measured along non-commuting axes ($[Z, X] \neq 0$), yielding complementary geometric projections.
- **Count**: $n$ features.

### 2.3. Entanglement Correlators ($Z_i Z_{i+1}$)
For adjacent qubit pairs along the cyclic entanglement ring $i \in \{0, 1, \dots, n - 2\}$:
$$O_i^{(ZZ)} = Z_i \otimes Z_{i+1}$$
$$\langle Z_i Z_{i+1} \rangle = \langle \psi(\mathbf{x}; \boldsymbol{\theta}) | Z_i \otimes Z_{i+1} | \psi(\mathbf{x}; \boldsymbol{\theta}) \rangle \in [-1, +1]$$
- **Physical Meaning**: Quantifies two-point quantum correlations and mutual phase locking induced by the cyclic CNOT ladder and hierarchical QAF controlled-phase rotations.
- **Count**: $n - 1$ features.

---

## 3. Dimensionality Taxonomy

The total output feature dimension $M$ is governed by the selected measurement basis:

| Mode | Basis Specification | Mathematical Operators | Output Dim ($n=8$) | Output Dim ($n=12$) |
| :--- | :--- | :--- | :---: | :---: |
| **`z_only`** | Computational only | $\{\langle Z_i \rangle\}_{i=0}^{n-1}$ | $M = 8$ | $M = 12$ |
| **`dual_basis`** | Non-commuting bases | $\{\langle Z_i \rangle, \langle X_i \rangle\}_{i=0}^{n-1}$ | $M = 16$ | $M = 24$ |
| **`full` (Standard)** | Dual-basis + Correlators | $\{\langle Z_i \rangle, \langle X_i \rangle\}_{i=0}^{n-1} \cup \{\langle Z_i Z_{i+1} \rangle\}_{i=0}^{n-2}$ | **$M = 23$** | **$M = 35$** |

### Proof of Feature Boundedness
For any normalized pure quantum state $|\psi\rangle$ ($\langle\psi|\psi\rangle = 1$) and any Pauli string $P \in \{Z, X, ZZ\}$:
Since the eigenvalues of any Pauli matrix or tensor product of Pauli matrices are strictly $\pm 1$:
$$\|P\|_{\infty} = 1 \implies \langle\psi|P|\psi\rangle \in [-1, +1]$$
Every column of the QVBE-v2 feature matrix $Q_{:, j}$ is strictly bounded within $[-1.0, +1.0]$. Zero division, infinite values, or unbounded gradients are physically impossible.

---

## 4. Analytic Simulation vs. Finite-Shot NISQ Execution

### 4.1. Analytic Statevector Simulation (Default Benchmark Mode)
In analytic mode (`shots=None`), expectation values are calculated via direct matrix-vector contraction:
$$\langle O \rangle = \mathbf{v}^\dagger \mathcal{M}_O \mathbf{v}$$
where $\mathbf{v} \in \mathbb{C}^{2^n}$ is the simulated statevector and $\mathcal{M}_O$ is the sparse observable operator.
- **Advantage**: Zero sampling noise. The feature representation is deterministic and mathematically exact down to machine epsilon ($\sim 10^{-16}$).
- **Scientific Necessity**: Enables rigorous isolation of the quantum representation's geometric expressivity without stochastic noise confounding downstream ML model evaluations.

### 4.2. Finite-Shot Simulation (Hardware Deployment Mode)
On real NISQ quantum processors (or when `shots=S`), expectations are estimated as empirical sample means of projective measurement outcomes $b_s \in \{-1, +1\}$:
$$\hat{\langle O \rangle}_S = \frac{1}{S} \sum_{s=1}^S b_s$$

#### Shot Noise Variance & Confidence Intervals
By the Central Limit Theorem:
$$\mathbb{E}[\hat{\langle O \rangle}_S] = \langle O \rangle$$
$$\text{Var}[\hat{\langle O \rangle}_S] = \frac{1 - \langle O \rangle^2}{S} \le \frac{1}{S}$$
The standard error of measurement is strictly bounded by:
$$\sigma_{\text{shot}} \le \frac{1}{\sqrt{S}}$$

For typical experimental budgets:
- $S = 1,024 \implies \sigma_{\text{shot}} \le 0.03125$ ($3.1\%$)
- $S = 4,096 \implies \sigma_{\text{shot}} \le 0.01562$ ($1.6\%$)
- $S = 8,192 \implies \sigma_{\text{shot}} \le 0.01105$ ($1.1\%$)

QVBE-v2 natively supports seamless switching between analytic statevector evaluation and finite-shot simulation via `QVBEV2Config(shots=1024)`.
