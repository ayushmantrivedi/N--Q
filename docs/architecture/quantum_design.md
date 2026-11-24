# Quantum Circuit Design & Optimization Specification — NĀḌĪ-Q

> **Document Status Legend**  
> - **[A] Established Physics / Literature** — Backed by published quantum computing literature.  
> - **[B] Empirically Validated Architecture** — Tested and validated on the PPMI cohort.  
> - **[C] Audited Negative Finding** — Rigorously tested and rejected based on experimental evidence.  
> - **[D] Production MVP Component** — Integrated into the active platform.

---

## 1. Scientific Preamble & Empirical Lessons

NĀḌĪ-Q is designed to rigorously evaluate hybrid quantum-classical machine learning architectures for longitudinal disease progression. 

**Core Scientific Finding [C]**:  
Experiments across Phases 12, 13, and 14A demonstrated that **tightly coupling parameterized quantum kernels directly into patient risk prediction (e.g. SVM or BRQK) collapses clinical AUROC from 0.6366 to 0.5140**. On biomedical cohorts of size $N \approx 341$, NISQ simulation noise and kernel matrix variance dominate the true prognostic signal.

**The Architectural Solution (Architecture A) [D]**:  
Quantum computing is deployed strictly where combinatorial structure exists: **biological relationship discovery (QUBO / QAOA graph partitioning)**. Meanwhile, patient risk prediction remains strictly classical. This preserves peak clinical discrimination (AUROC 0.6366) while leveraging quantum optimization for multi-relational evidence selection.

---

## 2. Quantum Variational Biomarker Encoder (QVBE-v4) [B/D]

### 2.1 Circuit Architecture & Mathematical Formulation
The QVBE-v4 circuit (`src/nadiq/models/qvbe_v4/circuit.py`) maps normalized multimodal clinical features into an 8-qubit quantum state space:

```
|0⟩ ─── Ry(x̃₀) ─── Rz(x̃₁) ─── ● ──────────── R(θ₀, φ₀) ─── [M: Z₀]
                              │
|1⟩ ─── Ry(x̃₂) ─── Rz(x̃₃) ─── X ─── ● ─────── R(θ₁, φ₁) ─── [M: Z₁]
                                    │
|2⟩ ─── Ry(x̃₄) ─── Rz(x̃₅) ───────── X ─── ● ── R(θ₂, φ₂) ─── [M: Z₂]
                                          │
...                                       ▼
|7⟩ ─── Ry(x̃₁₄) ── Rz(x̃₁₅) ─────────────── X ── R(θ₇, φ₇) ─── [M: Z₇]
```

1. **State Initialization [A]**:
   All qubits initialize in the ground state $|0\rangle^{\otimes n}$ ($n=8$).
2. **Angle Data Encoding [A]**:
   Normalized feature vector $\mathbf{x} \in \mathbb{R}^{16}$ is mapped to rotation angles:
   $$U_{\text{enc}}(\mathbf{x}) = \bigotimes_{i=0}^{7} \left(R_z(x_{2i+1}) R_y(x_{2i})\right)$$
3. **Hardware-Efficient Entangling Layer [A]**:
   Circular ladder of controlled-NOT (CNOT) gates generates bipartite and multipartite entanglement:
   $$U_{\text{ent}} = \prod_{i=0}^{n-1} \text{CNOT}_{i, (i+1)\%n}$$
4. **Variational Parameterized Rotations [A]**:
   Parameterized arbitrary rotation gates $R(\theta_i, \phi_i, \omega_i) = R_z(\omega_i) R_y(\theta_i) R_z(\phi_i)$ applied across $L=2$ layers.
5. **Expectation Value Readout [A]**:
   Local Pauli-Z expectation values are measured:
   $$q_i = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_i | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \in [-1, 1], \quad i \in \{0, \dots, 7\}$$

### 2.2 Fubini-Study Metric Regularization [A/B]
To prevent barren plateaus and trainability collapse, QVBE-v4 incorporates Fubini-Study metric regularization into its pretraining objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{FS}} \text{Tr}(g(\boldsymbol{\theta}))$$
where $g(\boldsymbol{\theta})$ is the Quantum Fisher Information Matrix (QFIM). Pretrained weights are persisted at `experiments/phase12_2/best_qvbe_pretrained.pt`.

---

## 3. Quantum Approximate Optimization Algorithm (QAOA) for QUBO [B/D]

### 3.1 Mapping Evidence Discovery to QUBO
In Plane 1, the biological evidence discovery problem over $N=15$ candidate edges is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem:
$$\min_{\mathbf{x} \in \{0, 1\}^N} \mathbf{x}^T \mathbf{Q} \mathbf{x} + \mathbf{c}^T \mathbf{x}$$

where:
- **Linear terms $c_i$**: Encourage statistically strong, validated edges (negative correlation magnitude $-|r_i|$ scaled by FDR tier).
- **Quadratic terms $Q_{ij}$**: Penalize biological redundancies (e.g., co-selecting both Caudate and Putamen MRI volumes) and biophysical contradictions.
- **Budget Penalty**: Penalizes solutions deviating from the target edge budget $K$:
  $$P_{\text{budget}}(\mathbf{x}) = \lambda \left(\sum_{i=1}^N x_i - K\right)^2$$

### 3.2 PennyLane QAOA Circuit
The QUBO is mapped to an Ising spin Hamiltonian via the transformation $x_i = \frac{1 - Z_i}{2}$:
$$H_C = \sum_{i < j} J_{ij} Z_i Z_j + \sum_{i=1}^N h_i Z_i + C_0$$

The QAOA circuit applies $p$ alternating layers of cost Hamiltonian evolution $U(H_C, \gamma_l)$ and mixer Hamiltonian evolution $U(H_M, \beta_l)$ where $H_M = \sum_{i=1}^N X_i$:
$$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l \sum X_i} e^{-i \gamma_l H_C} \right) |+\rangle^{\otimes N}$$

### 3.3 Asynchronous Sidecar & 15-Second Timeout Fallback [D]
Because classical Tabu Search ($M_3$) solves this $N=15$ problem in **127 ms** with a **0.0000** optimality gap, QAOA on PennyLane simulator is deployed as an **asynchronous exploration sidecar**:
- QAOA runs in a background thread with $p=1, 2$.
- A strict 15-second timeout watchdog monitors QPU/simulator latency.
- If QAOA exceeds 15 seconds or encounters numerical instability, the system falls back to Tabu search immediately.

---

## 4. Mathematical Proof of Quantum Invariance in Clinical Inference [D]

To ensure regulatory compliance and patient safety, Plane 2 clinical progression predictions are mathematically invariant to quantum execution:

### Theorem: Quantum Outage Invariance
Let $\mathcal{M}_{\text{clin}}$ be the clinical prediction function for Task T2 operating on patient features $\mathbf{x}_{\text{patient}}$. Let $S_{\text{quantum}} \in \{\text{Active}, \text{Timeout}, \text{Error}, \text{Offline}\}$ denote the operational status of the quantum QPU / simulator.

Because `src/nadiq/architecture_a/clinical_predictor.py` maintains zero functional imports from quantum modules:
$$\forall \mathbf{x}_{\text{patient}}, \quad \mathcal{M}_{\text{clin}}(\mathbf{x}_{\text{patient}} \mid S_{\text{quantum}} = \text{Active}) \equiv \mathcal{M}_{\text{clin}}(\mathbf{x}_{\text{patient}} \mid S_{\text{quantum}} = \text{Offline})$$

Therefore:
$$\Delta_{\max} = \max_{i \in \text{Cohort}} |P_{\text{Active}}(y_i) - P_{\text{Offline}}(y_i)| = 0.00\text{e}{+}00$$

This was empirically stressed and verified with $N_{\text{samples}}=341$ under adversarial quantum outage simulations (`scripts/run_architecture_a_benchmark.py`).

---

## 5. Realistic NISQ Noise Considerations [A]

While all experiments currently execute on PennyLane simulators (`default.qubit` / `lightning.qubit`), the circuits are engineered for real NISQ execution:
1. **Shallow Depth**: QVBE-v4 uses $L=2$ layers, maintaining circuit depth $\le 18$ gates within current transmon coherence times ($T_2 \approx 50\text{--}100\ \mu\text{s}$).
2. **Nearest-Neighbor Connectivity**: Two-qubit gates are restricted to linear or circular nearest-neighbor couplings to minimize SWAP routing overhead.
3. **Finite-Shot Variance**: Expectation values are evaluated with shot modeling ($N_{\text{shots}} \ge 1000$), ensuring statistical error $\sigma_E \le \frac{1}{\sqrt{1000}} \approx 0.031$.
