# Quantum Design — Nāḍī-Q

> **Status markers:**  
> **[A]** Established · **[B]** Hypothesis · **[C]** Experimental Claim · **[D]** Future Architecture

---

## Preamble

This document describes the quantum circuit design rationale for Nāḍī-Q. It is written to be honest about what is established quantum computing knowledge and what is speculative hypothesis specific to this research.

**Nothing in this document constitutes a claim of demonstrated quantum advantage.**  
Quantum advantage for clinical tabular data is an open research question and is the central question this platform investigates.

---

## 1. Quantum Computing Context [A]

### 1.1 Current Hardware Constraints

We are operating in the **NISQ era** (Noisy Intermediate-Scale Quantum) as defined by Preskill (2018). Current quantum devices:

- Have 10–1000+ physical qubits (IBM Eagle, Falcon; IonQ Aria; Quantinuum H1).
- Have gate fidelities typically in the range 99.0–99.9% for single-qubit and 98–99.5% for two-qubit gates.
- Have coherence times (T1, T2) on the order of microseconds to milliseconds.
- Do not yet have error-corrected logical qubits at scale.

**Implication:** Any quantum experiment in this project must account for noise. Noiseless simulation results are scientifically valid for understanding circuit properties but must not be presented as achievable on real hardware without noise modeling.

### 1.2 What Quantum Circuits Can Do [A]

Parameterized quantum circuits (PQCs) are differentiable quantum models where circuit parameters (rotation angles) are trained by classical optimization. This is an established area of research (Cerezo et al., 2021; Benedetti et al., 2019).

A PQC with n qubits and L layers of parameterized gates implements a mapping:

```
x ∈ ℝᴰ  →  encode  →  U(θ, x)  →  measure  →  ⟨O⟩ ∈ ℝᴷ
```

Where:
- `encode` maps classical features to quantum states
- `U(θ, x)` is the variational ansatz
- `⟨O⟩` is the expectation value of observables (the output)
- `θ` are trainable parameters optimized classically

### 1.3 Quantum Kernel Equivalence [A]

[A] It has been established (Schuld & Killoran, 2019; Havlíček et al., 2019) that PQCs can be interpreted as implementing a quantum kernel:

```
κ(x, x') = |⟨ψ(x)|ψ(x')⟩|²
```

**[B] Hypothesis:** This quantum kernel may capture feature interactions in clinical biomarker data that are not easily approximated by standard RBF or polynomial kernels. This is untested.

**[C] Experimental claim (to investigate):** Does the quantum kernel show better alignment with the true risk prediction function structure than classical kernels?

---

## 2. Quantum Variational Biomarker Embedding (QVBE) [B]

### 2.1 Concept

QVBE is the primary quantum hypothesis in Nāḍī-Q. The proposal is:

1. Encode a subset of classical biomarker features into a quantum state.
2. Apply a parameterized variational circuit (ansatz).
3. Measure expectation values of Pauli observables.
4. Use these expectation values as features for a classical classifier head.

**This is a hypothesis.** It has not been tested. It may produce no advantage over classical methods.

### 2.2 Proposed Architecture (Subject to Change)

```
Classical features x ∈ ℝᴰ
        │
        ▼ (optional classical reduction to n-dimensional subspace)
x̃ ∈ ℝⁿ  where n = number of qubits
        │
        ▼ Data Encoding Layer [B]
        U_enc(x̃) — maps features to rotation angles
        │
        ▼ Variational Ansatz U(θ) [B]
        L layers of:
          ├── Single-qubit rotations: Ry(θᵢ), Rz(φᵢ)
          └── Entangling gates: CNOT (linear or all-to-all)
        │
        ▼ Measurement
        {⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Zₙ₋₁⟩}  ∈ [-1, 1]ⁿ
        │
        ▼ Quantum feature vector q ∈ ℝⁿ
```

### 2.3 Encoding Strategies Under Investigation [B]

| Strategy | Description | Key Property | Status |
|----------|-------------|--------------|--------|
| Angle encoding | x → Ry(πxᵢ) | O(n) gates | To test |
| Amplitude encoding | x → quantum state with amplitudes ∝ xᵢ | Exponentially compact | To test |
| IQP encoding | Diagonal gates: e^(ixᵢxⱼZᵢZⱼ) | Hard to simulate classically | To test |
| Re-uploading | Repeated encoding interspersed with U(θ) | Universal approximator | To test |

**[C] Experimental claim:** We will compare encoding strategies on synthetic data (where the ground truth is known) to understand their inductive biases before applying them to clinical features.

### 2.4 Ansatz Choices Under Investigation [B]

- Hardware-efficient ansatz (HEA): rotations + nearest-neighbour CNOT
- Strongly entangling layers (Schuld et al.)
- Data re-uploading ansatz (Pérez-Salinas et al.)

Ansatz depth L and qubit count n are hyperparameters to be ablated experimentally.

---

## 3. Noise Modeling [A]

### 3.1 Noise Channels

Real quantum hardware introduces errors modeled as quantum channels. We will simulate the following:

| Channel | Description | Physical Source |
|---------|-------------|----------------|
| Depolarizing | Random Pauli error with probability p | Gate imperfection |
| Amplitude damping | |1⟩ → |0⟩ with probability γ | Energy relaxation (T1) |
| Phase damping | Dephasing without energy loss | T2 decoherence |
| Readout error | Bit-flip during measurement | Detector error |

### 3.2 Noise Simulation Strategy [A]

- All noise models will be calibrated to publicly available device calibration data (e.g., IBM Quantum calibration JSONs).
- Noise-free simulation is acceptable for circuit design but must be labelled clearly.
- Noise-injected results are required before drawing any conclusion about real-hardware feasibility.

### 3.3 Error Mitigation [D]

The following error mitigation techniques are known in the literature but are deferred to future implementation:

- Zero-noise extrapolation (ZNE)
- Probabilistic error cancellation (PEC)
- Measurement error mitigation (matrix inversion)

---

## 4. Backend Abstraction [A]

### 4.1 Design Requirement

Quantum circuit code in `src/nadiq/quantum/` must be backend-agnostic. The same circuit definition must be executable on:

- PennyLane + Lightning (default, fast CPU simulation)
- Qiskit Aer (noise simulation, IBM-compatible)
- IBM Quantum (real hardware, future)

### 4.2 Backend Interface

```python
class QuantumBackend(ABC):

    @abstractmethod
    def execute(self, circuit: QuantumCircuit, shots: int) -> MeasurementResult:
        """Execute a circuit and return measurement results."""

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Return device properties (qubits, gate set, noise model)."""

    @property
    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether this backend supports parameter-shift gradient computation."""
```

Backend selection is purely configuration-driven:

```yaml
quantum:
  backend: pennylane_lightning   # or: qiskit_aer, ibm_quantum
  n_qubits: 8
  shots: 1024
  noise_model: null              # or: configs/quantum/noise_model_basic.yaml
```

---

## 5. Gradient Computation [A]

Training PQCs requires computing gradients of expectation values with respect to circuit parameters. The **parameter-shift rule** is an established method:

```
∂⟨O⟩/∂θᵢ = (1/2)[⟨O⟩(θᵢ + π/2) - ⟨O⟩(θᵢ - π/2)]
```

This is analytically exact (not a finite-difference approximation) and is hardware-compatible. It is implemented by PennyLane natively.

---

## 6. Open Quantum Design Questions

The following design decisions are unresolved and require experimentation:

1. **Qubit count:** How many qubits are needed/useful for D-dimensional tabular clinical features? (Likely n ≪ D; the encoding strategy determines this.)
2. **Entanglement structure:** All-to-all vs. linear chain vs. random — does it matter for this data regime?
3. **Barren plateau risk:** Deep circuits suffer from exponentially vanishing gradients (barren plateaus; McClean et al., 2018). How deep can we go before training fails?
4. **Feature selection pre-encoding:** Which subset of D features should be encoded quantumly? This is itself a research question.
5. **Output observable:** Single-qubit Z expectation vs. multi-qubit correlators — which carries more discriminative information?

---

## 7. References

All references are to peer-reviewed literature. No claim in this document is asserted without citation or explicit hypothesis labelling.

- Cerezo et al. (2021). Variational quantum algorithms. *Nature Reviews Physics.*
- Benedetti et al. (2019). Parameterized quantum circuits as machine learning models. *Quantum Science and Technology.*
- Schuld & Killoran (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters.*
- Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature.*
- McClean et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications.*
- Pérez-Salinas et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum.*
- Preskill (2018). Quantum computing in the NISQ era and beyond. *Quantum.*
