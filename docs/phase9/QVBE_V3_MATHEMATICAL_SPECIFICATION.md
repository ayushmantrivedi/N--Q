# QVBE-v3 Mathematical Specification
## NĀḌĪ-Q Phase 9 — Quantum Variational Biomarker Encoder, Version 3

**Document status:** Authoritative specification  
**Date:** 2026-11-18  
**Supersedes:** QVBE-v2 (Phase 4–8)  
**Evidence tags used throughout:** [MATHEMATICAL DERIVATION], [VERIFIED FROM SOURCE CODE], [VERIFIED FROM EXPERIMENT OUTPUT], [REASONABLE INFERENCE], [LITERATURE REFERENCE]

---

## 1. Motivation & Failure Analysis

QVBE-v2 exhibited seven critical failures identified in Phase 8 deep diagnostic [VERIFIED FROM EXPERIMENT OUTPUT]:

| # | Failure | Root Cause |
|---|---------|-----------|
| 1 | Pre-quantum PCA bottleneck | 256→8 PCA collapses input before circuit |
| 2 | Mode B broken gradient loop | `y_sub` extracted but never enters backward pass |
| 3 | Effective rank = 8.68 | Capped by PCA output dimension, not qubit Hilbert space |
| 4 | Param variance > patient variance | σ²_params = 0.0152 > σ²_bio = 0.0098 |
| 5 | Entanglement negligible | Product state CCA r = 0.9977 ≈ entangled |
| 6 | Mode A: zero variational optimization | Fixed after one Ridge warm-start, no gradient flow |
| 7 | 23 local Paulis discard 232/255 d.o.f. | 8-qubit Hilbert space has 2⁸ = 256 state dimensions |

QVBE-v3 is a ground-up redesign that corrects all seven failures simultaneously.

---

## 2. Input Encoding Strategy

### 2.1 Data Re-uploading Encoder (Perez-Salinas 2020)

**Failure addressed:** #1 (PCA bottleneck), #3 (effective rank), #5 (entanglement negligible)

[LITERATURE REFERENCE] Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J.I. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.

The key insight of data re-uploading is that a single-qubit quantum circuit can approximate **any** continuous function if input features are uploaded multiple times as rotation angles, interspersed with trainable variational layers.

**Formal Statement.** For a circuit with K uploads of input vector **x** ∈ ℝᵈ:

```
|ψ_k(x, θ)⟩ = U_var(θ_k) · U_enc(x) · |ψ_{k-1}⟩
```

where `U_enc(x)` applies `RY(x_i)` to qubit i.

**Fourier Spectrum Extension.** The expectation value of a Pauli observable O as a function of the input is:

```
⟨O⟩(x) = Σ_{ω ∈ Ω} c_ω exp(i ω · x)
```

**Theorem (Perez-Salinas 2020, extended):** With K uploads and n qubits, the accessible Fourier frequency spectrum is:

```
Ω_K = {ω ∈ ℤⁿ : |ω_j| ≤ K  ∀j = 1,...,n}
```

This gives (2K+1)ⁿ accessible frequencies. For v2 (K=1, n=8): 3⁸ = 6,561 frequencies. For v3 (K=3, n=8): 7⁸ = 5,764,801 frequencies.

[MATHEMATICAL DERIVATION]

**Proof sketch.** Each upload introduces a factor `exp(i·x_j·σ_j/2)` for each qubit j (where σ_j is a Pauli generator). The composition of K such layers contributes frequency components at ±1 per upload per qubit. By the trigonometric product formula, K uploads on the same qubit generate frequencies in {-K, -K+1, ..., K-1, K}. Because the qubits are independent in the encoding layer (only entangling layers follow), the full multi-qubit spectrum is the Cartesian product over qubits, giving the stated Ω_K. ■

**Consequence for effective rank.** In QVBE-v2 the effective rank of the observable matrix was bounded by min(23, PCA_dim) = min(23, 8) = 8.68. With data re-uploading and no PCA bottleneck, the encoding injects arbitrary Fourier modes; the observable matrix can achieve full rank up to min(23+8, batch_size) = 31, limited only by the number of observables.

### 2.2 Trainable Input Adapter (for d > 24)

**Failure addressed:** #1 (replace fixed PCA with learnable projection)

For high-dimensional inputs (d > 24 features), a fixed PCA discards biological information permanently. QVBE-v3 replaces this with a **Trainable Input Adapter**: a learnable linear map

```
W ∈ ℝ^{n_qubits × d}  (default: 8 × d)
z = W · x ∈ ℝ^{n_qubits}
```

W is a `torch.nn.Parameter` trained jointly with the quantum circuit parameters via the InfoNCE objective. This ensures the projection is optimized for quantum representational utility, not for PCA variance.

**Key difference from PCA:**
- PCA: fixed eigenvectors of data covariance; discards components irrelevant to PCA variance (but not to biomarker signal)
- Trainable W: columns are learned end-to-end; every feature can contribute proportional to its discriminative value for the quantum objective

For d ≤ 24 (direct encoding feasible): features x ∈ ℝᵈ are padded to n_qubits=8 dimensions with zero-padding if d < n_qubits, or a simple learned linear layer if d > n_qubits but ≤ 24. Data re-uploading repeats the encoding K times.

---

## 3. Circuit Design

### 3.1 Hardware-Efficient Ansatz (HEA), Depth 4

**Failure addressed:** #5 (entanglement negligible), #6 (zero variational optimization)

```
|0⟩^⊗n ──[RY(θ)][RZ(φ)] ── CNOT_cyclic ── [RY(θ)][RZ(φ)] ── CNOT_cyclic ── ...
         Layer 1                            Layer 2                            Layer L
```

Each variational layer consists of:
1. **RY-RZ rotation block:** `RY(θ_{l,i}) · RZ(φ_{l,i})` for each qubit i, layer l
2. **Cyclic CNOT entangler:** `CNOT(i, (i+1) % n)` for i = 0,...,n-1

For n=8, L=4: total variational parameters = 2 × n × L = 2 × 8 × 4 = 64 parameters.

The data re-uploading interleaves encoding between variational layers:

```
U(x, θ) = ∏_{k=1}^{K} [U_var(θ_k) · U_enc(x, w_k)]
```

where `U_enc(x, w_k) = ⊗_i RY(x_i · w_{k,i})` and w_{k,i} are trainable input re-weighting scalars.

**Why HEA over U3 ansatz (v2)?** [MATHEMATICAL DERIVATION]  
The U3 ansatz in v2 has 3 parameters per qubit per layer (θ, φ, λ). However, global phase λ of U3 is irrelevant for expectation values (it cancels in ⟨O⟩ = ⟨ψ|O|ψ⟩). The HEA with RY + RZ captures the same SU(2) manifold per qubit with 2 parameters, is more hardware-native, and has better-studied trainability properties (see McClean 2018 on barren plateaus with layered circuits).

### 3.2 Entanglement Layer Analysis

[MATHEMATICAL DERIVATION]

For the cyclic CNOT ladder, the generated entanglement in a 4-layer circuit (L=4, n=8):

The cyclic CNOT entangles all qubits: qubit 0↔1, 1↔2, ..., 6↔7, 7↔0. After 4 layers with intervening rotations, the circuit generates genuine multipartite entanglement (GME). The state belongs to the full 2⁸-dimensional Hilbert space rather than a tensor product of 1-qubit spaces.

**Failure #5 fix:** In v2, entanglement contribution was negligible because the upstream PCA constrained the input to 8 near-identical features, making the entanglement between qubits meaningless. By uploading diverse features directly, different qubits encode different biomarker modalities, and CNOT entanglement creates biologically meaningful inter-modal correlations.

---

## 4. Observable Set

### 4.1 Retained Local Paulis from v2 (23 observables)

[VERIFIED FROM SOURCE CODE] From `qvbe_v2.py` measurement_basis="full" giving 3n-1 = 23 observables:
- Z_i for i = 0,...,7 (8 observables)
- X_i for i = 0,...,7 (8 observables)
- Z_i Z_{i+1} for i = 0,...,6 (7 nearest-neighbor ZZ correlators)

**Failure #7 analysis:** 23 observables over a 2⁸ = 256-dimensional Hilbert space. The density matrix ρ has 255 independent real parameters (SU(256) has 256²-1 = 65535 generators, but physical states are on the Bloch hypersphere). 23 observables capture at most 23 independent expectation values out of 255 real parameters: 232/255 ≈ 91% of quantum state information is discarded.

### 4.2 New: 8 Connected ZZ Correlators (non-adjacent pairs)

QVBE-v3 adds 8 non-adjacent ZZ correlators to capture longer-range qubit correlations:

```
C_ij = ⟨Z_i Z_j⟩ - ⟨Z_i⟩⟨Z_j⟩  for (i,j) ∈ S_nonlocal
```

where S_nonlocal = {(0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (0,4), (1,5)} (8 pairs spanning 1-hop and 4-hop non-adjacent connections).

**Why connected correlators?** [MATHEMATICAL DERIVATION]

The raw correlator ⟨Z_i Z_j⟩ can be decomposed as:
```
⟨Z_i Z_j⟩ = C_ij + ⟨Z_i⟩⟨Z_j⟩
```

For a product state: C_ij = 0 by definition. A non-zero C_ij implies genuine quantum correlations (entanglement) between qubits i and j. By adding C_ij to the observable set, we explicitly measure whether the circuit is generating entanglement beyond product-state behavior.

**Total observable count:** 23 (local) + 8 (connected ZZ) = **31 observables**.

New effective state space coverage: 31/255 ≈ 12.2% (vs. 9% in v2). While still a small fraction, the connected correlators capture a qualitatively different subspace of the state.

---

## 5. Learning Objective: Cross-Modal InfoNCE

### 5.1 Objective Design

**Failure addressed:** #2 (broken Mode B), #6 (zero variational optimization)

QVBE-v3 uses **cross-modal self-supervised alignment** between:
- Q_mol: quantum embedding of molecular/structural biomarkers (from QVBE-v3 circuit)
- h_temp: temporal representation from CTCLM hidden state

**Constraint:** Do NOT use 5-year progression labels (target_progression_5yr_mcid5). This is an unsupervised/self-supervised objective.

The **InfoNCE** (Information Noise-Contrastive Estimation) loss [LITERATURE REFERENCE]:

Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

```
L_InfoNCE = -E[log( exp(sim(q_i, h_i)/τ) / Σ_j exp(sim(q_i, h_j)/τ) )]
```

For a batch of N participants, expanded:

```
L = -(1/N) Σ_{i=1}^{N} log [ exp(q_i · h_i / τ) / Σ_{j=1}^{N} exp(q_i · h_j / τ) ]
```

where:
- q_i = L2-normalized quantum embedding for participant i: q_i = Q_mol_i / ||Q_mol_i||
- h_i = L2-normalized temporal embedding for participant i: h_i = h_temp_i / ||h_temp_i||
- sim(·,·) = cosine similarity = dot product after L2 normalization
- τ = temperature hyperparameter (default: 0.07)

### 5.2 Positive and Negative Pairs

[MATHEMATICAL DERIVATION]

For participant i:
- **Positive pair:** (q_i, h_i) — same participant, molecular ↔ temporal embeddings
- **Negative pairs:** (q_i, h_j) for all j ≠ i — different participants (N-1 negatives per anchor)

**Mutual information bound:** InfoNCE is a lower bound on the mutual information between q and h modalities:

```
I(q; h) ≥ log(N) - L_InfoNCE
```

Maximizing I(q; h) is equivalent to minimizing L_InfoNCE. When L → 0, the quantum encoder must produce representations that are maximally predictive of temporal features for the SAME participant, while being discriminative from ALL other participants.

**Why this fixes failures #2 and #6:**
- Mode B had `y_sub` (subset labels) that were never connected to the gradient. InfoNCE replaces arbitrary supervised targets with a well-defined contrastive objective that naturally creates gradients through the quantum circuit parameters.
- Mode A had a fixed Ridge classifier warm-start. InfoNCE provides a fully differentiable loss that propagates gradients back through ALL quantum parameters via parameter-shift.

### 5.3 Temperature Calibration

The temperature τ controls the sharpness of the probability distribution:
- τ → 0: loss focuses on hard negatives (easy to collapse to trivial solutions)
- τ → ∞: loss ignores negative pairs (too weak a signal)
- τ = 0.07: standard from SimCLR (Chen 2020), suitable for normalized embeddings

---

## 6. Gradient Flow and Differentiability

### 6.1 Parameter-Shift Rule

[MATHEMATICAL DERIVATION]

For a quantum circuit with parameters θ, the gradient of an expectation value with respect to θ_k is:

```
∂⟨O⟩/∂θ_k = [⟨O⟩(θ_k + π/2) - ⟨O⟩(θ_k - π/2)] / 2
```

This is the **parameter-shift rule** (Mitarai 2018, Schuld 2019). It computes exact gradients using two additional circuit evaluations per parameter, without finite difference approximation.

QVBE-v3 uses `diff_method="parameter-shift"` in PennyLane, enabling exact gradient computation. The input adapter W is a `torch.nn.Parameter` and receives gradients via standard PyTorch autograd (no parameter shift needed for classical parameters).

### 6.2 End-to-End Gradient Chain

```
L_InfoNCE
  → ∂L/∂q_i (autograd through cosine similarity)
  → ∂q_i/∂circuit_output (autograd through output projection if any)
  → ∂circuit_output/∂θ_k (parameter-shift in PennyLane)
  → ∂circuit_output/∂W (autograd through input adapter)
```

All parameters {θ, W} receive non-None gradients after a single backward() call.

---

## 7. Hyperparameter Summary

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_qubits | 8 | Match v2 hardware target |
| n_layers | 4 | 2× v2 depth for richer expressivity |
| n_uploads | 3 | Extends Fourier spectrum to 7⁸ frequencies |
| use_trainable_adapter | True | Replaces fixed PCA |
| measurement_basis | "full_plus_connected" | 23 + 8 = 31 observables |
| tau (InfoNCE) | 0.07 | Standard SimCLR temperature |
| learning_rate | 1e-3 | Adam default |
| adapter_dim | n_qubits (8) | Minimal projection preserving diversity |
| diff_method | "parameter-shift" | Exact gradients, hardware compatible |

---

## 8. Complexity Analysis

### 8.1 Circuit Evaluation Cost

For n=8 qubits, L=4 layers, K=3 uploads:
- Gates per upload: n RY gates = 8
- Variational layer: 2n rotations + n CNOTs = 24 gates
- Total gates: K × (n + 2n + n) = 3 × 32 = 96 gates per sample
- Observable evaluations: 31 (23 local + 8 connected ZZ)
- Parameter-shift evaluations: 2 per parameter × 64 params = 128 circuit evaluations per gradient step

### 8.2 Memory and Batch Considerations

The quantum circuit processes one sample at a time (PennyLane batching may cause issues with data re-uploading — see implementation note in circuit.py). For a batch of N samples, N sequential circuit calls are made. This is O(N) in time.

For N=32 participants (typical batch): 32 × 128 = 4,096 circuit evaluations per gradient step. On CPU simulator: ~0.5ms per evaluation → ~2 seconds per gradient step. This is acceptable for small batch sizes.

---

## 9. References

1. Pérez-Salinas et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
2. Mitarai, K., et al. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309.
3. Schuld, M., et al. (2019). Evaluating analytic gradients on quantum hardware. *Physical Review A*, 99(3), 032331.
4. van den Oord, A., et al. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*.
5. Chen, T., et al. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*.
6. McClean, J.R., et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812.
7. Cerezo, M., et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625-644.
