# NĀḌĪ-Q — QVBE-v2 Frozen Configuration & Model Checkpoint Manifest

**Phase**: Phase 5 — Confirmatory Validation & Open-Source Model Integration
**Status**: FROZEN & IMMUTABLE
**Module Path**: `src/nadiq/models/quantum/qvbe_v2.py`
**Module SHA256**: `D4B1F319CB3783F23AD51B30679DF90139838DDA22656D5549C985BD0FFC6247`
**Checkpoint Path**: `F:\NADI-Q_AUDIT_OUTPUT\PHASE_5\qvbe_v2_best_mode_a_8q.json`
**Checkpoint SHA256**: `96510D505C56F23AFA728D035A467E97C55E6CB9C400546B5018122D2FC1C08D`

---

## 1. Primary Winning Configuration Specification (D+ Mode A, 8Q)

| Hyperparameter / Setting | Value | Scientific Rationale |
| :--- | :--- | :--- |
| **Architecture** | `QuantumVariationalBiomarkerEncoderV2` | Standalone Scikit-Learn transformer interface with multi-basis readout |
| **Input Dimension** | 249 | Validated multimodal feature vector (FreeSurfer, MDS-UPDRS, Genetics, PRS) |
| **Qubit Count ($n$)** | 8 | Sufficient Hilbert dimension ($2^8 = 256$) avoiding barren plateaus |
| **Circuit Depth ($L$)** | 2 | Empirically validated optimal depth from Phase 4 ablations ($L=2$ beat $L=1$ and $L=4$) |
| **Unitary Parameterization** | Cyclic $U_3(\theta, \phi, \lambda)$ | Full $SU(2)$ arbitrary single-qubit rotations (3 parameters per qubit per layer) |
| **Entanglement Topology** | Cyclic CNOT Ladder | Periodic boundary connectivity: $\text{CNOT}(i, (i+1) \bmod n)$ |
| **Quantum Attention** | Hierarchical QAF | Multi-scale controlled phase rotations across dyadic strides $2^s$ |
| **Measurement Basis** | Full Multi-Basis ($M=23$) | $8 \times \langle Z \rangle + 8 \times \langle X \rangle + 7 \times \langle ZZ \rangle$ correlators |
| **Execution Backend** | PennyLane `default.qubit` | Exact analytic statevector contraction (zero shot noise) |
| **Operational Mode** | Mode A (`fixed`) | Deterministic seed initialization + two-stage convex warm-start on Layer 0 |
| **Downstream Classifier** | LightGBM | GBDT (`n_estimators=100`, `learning_rate=0.05`, `max_depth=4`, `seed=42`) |
| **Input Feature Representation** | Fused $[X \parallel Q]$ ($D=272$) | 249 raw standardized classical features + 23 quantum expectation features |

---

## 2. Parameter Tensors & Observables Registry

### A. $U_3$ Parameter Tensor (`u3_params`)
- **Shape**: `(2, 8, 3)` = 48 variational angles in $[-\pi, \pi]$
- **Layer 0**: Initialized via linear discriminant projection (Ridge Classifier on $X_{\text{angles}}$)
- **Layer 1**: Deterministic pseudo-random initialization from Seed 42

### B. QAF Parameter Tensor (`qaf_params`)
- **Shape**: `(3, 8)` = 24 controlled phase rotation angles in $[0, \pi/2]$

### C. Measurement Observables ($M = 23$)
- Features 0–7: $\langle Z_0 \rangle, \dots, \langle Z_7 \rangle \in [-1, 1]$
- Features 8–15: $\langle X_0 \rangle, \dots, \langle X_7 \rangle \in [-1, 1]$
- Features 16–22: $\langle Z_0 Z_1 \rangle, \langle Z_1 Z_2 \rangle, \dots, \langle Z_6 Z_7 \rangle \in [-1, 1]$

---

## 3. Environment & Software Stack

- **Operating System**: Windows 11 x64
- **Python Version**: 3.12.4
- **PennyLane**: 0.41.1
- **Scikit-Learn**: 1.6.1
- **LightGBM**: 4.6.0
- **XGBoost**: 3.2.0
- **NumPy**: 2.2.6
- **Pandas**: 2.3.3

---

## 4. Inviolable Freezing Guarantees

1. **Zero Out-of-Partition Adaptation**: All scalers (`StandardScaler`, `MinMaxScaler`), PCA projections, and Layer 0 warm-start angles are calibrated strictly on training partitions ($N=1,172$).
2. **Runtime Mutation Lock**: Any invocation of `fit()` on this frozen instance raises a strict `RuntimeError('Cannot fit a frozen QVBE instance.')`.
3. **Determinism**: Re-evaluating the serialized checkpoint on the same test data yields exact numerical reproducibility down to machine epsilon (max drift $< 10^{-14}$).
