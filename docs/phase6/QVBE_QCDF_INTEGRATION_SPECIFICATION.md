# NĀḌĪ-Q — QVBE-v2 to QCDF Integration Specification

**Phase**: Phase 6 — Multimodal Integration
**Component**: Frozen QVBE-v2 Interface Contract

---

## 1. Frozen Quantum Feature Ingestion Contract

- **Preserved Checkpoint**: `qvbe_v2_best_mode_a_8q.json` (SHA256: `96510D505C56F23AFA728D035A467E97C55E6CB9C400546B5018122D2FC1C08D`)
- **Qubit Dimension**: $N_q = 8$ qubits
- **Variational Depth**: $L = 2$ layers with cyclic $U_3$ parameterized rotations and CNOT entanglement
- **Measurement Observables ($M=23$)**: 8 Pauli-Z + 8 Pauli-X + 7 nearest-neighbor Pauli-ZZ correlators
- **Integration Mode**: Within each stratified fold, input scalers are fitted on $X_{\text{train}}$, parameters frozen, and $Q \in \mathbb{R}^{23}$ is delivered directly to QCDF without gradient backpropagation into quantum circuits.

