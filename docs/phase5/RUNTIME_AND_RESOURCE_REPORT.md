# NĀḌĪ-Q — Phase 5 Computational Runtime & Resource Utilization Report

**Phase**: Phase 5 — Confirmatory Validation & Open-Source Model Integration
**Platform**: Local Windows 11 x64, 12 Logical Cores, CPU-only Execution
**Cohort**: PPMI N=1,465 participants across 25 stratified folds (5 seeds x 5 folds)

---

## 1. Pipeline Execution Latencies

| Component | Description | Latency (Per Fold) | 25-Fold Total |
| :--- | :--- | :--- | :--- |
| **FT-Transformer Training** | 20 Epochs, AdamW, Batch Size 64, d=16 on CPU | ~55.0 s | ~22.9 min |
| **QVBE-v2 Transformation** | Mode A 8-Qubit Vectorized QNode Evaluation | ~1.5 s | ~0.6 min |
| **Downstream Classifiers** | LightGBM, XGBoost, Logistic L2 (8 conditions) | ~4.5 s | ~1.9 min |
| **Spectral Diagnostics** | Covariance, Eigenvalues, Cosine Sim, Entropies | ~1.0 s | ~0.4 min |
| **Total Fold Latency** | Full integration loop | ~63.5 s | **26.46 min** |

---

## 2. Hardware Resource Profile

- **RAM Footprint**: Peak memory remained under 1.8 GB during batched PyTorch training and PennyLane statevector caching.
- **QPU Resource Scaling**: With 8 qubits and 2 measurement bases (Z+ZZ and X), physical execution requires 2 circuit evaluations per sample. At 10 kHz repetition rate, per-patient latency is 0.20s (1,024 shots) to 1.64s (8,192 shots).
