# NĀḌĪ-Q — Phase 6 Computational Runtime & Resource Utilization Report

**Phase**: Phase 6 — CTCLM Development, QCDF Integration & Five-Year Longitudinal Prediction
**Hardware Platform**: Windows 11 x64, 12 Logical Cores, CPU-only Local Execution
**Evaluation Cohort**: N = 341 Multimodal Longitudinal Participants across 5 Stratified Folds

---

## 1. End-to-End Component Runtimes (Per Fold & Total 5-Fold Benchmark)

| Component | Description | Latency (Per Fold) | 5-Fold Total |
| :--- | :--- | :--- | :--- |
| **CTCLM Training** | PyTorch GRU (25 Epochs, batch size full, d=16, dt=4) | ~0.25 s | ~1.25 s |
| **Frozen QVBE-v2** | Mode A 8Q Vectorized Observable Transformation (23 observables) | ~0.35 s | ~1.75 s |
| **QCDF Gated Fusion** | Adaptive Gating Network (30 Epochs, Adam, lr=0.005) | ~0.30 s | ~1.50 s |
| **Classical Downstream Models** | LightGBM across 6 pipelines + 4 ablations + 3 missing patterns | ~1.20 s | ~6.00 s |
| **Total Fold Benchmark** | Full pipeline end-to-end execution | ~2.10 s | **10.54 s** |

---

## 2. Hardware Resource Profile

- **RAM Footprint**: Peak memory remained under 1.2 GB during batched PyTorch sequence tensor evaluation.
- **Zero External API Dependency**: 100% local computation on Windows host; zero patient-level transmission.
