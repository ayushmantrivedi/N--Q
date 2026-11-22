# NĀḌĪ-Q — Missing-Modality Robustness Report

**Phase**: Phase 6 — QCDF Integration
**Evaluation**: Systematic masking of modalities during inference

---

## 1. Missing Modality Empirical Performance

| missing_pattern             | available_modalities   |   auroc_mean |   auroc_std |   auprc_mean |   bal_acc_mean |
|:----------------------------|:-----------------------|-------------:|------------:|-------------:|---------------:|
| Missing_Quantum_Biofluids   | Temporal+Static        |     0.580841 |   0.0889687 |     0.602118 |       0.539514 |
| Missing_Static_MRI_Genetics | Quantum+Temporal       |     0.630435 |   0.0289355 |     0.617581 |       0.590542 |
| Missing_Temporal_MDSUPDRS   | Quantum+Static         |     0.578869 |   0.0831447 |     0.594048 |       0.548253 |

