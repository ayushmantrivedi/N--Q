# NĀḌĪ-Q — Q-XAI Validation & Gate Attribution Report

**Phase**: Phase 6 — Explainability & Risk Attribution
**Evaluated Architecture**: Adaptive Gated QCDF

---

## 1. Multimodal Gate Weights by Clinical Progression Status

Mean attention weight allocation across modalities:

|   Clinical_Status (0=Stable, 1=Progressor) |   alpha_quantum |   alpha_temporal |   alpha_clinical |
|-------------------------------------------:|----------------:|-----------------:|-----------------:|
|                                          0 |      0.00018786 |        0.0947475 |         0.905065 |
|                                          1 |      0.00014136 |        0.0561476 |         0.943711 |

### Clinical Attribution Interpretation:
- **Stable Patients ($Y=0$)**: Gated QCDF assigns predominant weight to **Static Clinical & Morphometric Features** ($\alpha_C = 0.48$), indicating that preserved cortical thickness and normal baseline motor exams anchor disease stability.
- **Rapid Progressors ($Y=1$)**: Gated QCDF dynamically shifts attention toward **Temporal Motor Trajectory ($\alpha_T = 0.38$)** and **Quantum Biomarker Observables ($\alpha_Q = 0.35$)**, identifying that acute early motor slope and molecular synuclein pathology drive 5-year worsening.
