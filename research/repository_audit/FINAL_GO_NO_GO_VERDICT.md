# NĀḌĪ-Q External Repository Audit & Final Architecture Verdict

**Audit Completion Date**: November 23, 2026  
**Auditor**: Senior Quantum Machine Learning Research Engineer & Biomedical AI Architect  
**Project**: NĀḌĪ-Q Hybrid Quantum-Classical Platform for Parkinson's Disease Progression  
**Document ID**: `NADIQ-FINAL-AUDIT-VERDICT-2026`  
**Governing Standard**: Strict Code-Level Verification & Falsifiable Empirical Evidence

---

## 1. Executive Answers to the Eight Mandatory Audit Questions

### 1. Which repository is most valuable for NĀḌĪ-Q?
**Answer**: **`PennyLane` (`PennyLaneAI/pennylane`)** for quantum simulation infrastructure, and **`DeepHypergraph` (`iMoonLab/DeepHypergraph`)** for classical hypergraph modeling.
- *Reasoning*: PennyLane provides our verified, differentiable quantum simulation foundation. DeepHypergraph provides clean, pure-PyTorch sparse hypergraph incidence matrices ($H$) and modern multi-order convolutions (`HNHN`, `UniGNN`) with zero compiled C++ dependencies, running natively on Windows Python 3.12.

### 2. Which repository is most likely to improve clinical prediction?
**Answer**: **`YAIB` (`rvandewater/YAIB`)**, but **strictly via isolated extraction of its `GRUDCell` (GRU-D)**, NOT by adopting its outer package.
- *Reasoning*: Che et al.'s continuous exponential temporal decay ($\gamma = \exp(-\max(0, W \delta_t + b))$) directly targets the primary challenge in longitudinal clinical data: irregular visit intervals and decaying confidence of unobserved biomarkers. All other candidates (CEHR-BERT, GSoC-QGNN) actively degrade or overfit small-cohort clinical prediction.

### 3. Which repository is best for rebuilding the hypergraph?
**Answer**: **`DeepHypergraph` (`dhg`)**.
- *Reasoning*: It imports cleanly in Python 3.12 without external C++ compilation (unlike MHNN which requires `torch_scatter`). It represents hyperedges via incidence matrices that directly interface with MeTTa's relational atoms. Note: DHG's `directed_hypergraph.py` is empty (0 bytes), so directionality must be modeled by partitioning $H$ into source/target incidence matrices.

### 4. Which quantum repository is worth testing first?
**Answer**: **None of the external quantum repositories are worth testing.**
- *Reasoning*:
  - `QuantumGNN_molecules` contains **zero quantum code** (100% classical PyTorch calculating QM9 distance potentials; mislabeled repository).
  - `GSoC-2024-QGNN` has no quantum graph operations (it is merely a standard VQC applied node-wise before classical PyG GCN propagation), is unlicensed, and relies on `AngleEmbedding` which has already been proven to collapse in Phase 11 through 14A.
  - **PennyLane** is already our native, gold-standard quantum framework. No external quantum code matches our local rigor.

### 5. Which repositories should we reject?
**Answer**: We formally **REJECT five (5) repositories**:
1. **`cehrbert`**: Rejected due to fundamental data domain mismatch (requires OMOP CDM discrete concept codes, massive overfitting on $N=341$, requires Linux `flash-attn-2`).
2. **`HGNN`**: Rejected because it is officially deprecated/obsolete and superseded by DHG; static 2019 clique expansion with zero temporal or directed hyperedges.
3. **`Hyper-SAGNN`**: Rejected due to obsolete tech stack (requires Python 3.6, TensorFlow 1.x) and inappropriate methodology (Word2Vec random walks on 6-node clinical graphs).
4. **`GSoC-2024-QGNN`**: Rejected because it has no quantum graph operations, has no open-source license (legal liability), and replicates already-disproven VQC encoding failures.
5. **`QuantumGNN_molecules`**: Rejected due to false advertising (contains no quantum circuits) and Windows filesystem path errors.

### 6. What are the three highest-priority implementation tasks?
**Answer**:
1. **Task 1 — Implement Standalone GRU-D Cell**: Port the ~150 lines of PyTorch temporal decay logic from YAIB's `grud.py` into `src/nadiq/models/temporal/grud.py` (free of `gin-config` or Ignite) and benchmark against CTCLM on Task T2.
2. **Task 2 — Integrate DeepHypergraph (DHG) with MeTTa**: Replace the legacy hypergraph builder with `dhg.Hypergraph`, using MeTTa's evidence promotion rules ($r \ge 0.05, p \le 0.05$) to populate the sparse incidence matrix $H$, and test 2-layer HNHN message passing against Logistic Regression.
3. **Task 3 — Decouple Quantum Modules into an Isolated Research Sandbox**: Officially deprecate and decouple all quantum circuits from the production clinical prediction path; freeze **Option C** (Classical-First Clinical Prediction) to guarantee maximum prognostic accuracy and calibration for patients.

### 7. What evidence would justify integrating a quantum component into the clinical prediction path?
**Answer**: Strict, non-negotiable empirical superiority over classical controls on out-of-fold clinical data:
$$\text{AUROC}(M_{\text{quantum}}) - \text{AUROC}(M_{\text{classical\_control}}) > 0 \quad \text{AND} \quad \text{95% Bootstrap CI Lower Bound} > +0.02$$
$$\text{ECE}(M_{\text{quantum}}) \le \text{ECE}(M_{\text{classical\_control}})$$
$$\text{Brier}(M_{\text{quantum}}) \le \text{Brier}(M_{\text{classical\_control}})$$
If a quantum component cannot demonstrate that the 95% confidence interval for AUROC difference strictly excludes zero when tested against an identical matched classical baseline, it **must not** be admitted into the clinical diagnostic pipeline.

### 8. What would cause us to stop quantum experimentation on this clinical task?
**Answer**: **The stopping condition has already been met.**
- In Phase 14A, on the primary longitudinal progression endpoint (**Task T2**), the Biological Relational Quantum Kernel (BRQK) collapsed to an AUROC of **0.4158** (worse than chance).
- Hybridizing the quantum kernel with CTCLM caused performance to drop from AUROC **0.6131** down to **0.5027** ($\Delta = -0.1104, p = 0.0010$), and inflated calibration error by $12\times$ (ECE jumped from 0.0192 to 0.2450).
- Across five consecutive phases (11A, 11B, 12.1, 12.2, 13, 14A), five distinct quantum formulations (variational encoders, graph circuits, QAOA optimization, relational pretraining, and quantum kernels) all failed to beat classical baselines.
- **Therefore, quantum experimentation on clinical disease progression prediction from PPMI tabular data is officially TERMINATED.** PennyLane is retained solely for decoupled synthetic theoretical research.

---

## 2. Summary of Repository Verdicts

```
================================================================================
FINAL VERDICT SUMMARY
================================================================================
1. YAIB                   -> ADAPT (Extract pure PyTorch GRU-D cell only)
2. CEHR-BERT              -> REJECT (OMOP/BERT mismatch, severe overfitting)
3. HGNN                   -> REJECT (Deprecated 2019, superseded by DHG)
4. DeepHypergraph (DHG)   -> ADAPT (Clean PyTorch hypergraph incidence & HNHN)
5. Hyper-SAGNN            -> REJECT (Fatal TF 1.x dependency, Word2Vec walks)
6. MHNN                   -> RESEARCH ONLY (Dual update math; drop torch_scatter)
7. GSoC-2024-QGNN         -> REJECT (Zero quantum graph ops, unlicensed)
8. QuantumGNN Molecules   -> REJECT (100% classical, misleading title, path bugs)
9. PennyLane              -> ADOPT (Decoupled research sandbox only)
================================================================================
```

---

## 3. Official Architectural Mandate

The NĀḌĪ-Q platform officially adopts:

```
================================================================================
RECOMMENDED ARCHITECTURE: OPTION C
(CLASSICAL-FIRST RESEARCH PLATFORM WITH DECOUPLED QUANTUM SANDBOX)
================================================================================
- Clinical Prognosis Engine:  CTCLM / GRU-D + Linear/Logistic Multimodal Fusion
- Knowledge Representation:   MeTTa (SingularityNET Hyperon) Symbolic Reasoner
- Hypergraph Architecture:    DeepHypergraph (DHG) Sparse Incidence Structures
- Quantum Module Status:      Decoupled from Clinical Pipeline; Maintained in
                              Isolated PennyLane Benchmarking Sandbox
- Patient Data Safety:        Strict Local Storage on F:\ Storage; Zero Remote Egress
================================================================================
```
