# NĀḌĪ-Q — Reproducibility & Audit Guide
## Complete Execution Blueprint, Environment Specification & Verification Standards

**Project**: NĀḌĪ-Q (Smart India Hackathon 2026, Submission SIH26139)  
**Document**: Reproducibility Guide & Execution Blueprint  
**Authors**: Lead Quantum ML Researcher, Biomedical AI Researcher & Senior Systems Architect  
**Date**: November 14, 2026  
**Status**: VERIFIED REPRODUCIBLE  

---

## 1. Environment & Dependency Specifications

### 1.1 Host Execution Environment
* **Operating System**: Microsoft Windows 11 Enterprise / Pro x64 (Build 22631+)
* **Python Runtime**: CPython 3.12.4 (64-bit)
* **Compute Resources**: 12 Logical Cores (Intel/AMD x64), 32 GB RAM, zero GPU/QPU cloud dependency (100% locally runnable).

### 1.2 Authoritative Dependency Lock
```
pennylane==0.41.1
torch==2.4.1+cpu
scikit-learn==1.6.1
lightgbm==4.6.0
xgboost==3.2.0
scipy==1.14.1
numpy==1.26.4
pandas==2.2.2
pyarrow==17.0.0
pytest==8.3.2
```

---

## 2. Checkpoint Hashes & Data Artifact Integrity

### 2.1 Frozen Quantum Model Checkpoint
* **Path**: `F:\NADI-Q_AUDIT_OUTPUT\PHASE_5\qvbe_v2_best_mode_a_8q.json`
* **SHA-256 Checksum**: `96510D505C56F23AFA728D035A467E97C55E6CB9C400546B5018122D2FC1C08D`
* **Circuit Configuration**: 8 Qubits, 2 Layers, Cyclic CNOT topology, Multi-Basis ($Z + X + ZZ$, 23 observables).
* **Float Parity Verification**: $\Delta = 0.00 \times 10^0$ across all 25 folds.

### 2.2 Curated Longitudinal Parquet Datasets
* Primary Multimodal Progression Cohort ($N=341$):  
  `F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet`
* Early Temporal Motor Sequences ($N=341, 3\times 35$):  
  `F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy`
* Stratified 5-Fold Participant Split Manifest:  
  `F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\PARTICIPANT_SPLIT_MANIFEST_PHASE6.csv`

---

## 3. Step-by-Step Reproduction Blueprint

### Step 1: Scientific Unit & Contract Tests
To verify all quantum circuit operators, leakage guards, scalers, and statistical routines:
```bash
python -m pytest tests/scientific/
```
*Expected Output*: `200 passed in ~20s` (100% passing).

### Step 2: Multimodal Cohort Assembly
To integrate FreeSurfer 204 IDPs with real PPMI quantitative biofluids (NfL, GFAP, pTau, tTau, A$\beta_{42}$) and SAA kinetics:
```bash
python scripts/phase7/build_multimodal_dataset.py
```
*Expected Output*: Materializes `ppmi_longitudinal_multimodal_biofluid_n341.parquet` (Shape: `(341, 279)`).

### Step 3: Biological Relationship Discovery & Statistical Benchmarking
To execute the complete cross-modal Canonical Correlation Analysis, partial correlations, 4-tier classical control suite, 1,000 bootstrap resamples, 1,000 permutation tests, and FDR correction:
```bash
python scripts/phase7/run_biological_relationship_discovery.py
```
*Expected Output*: Materializes:
- `RELATIONSHIP_DISCOVERY_RESULTS.md`
- `CROSS_MODAL_RELATIONSHIPS.csv`
- `CLASSICAL_CONTROL_COMPARISONS.csv`
- `PATIENT_SUBGROUP_PHENOTYPES.csv`
- `CLINICAL_QUESTIONS_FINDINGS.csv`

### Step 4: End-to-End Pipeline Demonstration
To run a zero-PHI, standalone clinical inference and visualization demonstration:
```bash
python scripts/phase7/demo_nadiq_pipeline.py
```
*Expected Output*: Generates multimodal risk scores, dynamic modality gating attributions, and patient progression trajectories.

---

## 4. Zero Protected Health Information (PHI) Guarantee

In strict compliance with HIPAA and GDPR data privacy mandates:
1. All patient identifiers are anonymized numeric codes (`PATNO`) assigned by PPMI.
2. No participant names, dates of birth, geographic locations, or clinical institution identifiers are ingested or output.
3. All visualization routines and terminal reports output aggregated statistical metrics and anonymized clinical profiles.
4. All computation is executed strictly on local offline storage (`localhost`), with zero telemetry or external API transmission.

