# NĀḌĪ-Q — PHASE 9 REPRODUCIBILITY GUIDE
## Exact Execution Commands, Environment Specifications, and Audit Hashes

> **Document Type**: Scientific Reproducibility Protocol  
> **Phase**: 9 — QVBE Remediation & Architectural Repair  
> **Repository Context**: NĀḌĪ-Q (SIH 2026 Submission SIH26139)

---

# 1. RUNTIME ENVIRONMENT

- **Operating System**: Windows 11 Home x64 (Build 26100)
- **Python Version**: `3.12.4` (64-bit)
- **Core Dependencies**:
  - `pennylane == 0.41.1`
  - `torch == 2.4.1+cpu`
  - `scikit-learn == 1.6.1`
  - `scipy == 1.15.2`
  - `numpy == 2.2.4`
  - `pandas == 2.2.3`
  - `pytest == 9.1.1`

Frozen requirements are archived in `docs/phase9/requirements_frozen.txt`.

---

# 2. INPUT DATA ARTIFACTS

All experiments rely strictly on local, non-uploaded PPMI data artifacts:

| Data Artifact | Path | Format & Dimension | MD5 / SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Multimodal Parquet** | `F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet` | $N=341, D=279$ | `3c099d0eb618fa9dd8e9daffc87dcfc7` |
| **Early Temporal Tensor** | `F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy` | $341 \times 3 \times 35$ | `3df53b1b6e49ecbe804bc6ba6b7f3f33` |

---

# 3. EXACT STEP-BY-STEP REPRODUCIBILITY COMMANDS

Execute all commands from the repository root (`C:\Users\ayush\OneDrive\Desktop\Nadi-Q`):

### Step 1: Execute the Unit Verification Suite (10/10 Tests)
```powershell
python -m pytest tests/phase9/test_qvbe_v3.py -v
```
*Expected Result*: 10 passed in ~30–40 seconds.

### Step 2: Run Synthetic Controlled Diagnostic Experiments
```powershell
python scripts/phase9/run_diagnostic_experiments.py
```
*Expected Outputs*:
- `F:\NADI-Q_AUDIT_OUTPUT\PHASE_9\DIAGNOSTIC_EXPERIMENTS_RESULTS.json`
- `docs/phase9/DIAGNOSTIC_EXPERIMENTS_RESULTS.json`

### Step 3: Run Quantum-Classical Comparison Benchmark (Real PPMI $N=341$)
```powershell
python scripts/phase9/run_quantum_classical_comparison.py
```
*Expected Outputs*:
- `F:\NADI-Q_AUDIT_OUTPUT\PHASE_9\QUANTUM_CLASSICAL_COMPARISON_RESULTS.csv`
- `docs/phase9/QUANTUM_CLASSICAL_COMPARISON_RESULTS.csv`

### Step 4: Run CTCLM Remediation and Handcrafted Benchmark
```powershell
python scripts/phase9/run_ctclm_correction.py
```
*Expected Outputs*:
- `F:\NADI-Q_AUDIT_OUTPUT\PHASE_9\CTCLM_CORRECTION_RESULTS.csv`
- `docs/phase9/CTCLM_CORRECTION_RESULTS.csv`

### Step 5: Run QCDF Multimodal Gating Benchmark
```powershell
python scripts/phase9/run_qcdf_correction.py
```
*Expected Outputs*:
- `F:\NADI-Q_AUDIT_OUTPUT\PHASE_9\QCDF_CORRECTION_RESULTS.csv`
- `docs/phase9/QCDF_CORRECTION_RESULTS.csv`

### Step 6: Run Clean (Target-Leak-Free) Clinical Clustering
```powershell
python scripts/phase9/run_clean_cluster_discovery.py
```
*Expected Outputs*:
- `F:\NADI-Q_AUDIT_OUTPUT\PHASE_9\CLEAN_CLUSTER_RESULTS.csv`
- `docs/phase9/CLEAN_CLUSTER_RESULTS.csv`

---

# 4. RANDOM SEED SPECIFICATIONS

- Data Splitting Seed: `42`
- PyTorch CPU Seed: `42`
- PennyLane Statevector Seed: `42`
- K-Fold Cross-Validation: `42`
- Diagnostic Multi-Seed Tests: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`
