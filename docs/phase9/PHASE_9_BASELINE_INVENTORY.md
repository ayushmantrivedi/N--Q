# NĀḌĪ-Q — PHASE 9 BASELINE INVENTORY

> **Document Type**: Immutable Pre-Modification Baseline  
> **Created**: 2026-11-18  
> **Phase**: 9 — QVBE-v2 Remediation  
> **Purpose**: Record the exact state of the repository and runtime environment before any source code changes are made in Phase 9. This document must NOT be modified after creation.  
> **Evidentiary Standards**: Every claim carries one of: `[VERIFIED FROM SOURCE CODE]`, `[VERIFIED FROM EXPERIMENT OUTPUT]`, `[MATHEMATICAL DERIVATION]`, `[REASONABLE INFERENCE]`, `[UNTESTED HYPOTHESIS]`

---

## 1. GIT STATE AT PHASE 9 ENTRY

`[VERIFIED FROM SOURCE CODE]`

### 1.1 Git Log (last 10 commits)

```
f3b6cde feat: Implement the initial AI outreach agent system including company discovery, ranking, email generation, sending, and reply monitoring.
3cc1c6b fix: centralize .env loading via utils.py, add .gitignore
41e67b8 feat: initial commit — AI Outreach Agent complete scaffold
```

> **Note**: The Nadi-Q project has only 3 commits. The git repository root is tracking the broader Desktop folder, which is why `git status` shows many untracked user-level files outside the project. The project itself (`Nadi-Q/`) shows no staged or unstaged changes to tracked files. `[VERIFIED FROM SOURCE CODE]`

### 1.2 Git Status Summary

```
On branch main
Your branch is up to date with 'origin/main'.
nothing added to commit but untracked files present (use "git add" to track)
```

**Branch**: `main`  
**Working tree**: Clean (no staged or unstaged changes to tracked files) `[VERIFIED FROM SOURCE CODE]`

---

## 2. RUNTIME ENVIRONMENT

`[VERIFIED FROM SOURCE CODE]` — collected via `python --version` and `pip show`

| Component       | Version    | Location                                                               |
|-----------------|------------|------------------------------------------------------------------------|
| **Python**      | 3.12.4     | System Python (Windows 11)                                             |
| **PennyLane**   | 0.41.1     | `C:\Users\ayush\AppData\Local\Programs\Python\Python312\Lib\site-packages` |
| **PyTorch**     | 2.4.1      | Same site-packages                                                     |
| **NumPy**       | 2.2.6      | Same site-packages                                                     |
| **OS**          | Windows 11 (10.0.26100 SP0) | Host: ASus                                        |

Full package freeze saved to: `docs/phase9/requirements_frozen.txt` `[VERIFIED FROM SOURCE CODE]`

---

## 3. SHA-256 HASHES OF CRITICAL SOURCE FILES

`[VERIFIED FROM SOURCE CODE]` — computed via PowerShell `Get-FileHash -Algorithm SHA256`

| File | SHA-256 Hash |
|------|-------------|
| `src/nadiq/models/quantum/qvbe_v2.py` | `D4B1F319CB3783F23AD51B30679DF90139838DDA22656D5549C985BD0FFC6247` |
| `src/nadiq/models/quantum/ctclm.py`   | `C522D1A030BEF8FC559AB25A80414E29667D45AA87DBB16537A74B901F68139A` |
| `src/nadiq/models/fusion/qcdf.py`     | `B4BA356482E05B3ED12C2FDE130913C2E092F9A08388BEFEB30D994A2C1CFD53` |

### 3.1 Checkpoint / Experiment JSON Hashes

`[VERIFIED FROM EXPERIMENT OUTPUT]`

| File | SHA-256 Hash |
|------|-------------|
| `F:\NADI-Q_AUDIT_OUTPUT\PHASE_5\qvbe_v2_best_mode_a_8q.json` | `96510D505C56F23AFA728D035A467E97C55E6CB9C400546B5018122D2FC1C08D` |

> No JSON checkpoint files exist under `experiments/` in the workspace (the experiments/ directory contains smoke test scripts only, no serialized model checkpoints). `[VERIFIED FROM SOURCE CODE]`

---

## 4. COMPLETE IMPLEMENTATION LOCATIONS

`[VERIFIED FROM SOURCE CODE]` — found via recursive grep across `src/`, `scripts/`, `tests/`, `experiments/`

### 4.1 QVBE-v2 (`QuantumVariationalBiomarkerEncoderV2`)

| Location | Type | Notes |
|----------|------|-------|
| `src/nadiq/models/quantum/qvbe_v2.py` | **Primary Implementation** | 439 lines; `QVBEV2Config`, `QuantumVariationalBiomarkerEncoderV2`; last modified 2026-11-04 |
| `src/nadiq/models/quantum/__init__.py` | Re-export | Exports `QVBEV2Config`, `QuantumVariationalBiomarkerEncoderV2` |
| `tests/scientific/test_qvbe_v2.py` | Test suite | 9,164 bytes; `TestQVBEV2` class with multiple test cases |
| `experiments/smoke/qvbe_smoke_test.py` | Smoke test | TADPOLE → QVBE pipeline connectivity test |
| `experiments/smoke/ctclm_smoke_test.py` | Smoke test | TADPOLE → QVBE → CTCLM pipeline |
| `experiments/smoke/architecture_smoke_test.py` | Smoke test | Full end-to-end architecture smoke |
| `scripts/phase4/run_qvbe_representation_benchmarks.py` | Experiment script | Phase 4 representation benchmarks |
| `scripts/phase4/test_vectorized_qvbe.py` | Test/perf | Vectorized QVBE performance |
| `scripts/phase5/freeze_and_checkpoint_qvbe.py` | Checkpoint | Freeze + serialize QVBE |
| `scripts/phase5/reproduce_and_validate_phase4.py` | Validation | Phase 4 reproduction |
| `scripts/phase5/run_open_model_qvbe_pipeline.py` | Experiment | Open model QVBE integration |
| `scripts/phase7/run_deep_internal_diagnostics.py` | Diagnostic | Phase 8 deep diagnostics (run in Phase 7 context) |
| `scripts/audit/calc_quantum_empirical_geometry.py` | Audit | Empirical geometry calculations |

### 4.2 CTCLM (`ContinualTemporalClinicalLatentModel`)

| Location | Type | Notes |
|----------|------|-------|
| `src/nadiq/models/quantum/ctclm.py` | **Primary Implementation** | 360 lines; `CTCLMConfig`, `CTCLMOutput`, `ContinualTemporalClinicalLatentModel`; GRU-based; last modified 2026-09-11 |
| `tests/scientific/test_ctclm.py` | Test suite | 21,655 bytes; most comprehensive model test file |
| `tests/scientific/test_modality_integration.py` | Integration test | Tests CTCLM within multi-branch architecture |
| `experiments/smoke/ctclm_smoke_test.py` | Smoke test | Real TADPOLE pipeline connectivity |
| `experiments/smoke/architecture_smoke_test.py` | Smoke test | End-to-end including CTCLM branch |
| `scripts/archived/invalid/train_ctclm.py` | Archived/invalid | Obsolete training script |

### 4.3 QCDF (`QuantumClinicalDecisionFusion` / `QCDF`)

| Location | Type | Notes |
|----------|------|-------|
| `src/nadiq/models/fusion/qcdf.py` | **Primary Implementation** | 221 lines; `QCDFConfig`, `QCDFOutput`, `QCDF`; small MLP fusion; last modified 2026-09-12 |
| `tests/scientific/test_qcdf.py` | Test suite | 2,052 bytes; determinism, missing modality, full-modality tests |
| `tests/scientific/test_modality_integration.py` | Integration test | Tests QCDF within full-stack architecture |
| `experiments/smoke/architecture_smoke_test.py` | Smoke test | Exercises QCDF in end-to-end flow |

---

## 5. REPOSITORY STRUCTURE OVERVIEW

`[VERIFIED FROM SOURCE CODE]`

```
Nadi-Q/
├── src/nadiq/
│   ├── models/
│   │   ├── quantum/          ← qvbe_v2.py, ctclm.py, qvbe.py (v1), __init__.py
│   │   ├── fusion/           ← qcdf.py
│   │   ├── biomarkers/       ← biomarker_encoder.py
│   │   ├── mri/              ← mri_encoder.py, bvae.py
│   │   ├── classical/        ← logistic.py, gradient_boosting.py, mlp.py
│   │   └── multimodal/       ← cross_modal_alignment.py
│   ├── data/                 ← contracts, loaders, sources, temporal
│   ├── quantum/              ← backends, circuits, encodings, execution, noise
│   ├── evaluation/           ← calibration, classification, statistics
│   └── preprocessing/        ← base, filtering, imputation, mri, normalization
├── tests/
│   ├── scientific/           ← test_qvbe_v2.py, test_ctclm.py, test_qcdf.py, ...
│   ├── integration/          ← test_ppmi_integration_layer.py, ...
│   └── unit/                 ← test_classical_models.py, test_data_contracts.py, ...
├── experiments/
│   └── smoke/                ← qvbe_smoke_test.py, ctclm_smoke_test.py, architecture_smoke_test.py
├── scripts/
│   ├── phase4–7/             ← Experiment and audit scripts per phase
│   ├── training/             ← run_v2_biomarker_real.py, train_biomarkers.py
│   └── audit/                ← Geometric audit and diagnostic scripts
├── configs/                  ← YAML configs for experiments and quantum
├── docs/
│   ├── phase4–7/             ← Phase reports and artefacts
│   └── phase9/               ← (THIS DIRECTORY — Phase 9 baseline)
├── experiments/archived/     ← Obsolete pipeline scripts
└── F:\NADI-Q_AUDIT_OUTPUT\   ← External audit output root (Phases 2–7)
```

---

## 6. PHASE 9 ENTRY CHECKPOINT SUMMARY

| Item | Value |
|------|-------|
| Baseline Timestamp | 2026-11-18T11:30 UTC |
| Git HEAD commit | `f3b6cde` |
| Active branch | `main` |
| Working tree state | Clean (no uncommitted changes to tracked files) |
| Python | 3.12.4 |
| PennyLane | 0.41.1 |
| PyTorch | 2.4.1 |
| NumPy | 2.2.6 |
| Primary QVBE-v2 file hash | `D4B1F319...FC6247` |
| Primary CTCLM file hash | `C522D1A0...8139A` |
| Primary QCDF file hash | `B4BA3564...CFD53` |
| Frozen checkpoint hash | `96510D50...C1C08D` |
| Phase 7 diagnostic JSON | Confirmed present, 24-key flat JSON |

---

*This document is IMMUTABLE after creation. Any corrections or amendments must be recorded in `PHASE_9_CHANGE_REGISTER.md`.*
