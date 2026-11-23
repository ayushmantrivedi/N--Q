# NĀḌĪ-Q Compatibility and Dependency Risk Analysis

**Audit Date**: November 23, 2026  
**Auditor**: Independent Software Auditor & Systems Architect  
**Platform**: Windows 11 Home x64 (Build 26100)  
**Python Runtime**: 3.12.4 (CPython)  
**Active Stack**: PyTorch 2.4.1+cpu, PennyLane 0.41.1, Scikit-Learn 1.6.1, Hyperon 0.2.10, Pandas 2.2.3, NumPy 2.2.6

---

## 1. Local Environment Constraints & Operating Boundaries

NĀḌĪ-Q operates under five non-negotiable architectural boundaries:
1. **Operating System**: Windows 11 x64 (POSIX-specific shell scripts, Unix sockets, and Linux-only C-extensions are incompatible).
2. **Python Version**: Python 3.12.4 (Strictly incompatible with packages that depend on removed standard library modules such as `distutils` or legacy CPython APIs).
3. **Execution Mode**: CPU-only workstation mode (no dedicated CUDA GPU in target test runner, no physical quantum processor).
4. **Data Isolation**: Restricted participant-level PPMI longitudinal datasets ($N=341$) must remain strictly local on disk (`F:\NADI-Q_AUDIT_OUTPUT` and `F:\NADI-Q data`). Zero data egress to remote APIs, Hugging Face Hub, or cloud clusters.
5. **Stability Policy**: Zero destructive changes to existing environment dependencies; no forced upgrades that break existing passing regression tests (361/361 tests passing).

---

## 2. External Repository Dependency & Compatibility Matrix

| Candidate Repository | Python Target | Key External Dependencies | Windows Compatibility | Python 3.12 Compatibility | Dependency Conflict Risk |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **YAIB** | `3.10 - 3.13` | `gin-config`, `pytorch-ignite`, `pytorch-lightning`, `uv_build`, `polars` | High | Moderate (Pins `torch==2.6.0`, conflicts with local PyTorch 2.4.1) | **High**: Full install attempts to force upgrade PyTorch, Lightning, and install heavy ICU dependencies. |
| **CEHR-BERT** | `>=3.10` | `flash-attn-2`, `transformers`, `deepspeed`, `meds`, `femr`, `dask` | Very Low | Low | **Critical**: `flash-attn-2` requires MSVC C++ compilation on Windows; `deepspeed` has known Windows multiprocessing issues. |
| **HGNN** | `>=3.6` | `torch>=1.0`, `scipy` | High | High | **Low**: Pure PyTorch, but unmaintained since 2019. |
| **DeepHypergraph (DHG)** | `>=3.9` | `torch>=2.1.0`, `scipy>=1.8`, `optuna`, `numpy`, `scikit-learn` | **Verified High** | **Verified High** | **Very Low**: Pure Python/PyTorch with zero compiled C++ extensions. Imports cleanly. |
| **Hyper-SAGNN** | `3.6.8` | `tensorflow<2.0.0`, `gensim`, `torch>=1.0` | Incompatible | Incompatible | **Fatal**: TensorFlow 1.x has no Python 3.12 wheels and cannot be installed. |
| **MHNN** | `3.9` | `torch_scatter`, `torch_sparse`, `rdkit`, `ogb` | Low | Very Low | **High**: `torch_scatter` requires MSVC compilation matching exact PyTorch version; fails on Windows Python 3.12. |
| **GSoC-2024-QGNN** | Unspecified | `torch_geometric`, `pennylane`, `qiskit` | Moderate | Moderate | **Medium**: Requires `torch_geometric`; unlicensed code introduces legal risk. |
| **QuantumGNN Molecules** | Unspecified | `torch`, `numpy` | **Incompatible** (Filesystem path bug) | Moderate | **High**: Dataset filename contains `<` and `|`, causing git checkout to fail on Windows. Contains no quantum code. |
| **PennyLane** | `>=3.10` | `numpy`, `scipy`, `autoray`, `pennylane-lightning` | **Verified High** | **Verified High** | **Zero**: Already installed and verified (`v0.41.1`). |

---

## 3. Detailed Failure Mode Analysis for High-Risk Repositories

### 3.1 Filesystem Failure Mode: `QuantumGNN_molecules`
- **Error Triggered**: `git clone` succeeded, but `git checkout` crashed with:
  ```
  error: invalid path 'dataset/QM9_molecularsize<=15/input/Cv(cal|molK).npy'
  fatal: unable to checkout working tree
  ```
- **Cause**: Windows NTFS strictly prohibits the characters `<` (less than) and `|` (pipe) in file and directory paths.
- **Risk Assessment**: Any workflow attempting to clone or install this repository directly on Windows will abort with fatal git errors unless complex sparse checkouts are configured.

### 3.2 Compilation Failure Mode: `MHNN` (`torch_scatter`)
- **Error Triggered**: `pip install torch-scatter` on Windows Python 3.12 fails unless a full Microsoft Visual C++ Build Tools toolchain with exact CUDA/C++ toolsets is configured.
- **Root Cause**: `torch_scatter` relies on custom C++/CUDA kernel extensions that do not distribute universal pre-built binary wheels for Python 3.12 CPU on Windows.
- **Remediation**: The repository's mathematical operations can be rewritten using standard PyTorch index operations (`torch.Tensor.scatter_add_`), eliminating `torch_scatter` entirely.

### 3.3 Framework Pollution & Dependency Bloat: `YAIB`
- **Issue**: YAIB specifies `requires-python = ">=3.10,<3.14"` and pins `torch==2.6.0`, `pytorch-ignite==0.5.0.post2`, and `pytorch-lightning==2.6.6` in its `pyproject.toml`.
- **Risk Assessment**: Installing the full `yaib` package into the active NĀḌĪ-Q environment would forcibly upgrade PyTorch from 2.4.1 to 2.6.0, risking regression in PennyLane's Torch interface and invalidating previous experiment checkpoints.
- **Remediation**: Do not install YAIB as a package. Extract only the self-contained `GRUDCell` class into NĀḌĪ-Q.

### 3.4 Deep Learning Stack Incompatibility: `CEHR-BERT`
- **Issue**: CEHR-BERT's `pyproject.toml` mandates `flash-attn-2` and `deepspeed`.
- **Risk Assessment**: `flash-attn` has no official Windows CPU wheels and requires custom ninja compilation that fails on standard workstations. Additionally, CEHR-BERT depends on OMOP CDM data conversion tools (`meds`, `femr`) that introduce massive overhead for a 341-patient tabular dataset.

---

## 4. Legal, License, and Provenance Audit

| Repository | Declared License | Commercial / Open-Source Rights | Attribution Requirements | Risk Level |
| :--- | :---: | :--- | :--- | :---: |
| **YAIB** | MIT | Permissive; full commercial and research use | Retain copyright notice | Low |
| **CEHR-BERT** | MIT | Permissive; full commercial and research use | Retain copyright notice | Low |
| **HGNN** | MIT | Permissive; full commercial and research use | Retain copyright notice | Low |
| **DeepHypergraph** | Apache-2.0 | Permissive; explicit patent grant | State changes; retain notice | Low |
| **Hyper-SAGNN** | MIT | Permissive; full commercial and research use | Retain copyright notice | Low |
| **MHNN** | MIT | Permissive; full commercial and research use | Retain copyright notice | Low |
| **GSoC-2024-QGNN** | **NONE** | **UNLICENSED**; all rights reserved by author | Cannot legally redistribute or adopt code | **High** |
| **QuantumGNN Molecules**| Apache-2.0 | Permissive | Retain copyright notice | Low |
| **PennyLane** | Apache-2.0 | Permissive; commercial and academic use | Retain notice | Low |

> [!WARNING]
> **Legal Red Flag on GSoC-2024-QGNN**:
> The repository `Haemanth-V/GSoC-2024-QGNN` contains no `LICENSE` file. Under international copyright law, unlicensed public GitHub repositories remain the exclusive intellectual property of the author with "All Rights Reserved". Direct copying or adoption of code from this repository into NĀḌĪ-Q poses legal and compliance risks for open-source publication.

---

## 5. Recommended Safe Integration Protocols

1. **Pure PyTorch Implementation**: Any external concept to be tested must be implemented using existing in-tree dependencies (`torch>=2.2`, `numpy`, `scipy`, `pandas`).
2. **DHG as a Light Extra**: DeepHypergraph (`dhg`) is the only hypergraph repository that satisfies all Windows, Python 3.12, and zero-compilation constraints. If hypergraph algorithms are evaluated, `dhg` can be installed safely as an optional research extra.
3. **Zero Package-Level Ingestion**: Under no circumstances should `yaib`, `cehrbert`, or `mhnn` be added to `pyproject.toml` dependencies. Their useful algorithmic patterns must be adapted as lightweight, standalone PyTorch modules.
