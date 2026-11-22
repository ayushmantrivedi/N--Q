# NĀḌĪ-Q — COMPLETE REPOSITORY INVENTORY & CLASSIFICATION AUDIT
**Phase 11: Repository Restructuring, Packaging & GitHub Readiness**  
**Lead Systems Architect & Research Maintainer Audit**  
**Audit Date**: November 2026 | **Host Environment**: Windows 11 Home x64 / Python 3.12.4 / PennyLane 0.41.1 / PyTorch 2.4.1+cpu  
**Workspace**: `c:\Users\ayush\OneDrive\Desktop\Nadi-Q` | **Protected Research Storage**: `F:\NADI-Q data\`, `F:\NADI-Q_AUDIT_OUTPUT\`  
**Test Suite Verification**: **313 / 313 unit, integration, and scientific tests passing (100%)**

---

## 1. Executive Summary & Disk Usage Inventory

The NĀḌĪ-Q repository is an experimental hybrid quantum-classical machine learning research platform for multimodal early neurological disease-risk detection and longitudinal progression forecasting. Over eleven iterative research phases (Phase 1 through Phase 10B, followed by Phase 11 cleanup), the codebase has accumulated core variational quantum circuits (QVBE-v2, QVBE-v3), classical temporal learning models (CTCLM), fusion decoders (QCDF), time-conditioned quantum hypergraph architectures (TQ-RHNet), symbolic knowledge reasoners (Hyperon / MeTTa), forensic audits, benchmark reports, and extensive unit/scientific test suites.

### Directory Storage and File Distribution Summary

| Directory Path | Role / Domain | File Count | Disk Usage (Bytes) | Disk Usage (MB) | Category Alignment |
|:---|:---|:---:|:---:|:---:|:---|
| `src/nadiq/` | Core Python Package (src-layout) | 164 | 737,729 | 0.70 MB | Category A (Core) |
| `tests/` | Unit, Integration & Scientific Test Suites | 61 | 816,828 | 0.78 MB | Category A (Core) |
| `configs/` | YAML Configurations (Data, Models, Runs) | 9 | 11,653 | 0.01 MB | Category A / B |
| `scripts/` | Historical Phase Scripts (Phases 2–9, 10A, 10B) | 130 | 920,249 | 0.88 MB | Category B (Experiments) |
| `experiments/` | Formal Experiment Runners & Results | 23 | 114,729 | 0.11 MB | Category B / C |
| `docs/` | Architectural, Forensic & Phase Master Reports | 100 | 853,005 | 0.81 MB | Category C (Records) |
| `reports/` | Historical & Current Technical Audits | 38 | 350,087 | 0.33 MB | Category C (Records) |
| `results/` | Benchmark JSON Metrics & Historical Outputs | 30 | 152,966 | 0.15 MB | Category C / E |
| `data/` | Data Access Contracts (README / Schemas) | 1 | 1,049 | 0.00 MB | Category A / E |
| `notebooks/` | Interactive Exploratory Notebook Stubs | 1 | 819 | 0.00 MB | Category B / C |
| `scratch/` | Temporary Developer Scripts (Local Only) | 27 | 84,724 | 0.08 MB | Category D / Ignored |
| `(root)` | Packaging, Configuration, Root Metadata | 11 | 68,955 | 0.07 MB | Category A / E |
| `.pytest_cache/` | Ephemeral Pytest Cache (Local Runtime) | 4 | 28,775 | 0.03 MB | Category D (Ignored) |
| **TOTAL** | **Entire Workspace (Post-Cleanup)** | **599** | **4,141,568** | **3.95 MB** | **All Categories** |

*Note: Large participant datasets (PPMI tabular CSVs, raw DICOM/NIfTI scans, 3D MRI tensors totaling ~400+ MB) and binary checkpoints are permanently secured on external protected storage (`F:\NADI-Q data\` and `F:\NADI-Q_AUDIT_OUTPUT\`) and strictly excluded from the git repository.*

---

## 2. Five-Tier System Classification (Categories A through E)

Every file and directory in the repository is mapped to an authoritative categorization tier to govern retention, packaging, and git-tracking.

### Category A — Core Implementation (Production & Active System)
*Files strictly required to import, instantiate, execute, and verify the active NĀḌĪ-Q platform.*

- **Package Root & CLI**:
  - `src/nadiq/__init__.py`: Package version (`0.1.0.dev0`) and top-level public API.
  - `src/nadiq/cli.py`: Click/Typer-based command-line interface entry point.
- **Quantum Machine Learning Modules (`src/nadiq/models/quantum/`)**:
  - `qvbe.py`: Initial Quantum Variational Biomarker Encoder (QVBE-v1).
  - `qvbe_v2.py`: Validated 8-qubit strongly entangling QVBE-v2 with hardware-efficient ansatz.
  - `ctclm.py`: Continual Temporal-Clinical Learning Module with ODE/GRU-based hidden state progression.
  - `__init__.py`: Unified quantum model public exports.
- **QVBE-v3 Mathematical Correction Engine (`src/nadiq/models/qvbe_v3/`)**:
  - `circuit.py`: Corrected Hamiltonian-variational and strongly-entangling parameterized quantum circuits.
  - `objectives.py`: Multi-objective loss formulation (Fubini-Study metric regularization, contrastive alignment, reconstruction).
  - `trainer.py`: Gradient-based variational optimizer with gradient clipping and shot-noise modeling.
  - `config.py`: Strictly validated dataclass hyperparameter configurations.
  - `diagnostics.py`: Quantum expressibility, entangling capability, and barren plateau diagnostics.
  - `classical_controls.py`: Paired classical baseline encoders (PCA, Beta-VAE, MLP autoencoders).
- **Temporal Hypergraph & TQ-RHNet Engine (`src/nadiq/models/hypergraph/`)**:
  - `temporal_hypergraph_builder.py`: Multi-relational incidence hypergraph constructor.
  - `time_conditioned_quantum_hyperedge.py`: Parameterized PennyLane $R_{ZZ}$ and $R_{ZZZ}$ multi-qubit hyperedge encoder.
  - `quantum_relational_drift.py`: Quantum state overlap fidelity ($F_Q = |\langle \psi_{t-1} | \psi_t \rangle|^2$) and drift ($D_Q = 1 - F_Q$).
  - `incidence_hypergraph_message_passing.py`: Incidence-aware bipartite message passing with non-incident preservation.
  - `hypergraph_message_passing.py`: Standard hypergraph neural network propagation layer.
  - `temporal_state_updater.py`: Recurrent temporal state updater combining hyperedge context and node observations.
  - `event_chain_extractor.py`: Temporal extraction of biomarker elevation, structural atrophy, and motor worsening chains.
  - `metta_event_reasoner.py`: Hyperon / OpenCog MeTTa symbolic reasoner integrating formal clinical disease rules.
- **Classical Biomedical Models (`src/nadiq/models/`)**:
  - `fusion/qcdf.py`: Quantum-Classical Diagnostic Fusion network projecting multimodal representations into joint risk space.
  - `biomarkers/biomarker_encoder.py`: Tabular biomarker transformation and clinical normalization.
  - `mri/bvae.py`: 3D Convolutional $\beta$-Variational Autoencoder for volumetric MRI latent extraction.
  - `classical/gradient_boosting.py`, `logistic.py`, `mlp.py`: Formal prognostic baseline models.
- **Preprocessing & Data Access Layer (`src/nadiq/preprocessing/`, `src/nadiq/data/`)**:
  - `preprocessing/pipeline.py`: Leakage-free preprocessing pipeline.
  - `preprocessing/imputation.py`, `normalization.py`, `filtering.py`: Median imputers and Z-score transforms.
  - `data/contracts.py`: Strict schema validation contracts (`ClinicalSample`, `ModalityRepresentation`).
  - `data/patient_timeline.py`: Multi-visit temporal alignment and observation-age tracking.
  - `data/modality_assembler.py`: Missing-modality assembler with explicit availability masks.
- **Evaluation & Statistical Testing (`src/nadiq/evaluation/`)**:
  - `classification/metrics.py`: AUROC, AUPRC, Brier score, ECE calibration error.
  - `statistics/delong.py`: DeLong non-parametric test for comparing paired ROC curves.
  - `statistics/bootstrap.py`: Stratified bootstrap confidence interval estimators.
- **Packaging & Testing Infrastructure**:
  - `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitattributes`, `.gitignore`.
  - `tests/unit/`, `tests/scientific/`, `tests/integration/`, `tests/phase9/`, `tests/conftest.py`.

---

### Category B — Research Experiments (Preserved Experimental Code)
*Complete, executable historical and exploratory scripts across all research phases.*

- **Phase 2 Research Scripts (`scripts/phase2/` — 26 files)**:
  - Tabular schema audits, SAA protein seed aggregation assay quantification, missingness analyses, Pennylane circuit benchmarking.
- **Phase 3 Kernel Benchmarks (`scripts/phase3/` — 7 files)**:
  - Benchmark dataset assembler, quantum kernel evaluation, ablation experiments, statistical significance testing.
- **Phase 4 Representation Learning (`scripts/phase4/` — 9 files)**:
  - Representation extraction, model registry generator, exploratory dimension audits.
- **Phase 5 QVBE Verification (`scripts/phase5/` — 7 files)**:
  - Finite-shot noise robustness simulations, open model pipeline integration, QVBE freezing scripts.
- **Phase 6 Integrated 5-Year Prediction (`scripts/phase6/` — 4 files)**:
  - Feasibility audits, longitudinal data contract verification, integrated benchmark execution.
- **Phase 7 Biological Relationship Discovery (`scripts/phase7/` — 6 files)**:
  - Multimodal dataset builders, CCA cross-modal discovery, deep internal diagnostics suite.
- **Phase 9 Mathematical Correction Experiments (`scripts/phase9/` — 5 files)**:
  - `run_diagnostic_experiments.py`: Expressibility and gradient audits.
  - `run_ctclm_correction.py`: Functional progression model benchmarks.
  - `run_qcdf_correction.py`: Calibrated fusion benchmarks.
  - `run_quantum_classical_comparison.py`: Paired rigorous quantum vs. classical baselines.
  - `run_clean_cluster_discovery.py`: Phenotype cluster discovery.
- **Phase 10A & 10B Hypergraph Experiments (`scripts/experiments/`)**:
  - `phase10a_quantum_temporal_hypergraph.py`: Static quantum hyperedge pilot.
  - `phase10b_tqrhnet.py`: Master benchmark runner for Time-Conditioned Quantum Relational Hypergraph Network.
- **Smoke Tests & Exploratory Scripts (`experiments/smoke/`, `experiments/archived/`)**:
  - Quick sanity checks (`architecture_smoke_test.py`, `ctclm_smoke_test.py`, `qvbe_smoke_test.py`).

---

### Category C — Research Records (Audits, Findings & Metrics)
*Permanent aggregate scientific outputs, methodology records, and architecture decision logs.*

- **Phase Master Reports**:
  - `docs/phase4/PHASE_4_MASTER_REPORT.md`
  - `docs/phase5/PHASE_5_MASTER_REPORT.md`
  - `docs/phase6/PHASE_6_MASTER_REPORT.md`
  - `docs/phase7/PHASE_7_MASTER_REPORT.md`
  - `docs/phase9/PHASE_9_MASTER_REPORT.md`
  - `experiments/phase10a_quantum_temporal_hypergraph/PHASE10A_EXPERIMENT_VERDICT.md`
  - `experiments/phase10b_tqrhnet/PHASE10B_VERDICT.md`
- **Empirical Results Data (Aggregate Metrics — No Raw PHI)**:
  - `experiments/phase10b_tqrhnet/TQRHNET_RESULTS.csv` (Main 5-fold cross-validation metrics across all models).
  - `experiments/phase10b_tqrhnet/TQRHNET_ABLATION_RESULTS.csv` (Drift and 3-body hyperedge ablations).
  - `experiments/phase10b_tqrhnet/TQRHNET_NODE_DRIFT_RESULTS.json` (Fidelity and drift differentials).
  - `experiments/phase10b_tqrhnet/DYNAMIC_LANDMARK_PREDICTIONS.csv` (Landmark prediction calibration).
  - `experiments/phase10b_tqrhnet/EVENT_CHAINS_RESULTS.json` (Extracted clinical event sequences).
  - `docs/phase9/QUANTUM_CLASSICAL_COMPARISON_RESULTS.csv`
  - `docs/phase9/CTCLM_CORRECTION_RESULTS.csv`
  - `docs/phase9/QCDF_CORRECTION_RESULTS.csv`
- **Technical Audits & Data Governance**:
  - `reports/current/` (29 comprehensive technical reports detailing PPMI accession, MRI quality control, and data provenance).
  - `reports/historical/` (Early provenance and environment records).

---

### Category D — Generated Artifacts (Cleaned & Ignored)
*Ephemeral build outputs, bytecode caches, and temporary scratch files.*

- **Python Bytecode**: All `*.pyc` files (116 files unlinked; 1,192.3 KB reclaimed).
- **Bytecode Cache Directories**: All `__pycache__` directories across the workspace (35 directories purged).
- **Test Caches**: `.pytest_cache/` (cleared and added to `.gitignore`).
- **Abandoned Run Dirs**: Five empty run directories in `results/20260910_*` from aborted September 2026 runs removed.
- **Scratch Scripts**: Developer utilities in `scratch/` indexed and strictly excluded via `.gitignore`.

---

### Category E — Sensitive, Non-Portable & Restricted Data (Protected & Excluded)
*Files containing participant identifiers, raw clinical records, or private paths. These MUST NEVER be uploaded to a public repository.*

- **Participant Manifests with Identifiers**:
  - `docs/phase6/PARTICIPANT_SPLIT_MANIFEST_PHASE6.csv` (Contains PPMI `PATNO` participant numbers and 5-year progression outcomes).
  - `ADNI_3T_T1_candidates.csv` in root (Contains ADNI subject identifiers, study UIDs, and acquisition dates).
  - *Action*: Preserved in local protected research storage; strictly excluded from Git tracking via `.gitignore`.
- **Historical Prediction Dumps**:
  - `results/20260910_123247_857187/predictions.parquet`
  - `results/tadpole_20260910_222018_727e1c/*/predictions_*.parquet`
  - *Action*: Preserved locally for audit traceability; strictly excluded via `*.parquet` in `.gitignore`.
- **External Protected Storage**:
  - `F:\NADI-Q data\`: Raw PPMI tabular clinical CSVs, raw DICOM directories, extracted NIfTI volumes, participant 3D MRI tensors (`*_BL_t1_tensor.npy`), and PyTorch model checkpoints (`*.pt`).
  - `F:\NADI-Q_AUDIT_OUTPUT\`: External mirrors of forensic audit artifacts and temporal sequences (`ctclm_early_temporal_sequences_*.npy`).
  - *Action*: Remains untouched, intact, and outside the repository directory.

---

## 3. Structural Reorganization & Bug Fixes Executed

1. **Cross-Platform Text Encoding Standardization**:
   - Discovered 16 `__init__.py` files across `src/nadiq/` that contained Windows cp1252 byte `\x97` (em-dash) instead of standard UTF-8. Converted all 16 files to pure UTF-8 (`—`).
   - Converted `final_tree.txt` and `docs/phase9/requirements_frozen.txt` from UTF-16-LE with BOM (`\xff\xfe`) to standard UTF-8 without BOM.
   - Result: Repository has **0 non-UTF-8 text files**, ensuring flawless cross-platform execution on Linux CI/CD, macOS, and Windows.
2. **Unified Model Exports & Alias Resolution**:
   - Upgraded `src/nadiq/models/fusion/__init__.py` to export `QCDF`, `QuantumClinicalDecisionFusion`, `QCDFConfig`, and `QCDFOutput`.
   - Upgraded `src/nadiq/models/quantum/__init__.py` to re-export `QCDF`, resolving historical import discrepancies between `nadiq.models.quantum` and `nadiq.models.fusion`.
3. **Empty Directory Normalization**:
   - Removed 12 empty abandoned directories in `results/`.
   - Placed `.gitkeep` markers in 8 standard architectural directories (`configs/models`, `experiments/multimodal`, `experiments/unimodal`, `scripts/analysis`, `scripts/utilities`, `tests/archived`, `docs/archived`, `docs/datasets`) to preserve repository structural integrity in Git.
4. **Missing Content Restoration**:
   - Identified 0-byte file `reports/current/NADIQ_REAL_MULTIMODAL_PROTOTYPE_V1.md` and restored its full 3,922-byte content from the verified research mirror on `F:\NADI-Q data\PPMI\reports\`.
5. **Hypergraph Test Suite Expansion**:
   - Created `tests/unit/test_hypergraph.py` with 5 automated unit tests covering `HypergraphNode`, `Hyperedge`, `TimeConditionedQuantumHyperedgeEncoder`, `QuantumRelationalDriftCalculator`, `IncidenceHypergraphMessagePassingLayer`, and `EventChainExtractor`.
   - All 5 newly added tests execute deterministically and pass.
6. **Full Test Suite Verification**:
   - Executed `python -m pytest` across the entire repository.
   - **Baseline: 308 passed -> Post-cleanup: 313 passed (100% pass rate, 0 failures, 0 errors)**.

---

## 4. Production Packaging Deliverables

- **`requirements.txt`**: Curated, pinned-minimum dependencies for runtime research deployment (NumPy, SciPy, Pandas, PyArrow, Scikit-Learn, PyTorch, PennyLane, PennyLane-Lightning, PyYAML, OmegaConf, MLflow, Loguru, Hyperon).
- **`requirements-dev.txt`**: Complete development, testing, and linting environment (PyTest, PyTest-Cov, Black, Ruff, Mypy, Pre-commit).
- **`.gitattributes`**: Line-ending normalization (`* text=auto eol=lf`), explicit syntax attributes (`*.py diff=python`), and binary declarations.
- **`.gitignore`**: Production-grade filter blocking all Category D generated artifacts, Category E sensitive files, private credentials, raw biomedical scans, model weights, and participant-level CSV/parquet manifests.
- **`README.md`**: Modernized to accurately present the completed, empirically validated platform, honest scientific findings (including classical benchmark superiority over TQ-RHNet), and exact reproduction commands.
