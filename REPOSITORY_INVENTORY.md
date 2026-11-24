# NĀḌĪ-Q — Complete Repository Inventory & Classification Audit

**Phase / Version**: Architecture A (Production MVP)  
**Lead Systems Architect & Research Maintainer Audit**  
**Audit Date**: November 2026 | **Host Environment**: Windows 11 Home x64 / Python 3.12.4 / PennyLane 0.41.1 / PyTorch 2.4.1+cpu  
**Workspace**: `c:\Users\ayush\OneDrive\Desktop\Nadi-Q` | **Protected Research Storage**: `F:\NADI-Q data\`, `F:\NADI-Q_AUDIT_OUTPUT\`  
**Test Suite Verification**: **413 / 413 unit, integration, and scientific tests passing (100% Pass, 0 Failures)**

---

## 1. Executive Summary & Disk Usage Inventory

The NĀḌĪ-Q codebase is an experimental hybrid quantum-classical machine learning platform for multimodal early neurodegenerative disease detection and longitudinal progression forecasting. Over fourteen research phases culminating in Architecture A, the repository implements:
- **Core Variational Quantum Circuits**: QVBE-v1 through QVBE-v4 (8-qubit hardware-efficient ansatz, Fubini-Study regularization, contrastive embedding).
- **Longitudinal Temporal Learning**: Continual Temporal-Clinical Learning Module (CTCLM) modeling prospective 5-year trajectories.
- **Multimodal Diagnostic Fusion**: Quantum-Classical Diagnostic Fusion (QCDF) with adaptive attention gating ($\alpha_q$ vs $\alpha_t$).
- **Biological Knowledge Representation**: OpenCog Hyperon MeTTa symbolic compiler extracting clinical progression constraints.
- **Combinatorial Optimization**: Quadratic Unconstrained Binary Optimization (QUBO) builder solved via Classical Tabu Search ($M_3$, 127 ms) and PennyLane QAOA ($p=1, 2$).
- **Two-Tier Adaptive FDR**: Benjamini–Hochberg statistical control ($q \le 0.05$ Validated, $q \le 0.10$ Provisional) with minimum detectable effect (MDE) power analysis.
- **User Interfaces & Production Tools**: Full Streamlit graphical application (`app.py`), multimodal patient predictor CLI (`scripts/predict_patient.py`), and end-to-end benchmark suites.

### Directory Storage and File Distribution Summary

| Directory Path | Role / Domain | File Count | Category Alignment | Description |
|:---|:---|:---:|:---|:---|
| `src/nadiq/` | Core Python Package (src-layout) | ~175 | Category A (Core) | Modular library code: `architecture_a/`, `models/`, `preprocessing/`, `evaluation/`, `quantum/`, `data/`. |
| `tests/` | Test Suites | ~65 | Category A (Core) | 413 unit, integration, and scientific tests across all modules. |
| `configs/` | YAML Configurations | 9 | Category A / B | Data, model, and experiment configuration files. |
| `scripts/` | Active Tools & Historical Scripts | ~135 | Category A / B | `predict_patient.py`, `nadiq_discover.py`, `nadiq_predict.py`, `run_architecture_a_benchmark.py`, plus historical phases. |
| `experiments/` | Experiment Runners & Artifacts | ~30 | Category B / C | Authoritative Architecture A artifacts, Phase 10A/10B verdicts, checkpoints. |
| `docs/` | Architecture, Data & Phase Reports | ~105 | Category C (Records) | System architecture, quantum design, data contracts, and Phase 4–9 master reports. |
| `reports/` | Technical & Historical Audits | ~40 | Category C (Records) | PPMI acquisition audits, provenance records, and forensic reviews. |
| `research/` | Deep Audits & Benchmarks | ~20 | Category C (Records) | Pre-implementation benchmarks, repository audit verdicts, forensic analyses. |
| `data/` | Data Access Contracts | 2 | Category A / E | README and schema definitions. Zero PHI or raw CSVs committed. |
| `notebooks/` | Exploratory Notebook Stubs | 1 | Category B / C | Exploratory guidelines and template stubs. |
| `(root)` | Packaging, App & Metadata | 12 | Category A / E | `app.py`, `README.md`, `PROJECT_STATE.md`, `REPOSITORY_INVENTORY.md`, `pyproject.toml`, etc. |

*Note: All participant-level clinical datasets (PPMI tabular CSVs, DICOM/NIfTI scans totaling ~400+ MB) are strictly secured on local protected storage and permanently excluded from git tracking.*

---

## 2. Five-Tier System Classification (Categories A through E)

Every file and directory in the repository is mapped to an authoritative categorization tier to govern retention, packaging, and git-tracking.

### Category A — Core Implementation (Production & Active System)
*Files strictly required to import, instantiate, execute, and verify the active NĀḌĪ-Q platform.*

- **Package Root & Entry Points**:
  - `app.py`: Streamlit interactive web application with 3 production tabs.
  - `src/nadiq/__init__.py`: Package version and top-level public API exports.
  - `src/nadiq/cli.py`: Command-line interface entry point.
- **Architecture A Package (`src/nadiq/architecture_a/`)**:
  - `pipeline.py`: End-to-end pipeline coordinating discovery (Plane 1) and clinical prediction (Plane 2).
  - `evidence_discovery.py`: Biological evidence discovery pipeline with correlation extraction, MeTTa compilation, and QUBO solving.
  - `optimization_engine.py`: `DualSolverEngine` supporting Tabu Search (default), Exact search, and PennyLane QAOA with 15s timeout fallback.
  - `fdr.py`: Two-tier adaptive Benjamini–Hochberg FDR screening ($q \le 0.05$ Validated, $q \le 0.10$ Provisional) + MDE power analysis.
  - `clinical_predictor.py`: Purely classical prospective clinical predictor for Task T2; zero quantum imports; $\Delta_{\max} = 0.00\text{e}{+}00$.
  - `evidence_exporter.py`: Serializer for evidence graph JSON and Hyperon MeTTa AtomSpace triplets.
  - `__init__.py`: Package exports for Architecture A.
- **Integrated Multimodal Architecture (`src/nadiq/models/integrated/`)**:
  - `model.py`: `NadiQIntegratedModel` combining QVBE-v4, CTCLM, and Adaptive QCDF fusion.
  - `config.py`: `NadiQIntegratedConfig` dataclass parameters.
  - `explainability.py`: `NadiQXAIExplainer` computing counterfactual attributions ($\Delta P_q, \Delta P_t$) and modality breakdown.
- **Quantum Machine Learning Modules (`src/nadiq/models/quantum/`, `qvbe_v3/`, `qvbe_v4/`)**:
  - `qvbe_v4/`: 8-qubit hardware-efficient parameterized quantum biomarker circuit with Fubini-Study regularization.
  - `quantum/ctclm.py`: Continual Temporal-Clinical Learning Module with ODE/GRU-based hidden state progression.
  - `quantum/qvbe_v2.py`: 8-qubit strongly entangling ansatz.
  - `qvbe_v3/`: Hamiltonian-variational quantum circuits and expressibility diagnostics.
- **Symbolic Reasoning & Graph Optimization (`src/nadiq/models/metta_quantum/`, `hypergraph/`)**:
  - `metta_quantum/qubo_compiler.py`: Compiles MeTTa biological constraints into QUBO matrices.
  - `metta_quantum/classical_solvers.py`: High-performance Tabu search, simulated annealing, and greedy solvers.
  - `metta_quantum/qaoa_solver.py`: PennyLane parameterized QAOA circuit optimizer.
  - `hypergraph/metta_event_reasoner.py`: Hyperon / OpenCog MeTTa clinical progression reasoner.
- **Classical Biomedical Models & Preprocessing (`src/nadiq/models/`, `preprocessing/`, `data/`)**:
  - `fusion/qcdf.py`: Quantum-Classical Diagnostic Fusion network with missingness indicator masks.
  - `biomarkers/biomarker_encoder.py`: Tabular biomarker transformation and clinical normalization.
  - `mri/bvae.py`: 3D Convolutional $\beta$-Variational Autoencoder for volumetric MRI latent extraction.
  - `classical/`: Logistic regression, MLP, and Gradient Boosting prognostic baselines.
  - `preprocessing/`: Leakage-free median imputation, Z-score normalization, and cohort filtering.
  - `data/contracts.py`: Strict schema validation contracts (`ClinicalSample`, `ModalityRepresentation`).
- **Evaluation & Statistical Testing (`src/nadiq/evaluation/`)**:
  - `classification/metrics.py`: AUROC, AUPRC, Brier score, ECE calibration error.
  - `statistics/delong.py`: DeLong non-parametric test for paired ROC curves.
  - `statistics/bootstrap.py`: Stratified bootstrap confidence interval estimators.
- **Packaging & Testing Infrastructure**:
  - `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.gitattributes`.
  - All sub-suites in `tests/`: `tests/unit/test_app.py`, `test_architecture_a.py`, `test_integrated_system.py`, `test_metta_quantum.py`, `tests/scientific/`, `tests/data/`, `tests/integration/`, `tests/phase14a/`, `tests/phase9/`.

---

### Category B — Active Tools & Research Experiments
*Executable command-line tools and reproducible experiment scripts.*

- **Active Production Tools (`scripts/`)**:
  - `scripts/predict_patient.py`: Multimodal patient prediction tool (Biomarkers, MRI, Motor $\rightarrow$ Quantum + Classical ML).
  - `scripts/nadiq_discover.py`: Plane 1 biological evidence discovery CLI.
  - `scripts/nadiq_predict.py`: Plane 2 decoupled clinical prediction CLI.
  - `scripts/run_architecture_a_benchmark.py`: End-to-end benchmark verification suite.
- **Historical Phase Scripts (`scripts/phase2/` through `phase14a/`)**:
  - Preserved historical research scripts across Phases 2, 3, 4, 5, 6, 7, 9, 10A, 10B, and 14A for scientific provenance.

---

### Category C — Documentation, Audits & Master Reports
*Audited scientific records, architecture documentation, and benchmark results.*

- **Architecture Documentation (`docs/architecture/`)**:
  - `architecture.md`: System philosophy, two-plane decoupling, UI workflows, and module topology.
  - `quantum_design.md`: Circuit specifications, NISQ noise modeling, and quantum invariance proofs.
- **Research Documentation (`docs/research/`)**:
  - `research_questions.md`: Primary and secondary research questions with audited empirical answers.
  - `limitations.md`: Comprehensive limitations covering quantum simulation, sample sizes, and clinical scope.
  - `temporal_prediction.md`: Mathematical formulation of Task T2 (Month 12 to Month 60) and leakage firewalls.
- **Data Contracts & Provenance (`docs/data/`)**:
  - `data_contract.md`: Schema contracts for clinical samples and feature blocks.
  - `ppmi.md`: Authoritative specification of the PPMI longitudinal cohort ($N=341$).
  - `PD_PUBLIC_DATASET_SELECTION_V2.md`: Public dataset provenance and selection audit.
- **Master Phase Reports (`docs/phase4/` through `phase9/`)**:
  - Formal peer-reviewed phase master reports documenting empirical progression across project milestones.
- **Pre-Implementation Benchmark & Repository Audits (`research/`)**:
  - `research/pre_implementation_benchmark/`: 14 forensic audit reports for Architecture A.
  - `research/repository_audit/`: Code-level feasibility audits of external repositories (YAIB, CEHR-BERT, etc.).
- **Experiment Verdicts (`experiments/architecture_a/`)**:
  - `ARCHITECTURE_A_BENCHMARK_RESULTS.csv`, `ARCHITECTURE_A_BENCHMARK_VERDICT.json`, `ARCHITECTURE_A_VERDICT.json`, `CLINICAL_PREDICTION_RESULT.json`, `evidence_graph.json`, `evidence_graph.metta`.

---

### Category D — Temporary & Runtime Files (Ignored / Ephemeral)
*Runtime caches and scratch execution files that are strictly excluded from version control.*

- `.pytest_cache/`: Ephemeral test execution cache.
- `__pycache__/`, `*.pyc`: Compiled Python bytecode.
- `*.log`, `*.tmp`, `*patient_prediction_output.json`: Runtime log outputs and scratch exports.

---

### Category E — Sensitive Data & Security Perimeter (Permanently Excluded)
*Protected research data, private patient identifiers, and credentials.*

- `data/*.csv`, `data/raw/`, `data/processed/`: Participant-level CSVs and clinical records.
- `*.dcm`, `*.nii`, `*.nii.gz`: Volumetric brain MRI scans.
- `*.pt`, `*.ckpt` (except small public pretrained weights): Large binary model checkpoints.
- `*.key`, `*.pem`, `credentials.json`: API credentials and access keys.
