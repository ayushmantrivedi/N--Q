# Nāḍī-Q — Hybrid Quantum-Classical Machine Learning Platform for Longitudinal Disease Progression

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Research Use Only](https://img.shields.io/badge/license-Research%20Use%20Only-green.svg)](pyproject.toml)
[![Tests: 413 passing](https://img.shields.io/badge/tests-413%20passed%20(100%25)-brightgreen.svg)](tests/)
[![PennyLane: 0.41](https://img.shields.io/badge/PennyLane-0.41.1-blueviolet.svg)](https://pennylane.ai/)
[![PyTorch: 2.4](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)

---

## 1. Scientific Overview

**Nāḍī-Q** (*Nāḍī* — Sanskrit for "vital channel" or "pulse") is a research-grade platform designed to evaluate **hybrid quantum-classical machine learning** architectures for early disease detection and 5-year longitudinal progression forecasting in neurodegenerative disorders (specifically Parkinson's Disease in the PPMI cohort).

### Architecture A: Quantum-Assisted Evidence Discovery (Production MVP)
Following empirical audits across Phases 11–14A demonstrating that tightly coupled quantum kernels (SVM/BRQK) suffered from sample-size noise and failed to beat classical baselines, NĀḌĪ-Q decouples quantum computation from patient risk prediction into two strictly independent planes:

- **Plane 1 (Evidence Discovery)**: OpenCog Hyperon **MeTTa** compiles biological constraints, redundancies, and contradictions into a **QUBO** problem. High-performance **Classical Tabu Search** solves this in **127 ms** (global optimum), while **QAOA** (PennyLane) runs as an asynchronous background exploration sidecar with a 15-second timeout watchdog. All discovered edges are screened via two-tier Benjamini–Hochberg FDR ($q \le 0.05$ / $q \le 0.10$) and labelled *Provisional/Explanatory*.
- **Plane 2 (Clinical Inference)**: 100% verified classical model operating on longitudinal PPMI multimodal features ($N=341$). Achieves **AUROC 0.6366** on Task T2 (Month-12 to Month-60 motor progression) with mathematical **$\Delta_{\max} = 0.00\text{e}{+}00$** quantum invariance.

---

## 2. Quickstart & User Interfaces

### 2.1 Interactive Streamlit Web Application (UI)
Launch the graphical clinical decision support & evidence discovery dashboard:
```bash
python -m streamlit run app.py
```
*(Alternatively, run `streamlit run app.py` if Streamlit is in your system PATH.)*

**Included Dashboard Tabs**:
- **Tab 1: Patient Clinical Inference (Quantum + Classical ML)**: Real-time multimodal prediction taking CSF Biomarkers (`asyn`, `ptau`), Structural MRI (`caudate`, `putamen`), and Motor scores (`UPDRS-III`), executing the PennyLane quantum circuit + CTCLM temporal model through Adaptive QCDF fusion with Q-XAI modality attributions ($\Delta P_q, \Delta P_t$) and JSON export.
- **Tab 2: Biological Evidence Discovery (MeTTa + QUBO + QAOA)**: Interactive MeTTa + QUBO + QAOA relationship explorer with topological Mermaid rendering and AtomSpace triplet downloads.
- **Tab 3: System Architecture & Benchmark Audit**: Real-time empirical metrics ($N=341$, AUROC $0.6366 \ge 0.635$, $\Delta_{\max} = 0.00\text{e}{+}00$), model comparison tables, and review defense FAQs.

### 2.2 Patient Multimodal Hybrid Inference CLI
Accepts a patient's **Biomarker Report**, **MRI Report**, and **Motor Report**, running both the Quantum Relational Branch (PennyLane QVBE) and Classical Temporal Branch (CTCLM) with Adaptive QCDF Fusion:
```bash
# Demo clinical profile
python scripts/predict_patient.py --demo

# Custom patient measurements
python scripts/predict_patient.py \
    --asyn 1380.0 --ptau 19.5 \
    --caudate 2750.0 --putamen 3450.0 \
    --motor_total 28.0 --motor_subscore 5.0

# Real PPMI cohort participant by PATNO
python scripts/predict_patient.py --patno 3102
```

### 2.3 Evidence Discovery CLI
```bash
python scripts/nadiq_discover.py --output_dir experiments/architecture_a --solver tabu --budget 4
```

### 2.4 Decoupled Clinical Prediction CLI
```bash
python scripts/nadiq_predict.py --task T2 --seeds 42 123 456 --output_dir experiments/architecture_a
```

### 2.5 End-to-End Verification Benchmark
```bash
python scripts/run_architecture_a_benchmark.py
```

---

## 3. Platform Architecture

```
nadi-q/
├── app.py                          # Streamlit Production Web Application
├── src/nadiq/                      # Core Python Package (src-layout)
│   ├── architecture_a/             # Production MVP (Pipeline, FDR, Engine, Exporter)
│   ├── cli.py                      # Command-line interface entry point
│   ├── data/                       # Data access contracts, timelines & assemblers
│   ├── evaluation/                 # AUROC, AUPRC, Brier score, ECE & DeLong tests
│   ├── models/
│   │   ├── integrated/             # Unified Model (QVBE-v4 + CTCLM + QCDF + Q-XAI)
│   │   ├── metta_quantum/          # MeTTa compiler, QUBO builder, Tabu & QAOA solvers
│   │   ├── quantum/                # QVBE-v1, QVBE-v2, CTCLM
│   │   ├── qvbe_v3/                # Mathematically regularized QVBE-v3 circuit
│   │   ├── qvbe_v4/                # 8-qubit hardware-efficient ansatz & trainer
│   │   ├── hypergraph/             # TQ-RHNet, quantum relational drift & MeTTa reasoner
│   │   ├── fusion/                 # Quantum-Classical Diagnostic Fusion (QCDF)
│   │   ├── classical/              # Logistic regression, MLP & Gradient Boosting
│   │   ├── biomarkers/             # Tabular biomarker encoders
│   │   └── mri/                    # 3D Convolutional Beta-VAE
│   ├── preprocessing/              # Leakage-free normalization & median imputation
│   └── quantum/                    # Quantum simulation backends & circuit primitives
├── configs/                        # YAML experiment & feature configurations
├── docs/                           # Architecture specifications, data contracts & reports
├── experiments/                    # Architecture A verdicts, Phase 10A/10B verdicts
├── reports/                        # Technical audits (PPMI data provenance & QC)
├── scripts/                        # Active CLI tools & historical phase scripts
├── tests/                          # 413 unit, integration & scientific tests
├── pyproject.toml                  # PEP 517/518 build & package configuration
├── requirements.txt                # Curated runtime dependencies
├── requirements-dev.txt            # Development & test dependencies
├── .gitattributes                  # Normalized LF line endings & binary declarations
└── .gitignore                      # Security filter protecting PHI, weights & data
```

---

## 4. Core Architectural Modules

| Component | Module Path | Scientific Description |
|:---|:---|:---|
| **Architecture A Pipeline** | `src/nadiq/architecture_a/pipeline.py` | Production coordinator ensuring zero-leakage and decoupled planes. |
| **Two-Tier Adaptive FDR** | `src/nadiq/architecture_a/fdr.py` | Two-tier Benjamini–Hochberg screening ($q \le 0.05$ / $q \le 0.10$) + MDE power analysis. |
| **DualSolverEngine** | `src/nadiq/architecture_a/optimization_engine.py` | Classical Tabu Search ($M_3$, 127 ms) + PennyLane QAOA ($p=1, 2$) async sidecar. |
| **Decoupled Predictor** | `src/nadiq/architecture_a/clinical_predictor.py` | Pure classical prospective model; zero quantum imports; $\Delta_{\max} = 0.00\text{e}{+}00$. |
| **Unified Multimodal Model**| `src/nadiq/models/integrated/model.py` | PennyLane QVBE-v4 + CTCLM LSTM + Adaptive QCDF dynamic attention gating. |
| **Q-XAI Explainer** | `src/nadiq/models/integrated/explainability.py` | Counterfactual branch attributions ($\Delta P_q, \Delta P_t$) and modality % breakdown. |
| **CTCLM** | `src/nadiq/models/quantum/ctclm.py` | Continual Temporal-Clinical Learning Module modeling evolving longitudinal trajectories. |
| **QVBE-v4** | `src/nadiq/models/qvbe_v4/` | 8-qubit hardware-efficient ansatz with Fubini-Study metric regularization. |
| **MeTTa Reasoner** | `src/nadiq/models/hypergraph/metta_event_reasoner.py`| Symbolic reasoning engine compiling clinical progression rules in Hyperon MeTTa. |

---

## 5. Empirical Benchmark Results (Task T2: Month 12 → Month 60)

Evaluated under strict 5-fold cross-validation on the local authoritative PPMI cohort ($N=341$ participants, 159 progression events [46.6%]):

| Architecture ID | Model Configuration | Clinical AUROC | Clinical AUPRC | Brier Score | Latency | Quantum Invariance |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **M0** | Classical Logistic Baseline | 0.6366 | 0.5879 | 0.2355 | 12 ms | Baseline |
| **M3 (Arch A)**| **Architecture A: CTCLM + Tabu QUBO Discovery** | **0.6366** | **0.5879** | **0.2355** | **195.7 ms** | **$\Delta_{\max} = 0.00\text{e}{+}00$** |
| **M4** | CTCLM + QVBE-v4 Adaptive QCDF Fusion | 0.6104 | 0.5610 | 0.2480 | 420 ms | Dynamic Gating |
| **M6** | Zero-Quantum Suppressed Control | 0.6080 | 0.5590 | 0.2490 | 18 ms | Pure Classical |
| **M7** | Tightly Coupled Quantum Kernel (Phase 14A) | 0.5140 | 0.4680 | 0.2850 | 7,400 ms | Noise Degraded |

*Detailed benchmark tables, CSV logs, and JSON verdicts are persisted in [`experiments/architecture_a/`](experiments/architecture_a/).*

---

## 6. Installation & Test Verification

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd Nadi-Q

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Running the Test Suite (413 / 413 Passing)
```bash
# Run all 413 automated tests
pytest

# Run core Architecture A and integration tests
pytest tests/unit/test_app.py tests/unit/test_architecture_a.py tests/unit/test_integrated_system.py

# Run scientific mathematical tests
pytest tests/scientific/
```

---

## 7. Data Governance & Security Perimeter

- **No Raw Clinical Data in Git**: All PPMI tabular CSVs, volumetric DICOM/NIfTI scans, and participant identifiers (`PATNO`) are strictly excluded via `.gitignore` and remain encrypted on local research storage.
- **Leakage Prevention**: Enforces strict left-truncation firewalls ($t \le 12\text{ months}$) and participant-disjoint cross-validation splits.
- **Deterministic Seeding**: Centralized seeding (`src/nadiq/utils/seeding.py`) ensures 100% bit-exact reproducibility across all runs.

---

## 8. License & Research Attribution

This software is released for **Research Use Only**. See [`pyproject.toml`](pyproject.toml), [`PROJECT_STATE.md`](PROJECT_STATE.md), and [`REPOSITORY_INVENTORY.md`](REPOSITORY_INVENTORY.md) for full architecture and governance details.
