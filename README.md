# Nāḍī-Q — Hybrid Quantum-Classical Machine Learning Platform for Longitudinal Disease Progression

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Research Use Only](https://img.shields.io/badge/license-Research%20Use%20Only-green.svg)](pyproject.toml)
[![Tests: 409 passing](https://img.shields.io/badge/tests-409%20passed%20(100%25)-brightgreen.svg)](tests/)
[![PennyLane: 0.41](https://img.shields.io/badge/PennyLane-0.41.1-blueviolet.svg)](https://pennylane.ai/)
[![PyTorch: 2.4](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)

---

## 1. Scientific Overview

**Nāḍī-Q** (*Nāḍī* — Sanskrit for "vital channel" or "pulse") is a rigorous, research-grade experimental platform designed to evaluate **hybrid quantum-classical machine learning** architectures for early disease detection and 5-year longitudinal progression forecasting in neurodegenerative disorders (specifically Parkinson's Disease and related parkinsonian syndromes).

### Architecture A: Quantum-Assisted Evidence Discovery (Production MVP)
In response to extensive pre-implementation benchmarking and empirical audits, NĀḌĪ-Q decouples quantum exploration from clinical inference into two strictly independent planes:

- **Plane 1 (Evidence Discovery)**: OpenCog Hyperon **MeTTa** compiles symbolic biological constraints, redundancies, and contradictions into a **QUBO** problem. High-performance **Classical Tabu Search** solves this in **132 ms** (global optimum), while **QAOA** (PennyLane) runs as an asynchronous background exploration sidecar. Every relationship is screened via two-tier Benjamini–Hochberg FDR ($q \le 0.05$ / $q \le 0.10$) and honestly labelled as *Provisional/Explanatory*.
- **Plane 2 (Clinical Inference)**: 100% verified classical model operating on longitudinal PPMI multimodal features ($N=341$). Achieves **AUROC 0.6366** on Task T2 (Month-12 to Month-60 motor progression) with mathematical **$\Delta_{\max} = 0.00\text{e}{+}00$** quantum invariance.

---

## 2. Quickstart & CLIs

### Interactive Streamlit Web Application (UI)
Launch the unified graphical clinical decision support & evidence discovery dashboard:
```bash
streamlit run app.py
```
**Included Dashboards**:
- **Tab 1: Patient Clinical Inference (Quantum + Classical ML)**: Real-time multimodal prediction taking CSF Biomarkers (`asyn`, `ptau`), Structural MRI (`caudate`, `putamen`), and Motor scores (`UPDRS-III`), executing the PennyLane quantum circuit + CTCLM temporal model through Adaptive QCDF fusion with Q-XAI modality attributions ($\Delta P_q, \Delta P_t$) and JSON export.
- **Tab 2: Biological Evidence Discovery (MeTTa + QUBO + QAOA)**: Interactive MeTTa + QUBO + QAOA relationship explorer with topological Mermaid rendering and AtomSpace triplet downloads.
- **Tab 3: System Architecture & Benchmark Audit**: Real-time empirical metrics ($N=341$, AUROC $0.6366 \ge 0.635$, $\Delta_{\max} = 0.00\text{e}{+}00$), model comparison tables, and review defense FAQs.

### Evidence Discovery CLI
```bash
python scripts/nadiq_discover.py --output_dir experiments/architecture_a --solver tabu --budget 4
```

### Clinical Prediction CLI (100% Decoupled)
```bash
python scripts/nadiq_predict.py --task T2 --seeds 42 123 456 --output_dir experiments/architecture_a --evidence_graph experiments/architecture_a/evidence_graph.json
```

### Patient Multimodal Hybrid Inference (Quantum + Classical ML)
Accepts a patient's **Biomarker Report**, **MRI Report**, and **Motor Report**, running both the Quantum Relational Branch (PennyLane QVBE) and Classical Temporal Branch (CTCLM) with Adaptive QCDF Fusion:
```bash
# Demo clinical profile
python scripts/predict_patient.py --demo

# Custom patient measurements
python scripts/predict_patient.py \
    --asyn 1420.0 --ptau 18.5 \
    --caudate 2850.0 --putamen 3600.0 \
    --motor_total 26.0 --motor_subscore 4.5

# Real PPMI cohort participant
python scripts/predict_patient.py --patno 3102
```

### Comprehensive Verification Benchmark
```bash
python scripts/run_architecture_a_benchmark.py
```

---

## 2. Platform Architecture

```
nadi-q/
├── src/nadiq/                      # Core Python Package (src-layout)
│   ├── cli.py                      # Command-line interface entry point
│   ├── data/                       # Data access contracts, timelines & assemblers
│   ├── evaluation/                 # AUROC, AUPRC, Brier score, ECE & DeLong tests
│   ├── models/
│   │   ├── quantum/                # QVBE-v1, QVBE-v2, CTCLM
│   │   ├── qvbe_v3/                # Mathematically corrected QVBE-v3 circuit & trainer
│   │   ├── hypergraph/             # TQ-RHNet, quantum drift & MeTTa reasoner
│   │   ├── fusion/                 # Quantum-Classical Diagnostic Fusion (QCDF)
│   │   ├── classical/              # Logistic regression, MLP & Gradient Boosting
│   │   ├── biomarkers/             # Tabular biomarker encoders
│   │   └── mri/                    # 3D Convolutional Beta-VAE
│   ├── preprocessing/              # Leakage-free normalization & median imputation
│   └── quantum/                    # Quantum simulation backends & circuit primitives
├── configs/                        # YAML experiment & feature configurations
├── docs/                           # Architecture specifications & Phase 4–9 master reports
├── experiments/                    # Phase 10A & 10B experiment outputs & verdicts
├── reports/                        # Technical audits (PPMI data provenance & QC)
├── scripts/                        # Historical research scripts (Phases 2–9, 10A, 10B)
├── tests/                          # 313 unit, integration & scientific tests
├── pyproject.toml                  # PEP 517/518 build & package configuration
├── requirements.txt                # Curated runtime dependencies
├── requirements-dev.txt            # Development & test dependencies
├── .gitattributes                  # Normalized LF line endings & binary declarations
└── .gitignore                      # Security filter protecting PHI, weights & data
```

---

## 3. Core Architectural Modules

| Component | Module Path | Scientific Description |
|:---|:---|:---|
| **QVBE-v2** | `src/nadiq/models/quantum/qvbe_v2.py` | 8-qubit hardware-efficient parameterized quantum biomarker embedding circuit. |
| **QVBE-v3** | `src/nadiq/models/qvbe_v3/` | Mathematically regularized quantum encoder with Fubini-Study metric regularization and contrastive alignment. |
| **CTCLM** | `src/nadiq/models/quantum/ctclm.py` | Continual Temporal-Clinical Learning Module modeling evolving latent trajectories across clinical visits. |
| **QCDF** | `src/nadiq/models/fusion/qcdf.py` | Multi-modal projection and diagnostic fusion network with explicit missingness awareness. |
| **TQ-RHNet** | `src/nadiq/models/hypergraph/` | Time-Conditioned Quantum Relational Hypergraph Network with $R_{ZZ}$/$R_{ZZZ}$ interactions and quantum drift. |
| **MeTTa Reasoner** | `src/nadiq/models/hypergraph/metta_event_reasoner.py` | Symbolic reasoning engine executing clinical progression rules in Hyperon/OpenCog MeTTa. |

---

## 4. Key Empirical Benchmark Results

Evaluated on the $N=341$ longitudinal PPMI cohort under strict 5-fold cross-validation for 5-year progression prediction ($\Delta \text{MDS-UPDRS III} \ge 5.0$ at Month 60):

| Model Architecture | Model Paradigm | 5-Fold Mean AUROC | 5-Fold Mean Brier Score | Expected Calibration Error (ECE) |
|:---|:---:|:---:|:---:|:---:|
| **Early Trajectory Delta (1-Yr)** | **Classical Clinical Rate** | **0.6809** | **0.2254** | **0.0429** |
| Classical Hypergraph MLP | Classical Tabular/Graph | 0.5657 | 0.3605 | 0.2825 |
| TQ-RHNet (2-Body Time-Conditioned) | Hybrid Quantum-Classical | 0.5527 | 0.4118 | 0.3421 |
| TQ-RHNet Full (2B + 3B + Drift) | Hybrid Quantum-Classical | 0.5493 | 0.4201 | 0.3582 |
| TQ-RHNet Ablation (No Drift) | Hybrid Quantum-Classical | 0.5524 | 0.4220 | 0.3601 |
| CTCLM Functional Latent Model | Classical Neural ODE | 0.5346 | 0.2657 | 0.1248 |
| Pairwise Temporal Graph | Classical Relational | 0.5335 | 0.3791 | 0.3012 |
| Static Quantum Hypergraph (10A) | Hybrid Quantum-Classical | 0.5108 | 0.4432 | 0.3914 |

*Detailed benchmark tables, ablation logs, and scientific conclusions are documented in [`experiments/phase10b_tqrhnet/PHASE10B_VERDICT.md`](experiments/phase10b_tqrhnet/PHASE10B_VERDICT.md).*

---

## 5. Quick Start & Installation

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Nadi-Q

# Create and activate a virtual environment (Python 3.10, 3.11, or 3.12)
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install core runtime dependencies
pip install -r requirements.txt

# Or install in editable development mode with testing tools
pip install -e ".[dev]"
```

### Running the Test Suite

The platform contains 313 automated unit, integration, and scientific correctness tests:

```bash
# Run the entire test suite (unit, integration, scientific)
pytest

# Run fast unit tests only
pytest tests/unit/

# Run scientific mathematical verification tests
pytest tests/scientific/

# Run newly added hypergraph and TQ-RHNet tests
pytest tests/unit/test_hypergraph.py
```

### Executing Phase Experiments

```bash
# Run Phase 10B Time-Conditioned Quantum Relational Hypergraph benchmarks
python scripts/experiments/phase10b_tqrhnet.py

# Run Phase 10A Quantum Hyperedge pilot
python scripts/experiments/phase10a_quantum_temporal_hypergraph.py

# Run Phase 9 Quantum vs. Classical baseline comparison
python scripts/phase9/run_quantum_classical_comparison.py
```

---

## 6. Data Governance & Participant Privacy

The NĀḌĪ-Q project adheres to strict biomedical data ethics and protected health information (PHI) standards:
- **No Raw Clinical Data in Git**: Raw PPMI/ADNI tabular files, DICOM archives, NIfTI volumes, and participant identifiers (`PATNO`) are strictly excluded via `.gitignore` and reside in external protected research storage.
- **Reproducible Data Contracts**: The platform provides synthetic mock loaders (`src/nadiq/data/loaders/synthetic.py`) and schema verification contracts (`src/nadiq/data/contracts.py`) to permit full end-to-end testing without access to proprietary patient registries.
- **Deterministic Seeding**: All stochastic procedures, train/test splits, and quantum simulator measurements are strictly governed by centralized seeds (`src/nadiq/utils/seeding.py`).

---

## 7. License & Research Attribution

This software is released for **Research Use Only**. See [`pyproject.toml`](pyproject.toml) and [`REPOSITORY_INVENTORY.md`](REPOSITORY_INVENTORY.md) for full architecture and governance details.
