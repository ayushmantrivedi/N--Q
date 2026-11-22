# System Architecture — Nāḍī-Q

> **Status markers:**  
> **[A]** Established · **[B]** Hypothesis · **[C]** Experimental Claim · **[D]** Future Architecture

---

## 1. Design Philosophy

This platform is built on four non-negotiable principles:

1. **Reproducibility** — Every experiment must be fully deterministic given a configuration file and a random seed.
2. **Scientific validity** — No component may claim a result that has not been empirically demonstrated. Untested components are clearly labelled.
3. **Modularity** — Classical and quantum components must be independently testable, replaceable, and ablatable.
4. **Experimental verifiability** — The system must support both confirmation of hypotheses and rejection of null hypotheses with equal facility.

---

## 2. Repository Layout Rationale

```
nadi-q/
├── src/nadiq/           [A] Python package (PEP 517 src layout)
├── configs/             [A] Experiment configurations (YAML + OmegaConf)
├── data/                [A] Data access layer only — no raw data committed
├── experiments/         [A] Reproducible experiment entry points
├── notebooks/           [A] Exploratory analysis only — not experiments
├── scripts/             [A] One-off utilities — not experiments
├── tests/               [A] Three-tier test structure
└── docs/                [A] Research documentation
```

### Why `src/` layout?

Using `src/nadiq/` instead of `nadiq/` at the root prevents accidental import of the source directory without an explicit install. This ensures that tests always run against the installed package, not a partially-constructed directory. This is a well-established Python packaging practice.

### Why separate `experiments/` from `src/`?

Experiment scripts (`experiments/`) are executable entry points that depend on `nadiq` but are not part of the package itself. They should not be importable as library code. This separation:
- Prevents research scripts from polluting the library namespace.
- Ensures the library is independently installable.
- Allows experiments to be tracked in version control as configuration-driven artifacts.

### Why no frontend/API/database yet?

These would add operational complexity before the fundamental scientific questions have been answered. Adding infrastructure prematurely creates obligations that constrain research decisions.

---

## 3. Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        nadiq Package                           │
│                                                                 │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────────┐ │
│  │  data/   │───▶│ preprocessing/│───▶│  models/classical/   │ │
│  │ (loaders)│    │ (transforms)  │    │  (encoders, baselines)│ │
│  └──────────┘    └───────────────┘    └──────────┬───────────┘ │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────────────────┐    ┌───────────────────────────┐ │
│  │  quantum/                │◀──▶│  models/quantum/  [B]     │ │
│  │  ├── encodings/  [B]     │    │  (QVBE circuits)          │ │
│  │  ├── circuits/   [B]     │    └───────────────────────────┘ │
│  │  ├── noise/      [A]     │                   │              │
│  │  ├── backends/   [A]     │                   ▼              │
│  │  └── execution/  [A]     │    ┌───────────────────────────┐ │
│  └──────────────────────────┘    │  models/fusion/   [B]     │ │
│                                  │  (QCDF)                   │ │
│                                  └───────────────────────────┘ │
│                                               │                 │
│  ┌──────────────┐    ┌──────────────┐         ▼                │
│  │explainability│    │  evaluation/ │◀────────────────────────┘│
│  │  [B]         │    │  [A]         │                          │
│  └──────────────┘    └──────────────┘                          │
│                                                                 │
│  ┌─────────────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │  experiments/   │  │ robustness/│  │  fairness/  [D]    │   │
│  │  [A]            │  │  [B]       │  │                    │   │
│  └─────────────────┘  └────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Data Layer (`src/nadiq/data/`) [A]

**Purpose:** Abstract all dataset access behind a uniform interface.

Design goals:
- A `DatasetContract` base class specifying the required interface every dataset loader must implement.
- Loaders return standardized `ClinicalSample` objects, not raw DataFrames.
- No raw data files are stored in the repository.
- Datasets are accessed via configuration-specified paths or registries.

**Status:** Interface designed; no loaders implemented yet.

### 3.2 Preprocessing (`src/nadiq/preprocessing/`) [A]

**Purpose:** Modular, pipeline-composable preprocessing transforms.

- Each transform is a stateless or fittable object with `fit()`, `transform()`, and `inverse_transform()`.
- Pipelines are serialisable so that the exact preprocessing applied to training data can be reproducibly applied to test data.
- Missing value handling, normalization, and feature selection are separate, composable stages.

**Status:** Interface designed; no transforms implemented yet.

### 3.3 Classical Models (`src/nadiq/models/classical/`) [A]

**Purpose:** Classical baselines that serve as rigorous comparison points.

- Must include at minimum: logistic regression, gradient-boosted trees (XGBoost/LightGBM), and an MLP of equivalent parameter count to any proposed quantum model.
- These baselines are the *primary comparison* in all experiments.
- A quantum model that does not outperform (or match) a well-tuned classical baseline is not a scientific success — this is expected and acceptable.

**Status:** Interface designed; no models implemented yet.

### 3.4 Quantum Layer (`src/nadiq/quantum/`) [B/A]

**Purpose:** Parameterized quantum circuits for biomarker embedding.

Sub-components:
- `encodings/` [B] — data encoding strategies (angle, amplitude, IQP, re-uploading)
- `circuits/` [B] — variational ansatz definitions
- `noise/` [A] — noise channel models for simulation
- `backends/` [A] — backend abstraction interface (simulator, real device)
- `execution/` [A] — shot management, batching, and error mitigation hooks

**Important:** The `backends/` interface ensures that quantum circuit code is backend-agnostic. The same circuit should be executable on PennyLane's Lightning simulator, Qiskit Aer, or (eventually) a real IBM device by changing only the configuration.

**Status:** Interface designed; no circuits implemented yet.

### 3.5 Fusion (`src/nadiq/models/fusion/`) [B]

**Purpose:** Hybrid quantum-classical fusion (QCDF).

This is the core research hypothesis: that a learned fusion of quantum-embedded features and classical features produces superior risk stratification.

**Status:** Architecture only. Implementation requires completed classical and quantum components.

### 3.6 Explainability (`src/nadiq/explainability/`) [B]

**Purpose:** Research into attribution methods for quantum circuits.

- Classical explanations (SHAP, integrated gradients) applied to the classical components.
- Quantum-specific attribution (parameter shift rule gradients, Pauli decomposition) — research direction only.

**Status:** Research direction; no implementation.

### 3.7 Evaluation (`src/nadiq/evaluation/`) [A]

**Purpose:** Metric computation and statistical testing.

- All metrics computed with proper uncertainty estimation (bootstrap CIs).
- Statistical significance tests explicitly applied before any claim of improvement.
- Calibration metrics included alongside discrimination metrics.

**Status:** Interface designed; no implementation.

---

## 4. Configuration System

All experiment parameters are specified in YAML using OmegaConf:

```
configs/
├── base.yaml                    # Global defaults
├── data/
│   ├── synthetic_tabular.yaml
│   └── mimic_iii.yaml           # Placeholder; data not included
├── models/
│   ├── classical_baseline.yaml
│   ├── qvbe_v1.yaml             # First quantum circuit design
│   └── qcdf_v1.yaml             # First fusion design
├── quantum/
│   ├── simulator.yaml           # Default: PennyLane Lightning
│   └── noise_model_basic.yaml
└── experiments/
    ├── ablation_01_encoding.yaml
    └── baseline_comparison_01.yaml
```

**Every experiment configuration must specify:**
- `seed`: global random seed
- `backend`: quantum backend identifier
- `data`: dataset identifier and split strategy
- `model`: model configuration reference
- `evaluation`: metrics and statistical test configuration

---

## 5. Dependency Graph (Module Level)

```
utils  ◀──────────────────── (all modules depend on this)
data   ──▶ preprocessing ──▶ models/classical
                         ──▶ models/quantum ◀── quantum/*
                         ──▶ models/fusion  ◀── models/classical + models/quantum
models/* ──▶ evaluation
evaluation ──▶ experiments
experiments ──▶ explainability
```

**Hard constraint:** `models/` must never import from `experiments/`. The dependency goes one way.

---

## 6. What Is NOT In This Architecture (Yet)

| Component | Reason for Exclusion |
|-----------|----------------------|
| REST API | No serving requirements at research stage |
| Database | Data accessed via file system with registered loaders |
| Frontend | No user-facing interface needed |
| Docker | Adds operational overhead before scientific value is established |
| Cloud deployment | Premature before baseline experiments complete |
| Authentication | No multi-user or API surface |
| Real patient data loaders | Ethical approval and data access agreements required first |
