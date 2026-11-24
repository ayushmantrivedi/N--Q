# System Architecture — NĀḌĪ-Q

> **Status markers:**  
> **[A]** Established / Verified Standard · **[B]** Empirically Validated Hypothesis · **[C]** Audited Negative Finding · **[D]** Production MVP

---

## 1. Scientific & Engineering Philosophy

NĀḌĪ-Q is a hybrid quantum-classical machine learning research platform for multimodal early neurodegenerative disease detection and longitudinal progression forecasting (specifically Parkinson's Disease in the PPMI cohort).

The architecture is governed by five non-negotiable principles:

1. **Strict Decoupling of Clinical Inference from Quantum Optimization [D]** — Quantum optimization must never jeopardize patient safety or clinical prediction stability. Clinical inference operates on 100% verified classical foundations with proven quantum invariance ($\Delta_{\max} = 0.00\text{e}{+}00$).
2. **Reproducibility [A]** — Every experiment, fold split, and benchmark is fully deterministic given a random seed and configuration file.
3. **Scientific Honesty [A]** — Negative results (e.g., tightly coupled quantum kernels failing to beat classical baselines) are never erased or suppressed; they directly informed our architectural pivot to Architecture A.
4. **Leakage Prevention [A]** — Strict left-truncation firewalls ensure zero temporal data leakage across longitudinal visit snapshots.
5. **Production Readiness [D]** — All components are accessible via a graphical Streamlit web application and scriptable command-line tools.

---

## 2. Platform Architecture: Two-Plane Decoupled Design

In response to forensic audits across Phases 11–14A, NĀḌĪ-Q decouples quantum computation from patient risk prediction into two independent operational planes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   NĀḌĪ-Q PLATFORM                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   PLANE 1: BIOLOGICAL EVIDENCE DISCOVERY (Quantum-Assisted Graph Search)               │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │  PPMI Cohort (N=341) ──▶ Pearson Correlations ──▶ Two-Tier FDR Screening        │   │
│   │                                                         │ (q <= 0.05 / 0.10)   │   │
│   │                                                         ▼                      │   │
│   │  OpenCog Hyperon MeTTa ──▶ Redundancy & Contradiction ──▶ QUBO Matrix (N=15)   │   │
│   │  Symbolic Reasoner          Constraints Formulation                            │   │
│   │                                                         │                      │   │
│   │                     ┌───────────────────────────────────┴─────────────────┐    │   │
│   │                     ▼ (Primary Production)                                ▼    │   │
│   │           Classical Tabu Search (M3)                            PennyLane QAOA │   │
│   │           Latency: 127 ms | Gap: 0.0000                         p=1, 2 Sidecar │   │
│   │                     │                                             (Timeout:15s)│   │
│   │                     └───────────────────┬─────────────────────────────────┘    │   │
│   │                                         ▼                                      │   │
│   │                             Discovered Biological Graph                        │   │
│   │                             [JSON + MeTTa AtomSpace KB]                        │   │
│   └─────────────────────────────────────────┬──────────────────────────────────────┘   │
│                                             │ (Informational / Relational Context)     │
│   PLANE 2: CLINICAL RISK PREDICTION         ▼                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Patient Longitudinal Data ──▶ Strict Left-Truncation Firewall (Month <= 12)   │   │
│   │                                          │                                     │   │
│   │                                          ▼                                     │   │
│   │                           Classical Clinical Predictor                         │   │
│   │                           (CTCLM / Logistic Ensemble)                          │   │
│   │                                          │                                     │   │
│   │                                          ▼                                     │   │
│   │                           Task T2: 5-Year Motor Progression                    │   │
│   │                           AUROC: 0.6366 | AUPRC: 0.5879                        │   │
│   │                           Quantum Invariance: Δ_max = 0.00e+00                 │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                        │
│   UNIFIED PATIENT PREDICTOR & USER INTERFACES                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │  • Unified Multimodal Neural Model (QVBE-v4 + CTCLM + Adaptive QCDF Fusion)    │   │
│   │  • Q-XAI Counterfactual Explainer (ΔP_q, ΔP_t & Modality Percentage Breakdown) │   │
│   │  • Streamlit Web Application (app.py: 3 Production Tabs)                       │   │
│   │  • CLI Tools (predict_patient.py, nadiq_discover.py, nadiq_predict.py)         │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Plane 1: Biological Evidence Discovery

The purpose of Plane 1 is to discover non-redundant, statistically validated biological interaction networks across multimodal PPMI data.

### 3.1 Two-Tier Adaptive FDR Screening (`src/nadiq/architecture_a/fdr.py`)
Candidate feature pairs $(X_i, X_j)$ across molecular biofluids (`asyn`, `ptau`), striatal MRI (`caudate`, `putamen`), and motor scores (`motor_total`, `motor_subscore`) are screened using Benjamini–Hochberg False Discovery Rate control:
- **Tier 1 (Validated)**: $q \le 0.05$ — Statistically robust cross-modal relationships.
- **Tier 2 (Provisional)**: $0.05 < q \le 0.10$ — Exploratory hypotheses warranting independent cohort replication.
- **Statistical Quality Tag**: All discovered edges are explicitly tagged as `Provisional/Explanatory` to prevent premature medical claims.

### 3.2 MeTTa Symbolic Constraint Compiler (`src/nadiq/models/metta_quantum/qubo_compiler.py`)
OpenCog Hyperon MeTTa compiles domain biological rules into formal constraints:
- **Redundancy Constraints**: Penalizes co-selection of collinear intra-modality features (e.g. Caudate volume vs Putamen volume).
- **Contradiction Constraints**: Forbids biophysically incompatible causal relationships.
- **Budget Constraint**: Enforces selection of exactly $K$ edges:
  $$H_{\text{budget}} = \lambda \left(\sum_{i=1}^N x_i - K\right)^2$$

### 3.3 Dual Optimization Engine (`src/nadiq/architecture_a/optimization_engine.py`)
- **Classical Tabu Search ($M_3$, Production)**: High-performance heuristic with tabu tenure and aspiration criteria. Solves $N=15$ edge selection in **127 ms** with an optimality gap of **0.0000** (exact global optimum).
- **PennyLane QAOA ($p=1, 2$, Exploration Sidecar)**: Quantum Approximate Optimization Algorithm executed asynchronously on PennyLane simulators. Protected by a **15-second timeout thread**; if execution exceeds budget, the system falls back to Tabu search seamlessly.

---

## 4. Plane 2: Decoupled Clinical Risk Prediction

The purpose of Plane 2 is patient-level 5-year progression forecasting on Task T2.

### 4.1 Zero Quantum Imports Guarantee
`src/nadiq/architecture_a/clinical_predictor.py` maintains zero dependencies on PennyLane, quantum backends, or QUBO solvers. This guarantees that:
- Quantum simulator crashes, timeouts, or numerical noise cannot alter clinical risk scores.
- The mathematical quantum invariance condition holds:
  $$\Delta_{\max} = \max_{i} |P_{\text{with\_quantum}}(y_i) - P_{\text{without\_quantum}}(y_i)| = 0.00\text{e}{+}00$$

### 4.2 Longitudinal Task T2 Performance
Evaluated across 5 stratified folds on $N=341$ PPMI participants:
- **Ensemble AUROC**: **0.6366** (exceeds pre-implementation target $\ge 0.635$).
- **Ensemble AUPRC**: **0.5879**.
- **Brier Score**: **0.2355** (well-calibrated probabilities).

---

## 5. Unified Multimodal Hybrid Model & Q-XAI

For patient-level clinical decision support, NĀḌĪ-Q provides a unified neural architecture:

### 5.1 Architecture Components (`src/nadiq/models/integrated/model.py`)
1. **Quantum Relational Encoder (QVBE-v4)**:
   - 8-qubit hardware-efficient ansatz mapping standardized biofluid and imaging inputs to quantum expectation values:
     $$\mathbf{q} = [\langle Z_0 \rangle, \dots, \langle Z_7 \rangle]$$
2. **Classical Temporal LSTM (CTCLM)**:
   - Models longitudinal motor trajectories from baseline to Month 12:
     $$\mathbf{h}_t = \text{LSTM}(\mathbf{m}_t, \mathbf{h}_{t-1}, \Delta t)$$
3. **Adaptive QCDF Fusion**:
   - Dynamic attention gating dynamically weighting the quantum and temporal branches:
     $$\hat{y} = \sigma\left(\alpha_q W_q \mathbf{q} + \alpha_t W_t \mathbf{h}_t + \alpha_n W_n \mathbf{x}_{\text{node}} + b\right)$$

### 5.2 Q-XAI Counterfactual Explainability (`src/nadiq/models/integrated/explainability.py`)
Provides clinicians with two layers of interpretability:
- **Branch Counterfactual Attributions**:
  $$\Delta P_q = P(\text{full}) - P(\mathbf{q} = \mathbf{0})$$
  $$\Delta P_t = P(\text{full}) - P(\mathbf{h}_t = \mathbf{0})$$
- **Modality Breakdown**: Dynamic percentage attribution across Molecular Biofluids, Structural MRI, Motor Progression, and Quantum Relational features.

---

## 6. User Interfaces & Verification

- **Streamlit Web Application (`app.py`)**:
  - Run via `python -m streamlit run app.py`.
  - Tab 1: Patient Clinical Inference.
  - Tab 2: Biological Evidence Discovery.
  - Tab 3: System Architecture & Benchmark Audit.
- **Multimodal CLI Predictor (`scripts/predict_patient.py`)**:
  - Accepts raw clinical measurements or cohort participant IDs.
- **Full Test Suite (`tests/`)**:
  - 413 automated unit, integration, and scientific tests (100% passing).
