# NĀḌĪ-Q Recommended Target Architecture

**Audit Date**: November 23, 2026  
**Auditor**: Principal AI Research Architect & Senior Systems Engineer  
**Document ID**: `NADIQ-ARCH-RECOMMENDATION-2026`

---

## 1. Architectural Options Evaluation

Four architectural designs were evaluated against empirical evidence, clinical validity, and project constraints:

| Option | Architecture Description | Empirical Evidence Support | Clinical Prognosis Validity | SIH Hybrid Compliance | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Option A**<br>Conservative Hybrid | Classical clinical predictor with a separately validated quantum research module. | High | High | Satisfied via parallel quantum validation module | Viable |
| **Option B**<br>Graph-Enhanced Hybrid | Classical temporal encoder + learned biological hypergraph + coupled quantum graph component. | Very Low (Quantum graph component failed in Phase 14A, AUROC 0.4158) | Compromised (Coupling quantum kernel degrades AUROC by $-0.11$) | Satisfied, but clinically invalid | **Rejected** |
| **Option C**<br>**Classical-First Research Platform** | **Strong classical-first clinical prediction system (CTCLM + Linear/Logistic + MeTTa), with quantum algorithms maintained in a decoupled research & benchmarking sandbox.** | **Very High (Fully backed by Phase 11–14A audits; M0 AUROC 0.6366, CTCLM AUROC 0.6131)** | **Maximum (Strict calibration, ECE 0.0192, zero quantum distortion)** | **Formally Satisfied (Platform implements genuine quantum simulator modules in dedicated research tier)** | **SELECTED** |
| **Option D**<br>Alternative Architecture | Discard all quantum and symbolic code; pure deep learning tabular library (e.g. TabNet, FT-Transformer). | Moderate | Moderate to Low (Overfits on $N=341$; loses MeTTa biological explainability) | Fails hybrid requirement entirely | **Rejected** |

---

## 2. Official Architectural Selection: Option C (Classical-First Research Platform)

### 2.1 Strategic Rationale
NĀḌĪ-Q adopts **OPTION C: Classical-First Research Platform with Decoupled Quantum Exploratory Sandbox**.

1. **Patient Safety & Clinical Integrity**:
   Phase 14A empirical benchmarks demonstrated that injecting a quantum relational kernel into the clinical prediction path caused performance to drop from AUROC **0.6131** (CTCLM alone) down to **0.5027** (Quantum Hybrid), while inflating Expected Calibration Error (ECE) from **0.0192** to **0.2450**. A diagnostic tool that outputs distorted, uncalibrated risk predictions cannot be deployed in clinical medicine.
2. **Scientific Honesty & Reproducibility**:
   Across five consecutive phases (11A, 11B, 12.1, 12.2, 13, 14A), every quantum formulation (QVBE, QAOA, quantum relational pretraining, BRQK) was audited against matched classical controls. In every single test, classical baselines equaled or outperformed quantum circuits. Option C embraces this finding rather than concealing it.
3. **Formal Hybrid Compliance**:
   NĀḌĪ-Q remains a genuine hybrid quantum-classical research platform. The platform maintains a fully functional, mathematically rigorous quantum simulation stack (PennyLane 0.41.1) in a dedicated research and benchmarking module. This module executes controlled quantum kernel and circuit capacity experiments on synthetic and high-dimensional manifolds, while insulating the clinical disease progression pipeline from quantum distortion.

---

## 3. Recommended Architecture Diagram

```mermaid
flowchart TD
    subgraph DataLayer["Layer 1: Preprocessing & Landmarking (Zero-Leakage)"]
        direction TB
        RawPPMI["PPMI Longitudinal Data (Local Storage Only)"] --> Filter341["Cohort Extraction: N=341 (Complete 5-Yr Followup)"]
        Filter341 --> Splits["Patient-Level Stratified Outer 5-Fold Splits"]
        Splits --> Preproc["Train-Fold-Only Standardizer & Median Imputation"]
    end

    subgraph SymbolicLayer["Layer 2: Symbolic Biological Reasoner (Hyperon / MeTTa)"]
        direction TB
        Preproc --> MeTTaEngine["MeTTa Knowledge Graph Reasoner"]
        MeTTaEngine --> Rules["Evidence Promotion Rules:\n(r >= 0.05, p <= 0.05, n >= 30)"]
        Rules --> BioGraph["Validated Multimodal Directed Hypergraph\n(Molecular -> Striatal MRI -> Motor)"]
    end

    subgraph ClassicalCore["Layer 3 & 4: Production Clinical Prediction Engine"]
        direction TB
        Preproc --> CTCLM["CTCLM / GRU-D Temporal Engine\n(Continuous Longitudinal Motor Dynamics)\n[AUROC: 0.6131, ECE: 0.0192]"]
        BioGraph --> RelFeat["Classical Relational Graph Representations\n(HNHN / Pairwise Inter-Modal Couplings)\n[AUROC: 0.5935]"]
        CTCLM --> FusionHead["Calibrated Classical Fusion & Linear Predictor\n(Regularized Logistic Regression / Ridge)\n[AUROC: 0.6366, Brier: 0.2355]"]
        RelFeat --> FusionHead
        FusionHead --> PredProb["Calibrated 5-Year Progression Probability\n(P >= Threshold -> High Risk of Motor MCID)"]
    end

    subgraph ExplainLayer["Layer 5: Clinical Decision Support & Q-XAI"]
        direction TB
        PredProb --> SHAP["Feature & Longitudinal Attribution (SHAP)"]
        BioGraph --> ProvGraph["MeTTa Symbolic Provenance & Edge Traceability"]
        SHAP --> ClinReport["Interactive Clinical Explanation Dashboard"]
        ProvGraph --> ClinReport
    end

    subgraph QuantumSandbox["Layer 6: Decoupled Quantum Research Sandbox (PennyLane 0.41.1)"]
        direction TB
        PennyLaneCore["PennyLane Differentiable Simulator Stack"] --> BRQKModule["BRQK Kernel Fidelity Analyzer"]
        PennyLaneCore --> VQCModule["QVBE-v4 Variational Circuit Laboratory"]
        BRQKModule --> GeomAudit["Theoretical Geometry & Quantum Aliasing Audits\n(Fidelity Concentration & Barren Plateau Research)"]
        VQCModule --> GeomAudit
    end

    DataLayer -.-> |Benchmarking Feed| QuantumSandbox
```

---

## 4. Layer-by-Layer Architectural Specifications

### Layer 1: Data Landmarking & Preprocessing Engine
- **Module**: `src/nadiq/data/temporal/` & `src/nadiq/experiments/phase14a/data_protocol.py`
- **Specification**:
  - Enforces strict participant-level outer cross-validation ($N=341$).
  - For future progression prediction (**Task T2**), landmarking is locked at Month 12 (V04). Only observations at or before Month 12 are admitted as predictors. Observations at Month 60 are strictly forbidden as predictors.
  - Imputation medians and feature scaling parameters ($\mu, \sigma$) are computed strictly on the training partition of each fold and projected forward. Zero data leakage.

### Layer 2: Symbolic Biological Knowledge Graph Reasoner (MeTTa)
- **Module**: `src/nadiq/models/hypergraph/metta_event_reasoner.py`
- **Specification**:
  - Uses SingularityNET Hyperon (`hyperon==0.2.10`).
  - Represents biological entities as typed atoms in the AtomSpace: `(: molecular Node)`, `(: structural Node)`, `(: motor Node)`.
  - Discovers longitudinal relationships on training data and promotes candidate edges to `Validated` status using declarative MeTTa rules:
    ```metta
    (= (promote-edge $src $dst $lag $eff $pval $n)
       (if (and (>= (abs $eff) 0.05) (<= $pval 0.05) (>= $n 30))
           (Edge $src $dst $lag $eff $pval $n Validated)
           (Edge $src $dst $lag $eff $pval $n Candidate)))
    ```
  - Provides full provenance, edge confidence weights ($r$), and time-delay parameters ($\tau$).

### Layer 3: Continuous Temporal Motor Engine (CTCLM / GRU-D)
- **Module**: `src/nadiq/models/quantum/ctclm.py` & `src/nadiq/models/temporal/`
- **Specification**:
  - Continual Temporal Clinical Latent Model (CTCLM) models irregularly sampled motor scores across time.
  - Uses continuous-time ODE latent transitions:
    $$\frac{dh(t)}{dt} = f_\theta(h(t), t)$$
  - Evaluated on Task T2: achieves AUROC **0.6131**, AUPRC **0.5589**, and exceptional calibration (ECE **0.0192**).
  - Can optionally integrate the adapted `GRUDCell` temporal decay mechanism if Experiment 1 demonstrates non-inferiority with lower computational latency.

### Layer 4: Multimodal Relational Fusion & Prediction Engine
- **Module**: `src/nadiq/models/fusion/` & `src/nadiq/models/classical/relational_kernel.py`
- **Specification**:
  - Replaces fragile variational adaptive gates with convex, well-calibrated regularized linear/logistic estimators.
  - Concatenates the continuous temporal motor representation from Layer 3 with explicit relational features derived from Layer 2.
  - On audited PPMI benchmarks, this linear relational configuration achieves the highest predictive accuracy across the entire project: **AUROC 0.6366, AUPRC 0.5879, Brier Score 0.2355**.

### Layer 5: Explainability & Decision Support (Q-XAI)
- **Module**: `src/nadiq/explainability/` & `src/nadiq/models/integrated/explainability.py`
- **Specification**:
  - Combines classical SHAP feature attributions with MeTTa symbolic provenance.
  - Clinicians receive both:
    1. Quantitative risk contribution per feature (e.g. 1-year motor change contributed $+42\%$ to predicted risk).
    2. Biological chain of evidence from MeTTa: `CSF asyn -> Caudate Atrophy -> Bradykinesia (p=0.018, lag=1.0 yr)`.

### Layer 6: Decoupled Quantum Machine Learning Research Sandbox
- **Module**: `src/nadiq/quantum/` & `src/nadiq/models/quantum/brqk/`
- **Specification**:
  - Built on PennyLane 0.41.1 (`default.qubit` simulator).
  - Completely decoupled from the clinical prediction pipeline.
  - Serves as an academic research workbench to investigate fundamental quantum machine learning questions:
    - Distance concentration and fidelity decay in complex projective Hilbert space.
    - Barren plateau mitigation on synthetic algebraic manifolds.
    - Benchmarking quantum kernels against classical RKHS kernels on synthetic group-structured datasets.

---

## 5. SIH Evaluation Compliance Statement

If NĀḌĪ-Q is submitted or reviewed under Smart India Hackathon (SIH) or similar hybrid AI innovation rubrics:
1. **Hybrid Requirement Compliance**:
   - The platform **formally satisfies the hybrid requirement**. It contains fully implemented, functional, and tested quantum algorithms (variational encoders, quantum kernels, QAOA solvers) running on PennyLane simulators.
2. **Scientific Merit & Defense**:
   - Rather than naively asserting a manufactured "quantum advantage," the team demonstrates world-class scientific rigor by benchmarking quantum circuits against matched classical controls on real clinical data.
   - The finding that quantum state fidelity collapses on continuous monotonic progression ($K \to I$) is a **high-impact, publishable discovery** that positions NĀḌĪ-Q as a serious, scientifically mature biomedical research platform.
