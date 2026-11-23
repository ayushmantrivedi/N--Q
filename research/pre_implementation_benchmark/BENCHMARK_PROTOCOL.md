# NĀḌĪ-Q Pre-Implementation Benchmark Validation Protocol
## Architecture A: Quantum-Assisted Evidence Discovery with Decoupled Classical Clinical Prediction

**Status**: Executed & Verified  
**Date**: November 23, 2026  
**Auditing Body**: Independent Research & Engineering Validation Board  
**Target Repository**: `https://github.com/ayushmantrivedi/N--Q`  
**Execution Environment**: Windows 11 Home x64, Python 3.12.4, PyTorch 2.4.1+cpu, PennyLane 0.41.1, Hyperon 0.2.10, Scikit-Learn 1.6.1, NumPy 2.2.6, Pandas 2.2.3  

---

## 1. Executive Summary & Objective

The objective of this pre-implementation benchmark validation is to establish empirical, mathematical, and architectural ground truth for **Architecture A** prior to any further engineering investment. 

### Architecture A Core Definition
In contrast to previously attempted architectures (Phases 11–14A) where quantum circuits were placed directly in the patient clinical prediction loop—culminating in Phase 14A where the Biological Relational Quantum Kernel (BRQK) degraded motor progression AUROC by $-0.1104$ ($p=0.0010$) and collapsed to worse-than-chance AUROC $0.4158$—**Architecture A strictly bifurcates the system into two decoupled planes**:
1. **Biological Evidence Discovery Plane**: MeTTa symbolic knowledge graph combined with Quadratic Unconstrained Binary Optimization (QUBO) / QAOA to select parsimonious, non-redundant, and non-contradictory biological relationship motifs from longitudinal multimodal data.
2. **Clinical Inference Plane**: Fully classical longitudinal machine learning models (Logistic Regression, CTCLM) operating on validated patient biomarkers and clinical motor scores. The clinical prediction pipeline does **not** execute quantum kernels or quantum variational encoders.

```
       ┌─────────────────────────────────────────────────────────────┐
       │             BIOLOGICAL EVIDENCE DISCOVERY PLANE             │
       │                                                             │
       │  PPMI Longitudinal Data (Train Folds Only, N=341)            │
       │                      │                                      │
       │                      ▼                                      │
       │             MeTTa Symbolic AtomSpace                        │
       │     (Evidence Promotion: p<=0.10, r>=0.05, n>=30)           │
       │                      │                                      │
       │                      ▼                                      │
       │              Biological QUBO Builder                        │
       │     (Redundancy R_ij, Contradictions C_ij, Budget K=4)      │
       │                      │                                      │
       │         ┌────────────┴────────────┐                         │
       │         ▼                         ▼                         │
       │  Classical Solvers         QAOA Quantum Ansatz              │
       │ (Exact, Tabu, Greedy, SA)  (PennyLane Simulator)            │
       │         │                         │                         │
       │         └────────────┬────────────┘                         │
       │                      ▼                                      │
       │          Validated Biological Graph                         │
       │       (Stored in Evidence JSON / KG)                        │
       └──────────────────────┬──────────────────────────────────────┘
                              │
                    Asynchronous Export
                    (Zero Runtime Coupling)
                              │
       ┌──────────────────────▼──────────────────────────────────────┐
       │                  CLINICAL INFERENCE PLANE                   │
       │                                                             │
       │  Patient Clinical Features (Baseline & Month 12)            │
       │                      │                                      │
       │                      ▼                                      │
       │         Classical Predictors (M0 / CTCLM)                   │
       │                      │                                      │
       │                      ▼                                      │
       │      Progression Probability: P(Delta >= 5.0)               │
       │      (AUROC: 0.6353, Delta vs Quantum Disabled: 0.00e+00)   │
       └─────────────────────────────────────────────────────────────┘
```

---

## 2. Tested Hypotheses

| ID | Hypothesis | Success Criterion | Empirical Outcome | Verdict |
|:---|:---|:---|:---|:---|
| **H1** | **Mathematical & Ising Equivalence**: QUBO minimization energy matches Ising spin Hamiltonian energy exactly across random states. | $\max \|H_{\text{QUBO}}(x) - H_{\text{Ising}}(1-2x)\| < 10^{-9}$ across $N \in \{6, 8, 10, 12, 14\}$. | $\max \text{error} = 5.68 \times 10^{-14}$ across 2,500 samples. | **CONFIRMED** |
| **H2** | **Objective Non-Separability**: Pairwise redundancy and contradiction penalties make the objective non-separable, causing naive Top-$K$ to fail. | Top-$K$ heuristic yields non-zero optimality gap and violates constraints on coupled graphs. | Top-$K$ gap $= 3.0700$, feasibility = False (contradiction violated). Exact finds feasible global minimum. | **CONFIRMED** |
| **H3** | **Quantum Optimization Advantage**: QAOA ($p=1, 2$) achieves lower energy, smaller optimality gap, or faster runtime than classical solvers. | $\text{Gap}(\text{QAOA}) < \text{Gap}(\text{Tabu})$ or $\text{Runtime}(\text{QAOA}) < \text{Runtime}(\text{Tabu})$. | Tabu gap $= 0.0011$ (runtime 74.8 ms). QAOA $p=2$ gap $= 0.6593$ (runtime 12,930 ms, $173\times$ slower). | **REJECTED** |
| **H4** | **Biological Discovery & Replication**: QAOA selection improves held-out biological replication rate or direction consistency over classical methods. | $\text{Rep}(\text{QAOA}) > \text{Rep}(\text{Classical}) + 0.05$. | Exact $= 6.67\%$, Tabu $= 6.67\%$, QAOA $p=1 = 8.78\%$, QAOA $p=2 = 7.33\%$ ($p > 0.40$, indistinguishable). | **REJECTED** |
| **H5** | **Clinical Prediction Invariance**: Decoupled clinical predictions under Architecture A are strictly invariant to quantum service status. | $\max_i \|P_{\text{enabled}}(y_i) - P_{\text{disabled}}(y_i)\| < 10^{-8}$. | $\max \Delta = 0.00\text{e}+00$ across all $N=341$ subjects. Identical AUROC (0.6353), Brier (0.2359), ECE (0.0615). | **CONFIRMED** |

---

## 3. Cohort & Data Protocols

### 3.1 PPMI Cohort Specification
- **Participants**: $N = 341$ longitudinally tracked Parkinson's Disease participants from the PPMI cohort (`ppmi_longitudinal_multimodal_biofluid_n341.parquet`, `curated_cut_cache.parquet`).
- **Primary Clinical Target (Task T2)**: Future motor progression from Month 12 to Month 60:
  $$\Delta_{\text{future}} = (\text{NP3TOT}_{\text{M60}} - \text{NP3TOT}_{\text{BL}}) - (\text{NP3TOT}_{\text{M12}} - \text{NP3TOT}_{\text{BL}}) \ge 5.0$$
  - Progressors: 159 subjects ($46.6\%$)
  - Non-progressors: 182 subjects ($53.4\%$)
- **Multimodal Features**:
  1. *Molecular Biofluid*: CSF $\alpha$-synuclein (`asyn`), CSF phospho-Tau181 (`ptau`), CSF amyloid-$\beta_{1-42}$ (`abeta`).
  2. *Structural Striatal MRI*: Bilateral Caudate volume (`MIA_CAUDATE_BILAT`), Bilateral Putamen volume (`MIA_PUTAMEN_BILAT`).
  3. *Motor / Clinical*: MDS-UPDRS Part III Total (`updrs3_score`), Bradykinesia subscore (`NP3BRADY`), Gait subscore (`NP3GAIT`).

### 3.2 Cross-Validation & Data Leakage Protections
- **Folds**: 3 Random Seeds ($42, 123, 456$) $\times$ 5-Fold Stratified K-Fold Cross-Validation = 15 total folds.
- **Participant Level Partitioning**: Splitting is executed strictly by unique `PATNO`. No participant appears in both training and test folds.
- **Fold-Contained Imputation & Scaling**:
  - Missing values are imputed using the median of the **training fold only**.
  - Robust scaling (molecular) and standard scaling (structural/motor) are fit on the **training fold only** and transformed onto the held-out test fold.
  - All correlation screening, MeTTa symbolic promotions, and QUBO matrix constructions occur exclusively on training fold data.

---

## 4. Benchmark Harness Suite

The benchmark is executed by four isolated scripts located in `research/pre_implementation_benchmark/scripts/`:

1. `harness_mathematical_validation.py`:
   - Validates algebraic mapping: $x_i = \frac{I - Z_i}{2}$.
   - Tests floating-point tolerance check across $N \in \{6, 8, 10, 12, 14\}$ with 500 samples each (2,500 total checks).
   - Verifies constraint penalty scaling bounds: $\gamma > \max_i U_i$, $\mu > \max_{(i,j)} (U_i + U_j)$.
   - Demonstrates Top-$K$ heuristic failure on coupled non-separable graphs.

2. `harness_synthetic_benchmark.py`:
   - Generates synthetic multimodal cohort ($N=341$) across 10 independent random seeds.
   - Ground truth has 4 planted causal relationships, 2 redundant collinear distractors ($R_{ij} \ge 0.70$), 2 contradictory pairs, and 2 pure noise associations.
   - Evaluates 7 solvers: Top-$K$, M0 Exact, M1 Simulated Annealing, M2 Greedy, M3 Tabu, QAOA $p=1$, QAOA $p=2$.
   - Measures Precision, Recall, F1 against planted ground truth, feasibility, optimality gap, and stability.

3. `harness_ppmi_discovery.py`:
   - Evaluates 8 discovery protocols across all 15 stratified folds on real PPMI data: Raw Pearson ($p \le 0.10$), Pearson FDR-BH ($q \le 0.05$), MeTTa FDR, Top-$K$, M0 Exact, M3 Tabu, QAOA $p=1$, QAOA $p=2$.
   - Measures held-out replication rate ($p_{\text{test}} \le 0.10$ and matching sign), direction consistency, multimodal coverage, selection stability, and runtime.

4. `harness_clinical_isolation.py`:
   - Executes Task T2 5-fold cross-validation with the quantum discovery service active vs completely disabled.
   - Measures maximum absolute prediction difference across all $N=341$ subjects.
   - Compares performance against historical Phase 14A coupled hybrid benchmarks.

---

## 5. Reproduction Command Reference

To reproduce the benchmark results independently:

```powershell
# Set repository environment
cd c:\Users\ayush\OneDrive\Desktop\Nadi-Q

# 1. Run Mathematical & Formulation Validation
python research/pre_implementation_benchmark/scripts/harness_mathematical_validation.py

# 2. Run Synthetic Ground-Truth Benchmark (10 Replicates)
python research/pre_implementation_benchmark/scripts/harness_synthetic_benchmark.py

# 3. Run PPMI Longitudinal Multimodal Discovery Benchmark (15 Folds)
python research/pre_implementation_benchmark/scripts/harness_ppmi_discovery.py

# 4. Run Clinical Prediction Isolation Verification
python research/pre_implementation_benchmark/scripts/harness_clinical_isolation.py
```
