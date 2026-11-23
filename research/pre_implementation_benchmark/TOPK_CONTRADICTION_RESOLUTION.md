# NĀḌĪ-Q Forensic Validation: Top-K Contradiction Resolution
## Comprehensive Reconciliation of Discrepant Top-K Optimality Gaps and Feasibility Results

**Date**: November 24, 2026  
**Auditor**: Independent Algorithm & Optimization Forensic Reviewer  
**Audited Artifacts**:
- `research/pre_implementation_benchmark/scripts/harness_mathematical_validation.py`
- `research/pre_implementation_benchmark/scripts/harness_synthetic_benchmark.py`
- `research/pre_implementation_benchmark/scripts/harness_ppmi_discovery.py`
- `research/pre_implementation_benchmark/results/mathematical_validation_results.json`
- `research/pre_implementation_benchmark/results/synthetic_benchmark_summary.json`
- `research/pre_implementation_benchmark/results/ppmi_discovery_summary.json`

---

## 1. Executive Summary of the Discrepancy

In the pre-implementation benchmark documentation, two seemingly contradictory findings were reported regarding the performance of the naive **Top-$K$ heuristic**:

1. **In the Master Review Matrix & Mathematical Validation**:
   - Top-$K$ was reported to have an **optimality gap of 3.0700**, **0% feasibility**, and **violated a mutual contradiction constraint**.
2. **In the Synthetic Benchmark Table**:
   - Top-$K$ was reported to have an **optimality gap of 0.0000**, **100% feasibility**, and **F1 = 1.0000** (identical to the exact global solver).

### Forensic Verdict
**These two results are NOT a mathematical or software error; they represent evaluations of two entirely distinct QUBO instances with different problem dimensions, coefficient matrices, and constraint structures.**

However, the master report created confusion by presenting both numbers under the general label "Top-$K$ performance" without clearly differentiating between:
- An **adversarial analytical counterexample** designed specifically to prove theoretical non-separability ($N=4, K=2$).
- A **causal synthetic benchmark generator** where true edges had higher marginal correlations than distractors ($N=10, K=4$).
- A **real-world PPMI empirical benchmark** where Top-$K$ achieved an intermediate optimality gap of **0.3062** ($N=10, K=4$).

---

## 2. Deep Dive: Instance 1 — The Analytical Counterexample

**Source Script**: `research/pre_implementation_benchmark/scripts/harness_mathematical_validation.py` (`demonstrate_non_separability()`)  
**Stored Artifact**: `research/pre_implementation_benchmark/results/mathematical_validation_results.json`  

### 2.1 Problem Specification
- **Dimension**: $N = 4$ candidate variables, Budget $K = 2$.
- **Hyperparameters**: $\lambda = 0.80$ (redundancy), $\mu = 2.50$ (contradiction), $\gamma = 2.00$ (budget).
- **Candidate Definitions**:
  - $e_0$: $\text{mol}_A \to \text{struct}_B$ ($U_0 = 1.00$, Rank 1)
  - $e_1$: $\text{mol}_{A,\text{iso}} \to \text{struct}_B$ ($U_1 = 0.95$, Rank 2)
  - $e_2$: $\text{struct}_B \to \text{motor}_C$ ($U_2 = 0.80$, Rank 3)
  - $e_3$: $\text{mol}_D \to \text{motor}_E$ ($U_3 = 0.75$, Rank 4)
- **Constraints**:
  - Redundancy: $(e_0, e_1)$ has collinearity $R_{01} = 0.90$.
  - Contradiction: $(e_0, e_1) \in \mathcal{C}$ (mutually exclusive biological pathways).

### 2.2 Resulting QUBO Matrices
Using the standard minimization formula:
$$q_i = -U_i + \gamma (1 - 2K) = -U_i + 2.0 \times (1 - 4) = -U_i - 6.0$$
$$Q_{ij} = \lambda R_{ij} + \mu C_{ij} + 2\gamma = 0.80 \times R_{ij} + 2.50 \times C_{ij} + 4.0$$
$$\text{offset} = \gamma K^2 = 2.0 \times 4 = 8.0$$

Numerical values:
$$q = [-7.00, -6.95, -6.80, -6.75]^T$$
$$Q = \begin{bmatrix}
0.0 & 7.22 & 4.00 & 4.00 \\
0.0 & 0.0 & 4.00 & 4.00 \\
0.0 & 0.0 & 0.0 & 4.00 \\
0.0 & 0.0 & 0.0 & 0.0
\end{bmatrix}, \quad Q_{01} = 0.8(0.9) + 2.5(1) + 4.0 = 7.22$$

### 2.3 Independent Recalculation
- **Top-$K$ Solution**: Selects top 2 by utility $U_i \implies x_{\text{topk}} = [1, 1, 0, 0]^T$.
  - Linear cost: $q_0 + q_1 = -7.00 - 6.95 = -13.95$
  - Quadratic cost: $Q_{01} = +7.22$
  - Offset: $+8.00$
  - **Total Energy**: $H(x_{\text{topk}}) = -13.95 + 7.22 + 8.00 = \mathbf{+1.2700}$
  - **Feasibility**: Contradiction $(0, 1)$ violated $\implies$ **INFEASIBLE**.
- **Exact Global Optimum**: Selects $[1, 0, 1, 0]^T \implies \{e_0, e_2\}$.
  - Linear cost: $q_0 + q_2 = -7.00 - 6.80 = -13.80$
  - Quadratic cost: $Q_{02} = +4.00$
  - Offset: $+8.00$
  - **Total Energy**: $H(x_{\text{exact}}) = -13.80 + 4.00 + 8.00 = \mathbf{-1.8000}$
  - **Feasibility**: Budget error $= 0$, Contradictions $= 0 \implies$ **FEASIBLE**.
- **Optimality Gap**:
  $$\text{Gap}_{\text{topk}} = H(x_{\text{topk}}) - H(x_{\text{exact}}) = 1.2700 - (-1.8000) = \mathbf{3.0700}$$

**Conclusion for Instance 1**: The gap of 3.0700 and contradiction failure are mathematically exact and independently verified for this counterexample.

---

## 3. Deep Dive: Instance 2 — The Synthetic Ground-Truth Benchmark

**Source Script**: `research/pre_implementation_benchmark/scripts/harness_synthetic_benchmark.py` (`run_synthetic_benchmark()`)  
**Stored Artifact**: `research/pre_implementation_benchmark/results/synthetic_benchmark_summary.json`  

### 3.1 Problem Specification
- **Dimension**: $N = 10$ candidate variables, Budget $K = 4$.
- **Hyperparameters**: $\lambda = 0.60$, $\mu = 2.50$, $\gamma = 1.50$.
- **Synthetic Data Generating Process (DGP)**:
  - Latent factors: $z_{\text{mol0}}, z_{\text{mol1}} \sim \mathcal{N}(0, 1)$.
  - Direct causal propagation:
    - $z_{\text{struct0}} = 0.55 z_{\text{mol0}} + \epsilon_1 \implies r(\text{mol}_0, \text{struct}_0) \approx 0.50$ (Edge 0)
    - $z_{\text{struct1}} = 0.50 z_{\text{mol1}} + \epsilon_2 \implies r(\text{mol}_1, \text{struct}_1) \approx 0.45$ (Edge 1)
    - $z_{\text{motor0}} = 0.52 z_{\text{struct0}} + \epsilon_3 \implies r(\text{struct}_0, \text{motor}_0) \approx 0.48$ (Edge 2)
    - $z_{\text{motor1}} = 0.48 z_{\text{struct1}} + \epsilon_4 \implies r(\text{struct}_1, \text{motor}_1) \approx 0.42$ (Edge 3)
  - Indirect cross-edges (Distractors):
    - Edge 4: $\text{mol}_0 \to \text{struct}_1$ ($r \approx 0.25$)
    - Edge 5: $\text{struct}_0 \to \text{motor}_1$ ($r \approx 0.25$)
    - Edge 6: $\text{mol}_0 \to \text{motor}_0$ (Mediated shortcut, $r \approx 0.55 \times 0.52 \approx 0.28$)
    - Edge 7: $\text{struct}_1 \to \text{motor}_0$ ($r \approx 0.24$)
  - Pure noise:
    - Edge 8, Edge 9: ($r \approx 0.05$)
- **Constraints**:
  - Redundancies: $(0, 4), (1, 4), (2, 5), (3, 5)$.
  - Contradictions: $(2, 6)$ and $(2, 7)$.

### 3.2 Why Top-K Achieved Gap 0.0000 and F1 = 1.0000
1. In this synthetic DGP, the 4 planted causal edges $\{0, 1, 2, 3\}$ have direct causal links with correlations $r \in [0.42, 0.50]$.
2. The distractor edges $\{4, 5, 6, 7\}$ have indirect product correlations $r \in [0.24, 0.28]$.
3. The noise edges $\{8, 9\}$ have correlations $r \le 0.05$.
4. Consequently, in all 10 replicates, **Edges 0, 1, 2, and 3 had the 4 highest marginal utilities $U_i$**.
5. Crucially, **there are no collinear redundancies or contradictions among $\{0, 1, 2, 3\}$** (redundancies were between planted edges and distractors, contradictions were between Edge 2 and distractors 6/7).
6. When Top-$K$ selected the 4 items with highest utility, it selected:
   $$x_{\text{topk}} = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]^T$$
7. Because $\{0, 1, 2, 3\}$ has zero internal contradictions and zero internal redundancy, its quadratic cost is purely the minimum budget penalty ($2\gamma \binom{4}{2} = 12\gamma$).
8. Thus, $x_{\text{topk}}$ happened to coincide exactly with the global unconstrained minimum $x_{\text{exact}}$:
   $$\text{Energy}(\text{Top-}K) = -2.7204, \quad \text{Energy}(\text{Exact}) = -2.7204 \implies \mathbf{\text{Gap} = 0.0000}$$
   $$\text{Precision} = 1.0000, \quad \text{Recall} = 1.0000, \quad \mathbf{F1 = 1.0000}$$

**Methodological Takeaway**: The synthetic generator had an unintentional design bias: the marginal effect sizes of the true edges were so much stronger than the distractors that the problem was rank-separable by raw correlation alone. Top-$K$ did not encounter trade-offs between utility and quadratic penalties.

---

## 4. Deep Dive: Instance 3 — The PPMI Longitudinal Real Data Benchmark

**Source Script**: `research/pre_implementation_benchmark/scripts/harness_ppmi_discovery.py`  
**Stored Artifact**: `research/pre_implementation_benchmark/results/ppmi_discovery_summary.json`  

On real PPMI data across 15 cross-validation folds:
- Multimodal candidate correlations do NOT have a clean separation between true edges and distractors.
- For example, `asyn -> caudate` (Edge 0) and `ptau -> caudate` (Edge 1) both have marginal correlations around $r \approx 0.11$, but they share an explicit collinear redundancy ($R_{01} = 0.70$).
- When Top-$K$ selects both based on raw utility, it incurs the redundancy penalty $\lambda R_{01} x_0 x_1$.
- Exact M0 recognizes this penalty and swaps one out for an alternative non-redundant edge.
- **Result on Real PPMI Data**:
  - Exact M0 Energy: **-0.6273**
  - Top-$K$ Energy: **-0.3211**
  - **Top-$K$ Optimality Gap: 0.3062 ($\pm 0.45$)**
  - Top-$K$ Feasibility: 100% (no contradiction in top 4, but suboptimal due to redundancy).

---

## 5. Summary Comparison Matrix of the Three Instances

| Feature / Metric | Instance 1: Analytical Counterexample | Instance 2: Synthetic DGP Benchmark | Instance 3: PPMI Real-World Benchmark |
|:---|:---:|:---:|:---:|
| **Script** | `harness_mathematical_validation.py` | `harness_synthetic_benchmark.py` | `harness_ppmi_discovery.py` |
| **Variables ($N$)** | $N = 4$ | $N = 10$ | $N = 10$ |
| **Budget ($K$)** | $K = 2$ | $K = 4$ | $K = 4$ |
| **Hyperparameters** | $\lambda=0.8, \mu=2.5, \gamma=2.0$ | $\lambda=0.6, \mu=2.5, \gamma=1.5$ | $\lambda=0.5, \mu=2.5, \gamma=1.5$ |
| **Problem Structure** | Adversarially coupled: Top 2 contradict | Rank-separable: True causal links dominate | Real clinical data: Noisy & collinear |
| **Top-$K$ Selected Indices** | $\{0, 1\}$ | $\{0, 1, 2, 3\}$ | Varies by fold |
| **Exact M0 Selected Indices** | $\{0, 2\}$ | $\{0, 1, 2, 3\}$ | Varies by fold |
| **Top-$K$ Energy** | **+1.2700** | **-2.7204** | **-0.3211** |
| **Exact M0 Energy** | **-1.8000** | **-2.7204** | **-0.6273** |
| **Top-$K$ Optimality Gap** | **3.0700** | **0.0000** | **0.3062** |
| **Top-$K$ Feasibility** | **0.0% (Violates Contradiction)** | **100.0%** | **100.0%** |
| **Top-$K$ F1 vs Ground Truth** | N/A | **1.0000** | N/A (Ground truth unknown) |

---

## 6. Required Corrections to Documentation

The following corrections must be updated in all benchmark reports:
1. **Clarify Question D in Review Matrix**:
   - Explicitly state that the **3.0700 optimality gap** refers specifically to the **4-variable analytical counterexample** used to demonstrate theoretical non-separability under adversarial collinearity and contradiction.
2. **Clarify Synthetic Benchmark Table**:
   - Explicitly note that the **0.0000 gap** in the synthetic benchmark was an artifact of the synthetic DGP where true causal edges had significantly higher marginal correlations than distractors ($0.48$ vs $0.25$), rendering the synthetic problem rank-separable.
3. **Present Real PPMI Top-$K$ Gap**:
   - On real PPMI data, Top-$K$ achieves an optimality gap of **0.3062**, accurately reflecting that real-world collinearity causes naive ranking to be suboptimal compared to quadratic optimization.
