# NĀḌĪ-Q Controlled Synthetic Ground-Truth Benchmark Report
## Evaluation of Planted Causal Association Recovery, Noise Robustness, and Solver Performance Across 10 Independent Replicates

**Date**: November 23, 2026  
**Auditing Specialist**: Biomedical Data Scientist & Statistical Validation Specialist  
**Artifacts**: `research/pre_implementation_benchmark/results/synthetic_benchmark_results.csv`, `synthetic_benchmark_summary.json`  

---

## 1. Benchmark Design & Ground-Truth Specification

To evaluate whether optimization algorithms can reliably recover true biological relationships amidst collinearity, contradiction, and noise, `harness_synthetic_benchmark.py` established a controlled synthetic testbed:
- **Sample Size**: $N = 341$ longitudinal multimodal participants per replicate, exactly matched to the PPMI cohort size.
- **Replications**: 10 independent random seeds (Seeds 101 through 110).
- **Candidate Pool ($N=10$)**:
  - **4 Planted True Causal Relationships** (The true optimal ground truth $\mathcal{S}^* = \{0, 1, 2, 3\}$):
    - $e_0$: $\text{mol}_0 \to \text{struct}_0$ ($r \approx 0.50, p < 0.001$)
    - $e_1$: $\text{mol}_1 \to \text{struct}_1$ ($r \approx 0.45, p < 0.001$)
    - $e_2$: $\text{struct}_0 \to \text{motor}_0$ ($r \approx 0.48, p < 0.001$)
    - $e_3$: $\text{struct}_1 \to \text{motor}_1$ ($r \approx 0.42, p < 0.001$)
  - **2 Collinear Redundant Distractors**:
    - $e_4$: $\text{mol}_0 \to \text{struct}_1$ ($R_{04} = 0.75, R_{14} = 0.70$)
    - $e_5$: $\text{struct}_0 \to \text{motor}_1$ ($R_{25} = 0.70, R_{35} = 0.65$)
  - **2 Contradictory Incompatible Relationships**:
    - $e_6$: $\text{mol}_0 \to \text{motor}_0$ (Direct shortcut contradicting intermediate cascade $e_0 + e_2$)
    - $e_7$: $\text{struct}_1 \to \text{motor}_0$ (Conflicting convergence on $\text{motor}_0$)
  - **2 Pure Noise Associations**:
    - $e_8$: $\text{noise}_0 \to \text{motor}_0$ ($r \approx 0.05, p \approx 0.35$)
    - $e_9$: $\text{noise}_1 \to \text{struct}_1$ ($r \approx -0.04, p \approx 0.45$)
- **Selection Budget**: $K = 4$.

---

## 2. Experimental Benchmark Results

Across all 10 replicates, each solver's selected edge set was evaluated against the planted ground truth $\mathcal{S}^*$:
- **Precision**: $|\mathcal{S} \cap \mathcal{S}^*| / |\mathcal{S}|$
- **Recall**: $|\mathcal{S} \cap \mathcal{S}^*| / |\mathcal{S}^*|$
- **F1 Score**: $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$
- **Selection Stability**: Pairwise Jaccard similarity across the 10 replicates ($\binom{10}{2} = 45$ pairs).
- **Feasibility Rate**: Fraction of runs with $| \sum x_i - K | = 0$ and $0$ contradictions.

### Summary Results Table (10 Replicates, 70 Evaluations)

| Solver Method | Mean Energy | Optimality Gap | Feasibility Rate | Mean Precision | Mean Recall | Mean F1 Score | Selection Stability | Mean Runtime (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0: Exact Exhaustive** | **-2.7204** | **0.0000** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **32.5 ms** |
| **M2: Greedy Selection** | **-2.7204** | **0.0000** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **32.0 ms** |
| **M3: Tabu Search** | **-2.7204** | **0.0000** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **69.4 ms** |
| **Top-K Heuristic** | **-2.7204** | **0.0000** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.33 ms** |
| **M1: Simulated Annealing** | -2.3289 | 0.3915 | 100.0% | 0.7750 | 0.7750 | 0.7750 | 0.5145 | 50.3 ms |
| **QAOA $p=1$** | -0.5705 | 2.1499 | 40.0% | 0.4783 | 0.5500 | 0.5091 | 0.3014 | 1,531.4 ms |
| **QAOA $p=2$** | -0.5023 | 2.2181 | 70.0% | 0.4517 | 0.5000 | 0.4717 | 0.2852 | 2,018.3 ms |

---

## 3. Detailed Findings & Critical Analysis

### 3.1 Perfect Classical Ground-Truth Recovery
- **M0 (Exact Exhaustive)**, **M2 (Greedy)**, and **M3 (Tabu Search)** achieved **100% Precision, 100% Recall, and 1.000 F1 Score** across every single replicate (10/10).
- In every replicate, the exact global minimum matched the planted ground-truth causal graph $\{0, 1, 2, 3\}$, successfully rejecting the collinear distractors ($e_4, e_5$), contradictory shortcuts ($e_6, e_7$), and noise edges ($e_8, e_9$).
- **Runtime**: Classical solvers resolved the ground truth in **32 ms to 69 ms**.

### 3.2 Severe Failure of QAOA on Ground-Truth Recovery
- **QAOA $p=1$**:
  - F1 Score dropped to **0.5091** ($\pm 0.18$).
  - Feasibility rate was only **40.0%** (violating budget or contradiction penalties in 6 out of 10 runs).
  - Selection stability collapsed to **0.3014**.
  - Runtime was **1,531 ms** ($47\times$ slower than Exact).
- **QAOA $p=2$**:
  - F1 Score dropped to **0.4717** ($\pm 0.16$).
  - Feasibility rate was **70.0%** (violating constraints in 3 out of 10 runs).
  - Optimality gap averaged **2.2181** ($\pm 1.83$).
  - Runtime was **2,018 ms** ($62\times$ slower than Exact, $29\times$ slower than Tabu).

### 3.3 Root Cause of QAOA Failure
1. **Shallow Circuit Parameter Landscape**: At depth $p \in \{1, 2\}$, the parameterized ansatz has only 2 to 4 angles $(\gamma_k, \beta_k)$. On a non-uniform Ising graph with 10 qubits and 45 interaction terms, the expectation landscape contains numerous shallow local minima and barren-plateau-like regions where gradient descent terminates far from the ground state.
2. **Soft Constraint Inadequacy**: QAOA samples from a quantum statevector $|\psi(\gamma, \beta)|^2$. In states with high superposition entropy, bitstrings violating the budget constraint or contradiction penalty frequently have non-zero probability and are selected as top candidates.
3. **Simulation Scale Mismatch**: Because $N=10$, classical exhaustive search evaluates $2^{10} = 1,024$ states directly in vectorized CPU memory in 32 ms. Simulating unitary matrix multiplications for 10 qubits requires thousands of complex floating-point operations per step, yielding high runtime with poor solution quality.

**Conclusion**: On controlled ground-truth biological graphs, classical optimization methods (Exact, Greedy, Tabu) achieve perfect causal recovery ($F1 = 1.000$), while QAOA degrades causal discovery ($F1 \approx 0.49$) and violates constraints.
