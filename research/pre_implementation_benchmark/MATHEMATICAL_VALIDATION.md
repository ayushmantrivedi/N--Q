# NĀḌĪ-Q Mathematical & Formulation Validation Report
## Rigorous Formulation Derivation, Ising Equivalence, and Objective Non-Separability Analysis

**Date**: November 23, 2026  
**Auditing Specialist**: Quantum Optimization & Mathematical Formulations Auditor  
**Result Artifact**: `research/pre_implementation_benchmark/results/mathematical_validation_results.json`  

---

## 1. Problem Formulation & Exact Mapping

### 1.1 Biological Relationship Selection Objective
Let $\mathcal{E} = \{e_1, \dots, e_N\}$ be a candidate pool of $N$ biological relationships discovered from longitudinal multimodal data. Each edge $e_i$ is associated with a binary decision variable $x_i \in \{0, 1\}$, where $x_i = 1$ denotes selecting relationship $e_i$.

The biological goal is to maximize evidence utility subject to:
1. Penalizing pairwise collinear redundancy $R_{ij} \in [0, 1]$.
2. Enforcing mutual exclusivity on contradictory edge pairs $(i, j) \in \mathcal{C}$.
3. Restricting the total number of selected edges to a target budget $K$.

The **maximization objective** is formulated as:
$$\text{Obj}_{\max}(x) = \sum_{i=1}^N U_i x_i - \lambda \sum_{i < j} R_{ij} x_i x_j - \mu \sum_{(i,j) \in \mathcal{C}} x_i x_j - \gamma \left( \sum_{i=1}^N x_i - K \right)^2$$

Where:
- $U_i > 0$ is the evidence-based utility of edge $e_i$ derived from MeTTa symbolic confidence ($U_i = |r_i| \cdot (1 - p_i) \cdot c_i$).
- $\lambda \ge 0$ is the redundancy penalty coefficient.
- $\mu \ge 0$ is the contradiction penalty coefficient.
- $\gamma \ge 0$ is the budget constraint penalty coefficient.
- $K \in \mathbb{N}$ is the target edge budget.

---

### 1.2 Transformation to QUBO Minimization Energy
To express the problem as a standard Quadratic Unconstrained Binary Optimization (QUBO) minimization:
$$H(x) = -\text{Obj}_{\max}(x) = \sum_{i=1}^N q_i x_i + \sum_{i < j} Q_{ij} x_i x_j + \text{const}$$

Expanding the quadratic budget penalty:
$$\left( \sum_{i=1}^N x_i - K \right)^2 = \left( \sum_{i=1}^N x_i \right)^2 - 2K \sum_{i=1}^N x_i + K^2$$

Since $x_i \in \{0, 1\}$, we have $x_i^2 = x_i$, which yields:
$$\left( \sum_{i=1}^N x_i \right)^2 = \sum_{i=1}^N x_i^2 + 2 \sum_{i < j} x_i x_j = \sum_{i=1}^N x_i + 2 \sum_{i < j} x_i x_j$$

Substituting back:
$$\gamma \left( \sum_{i=1}^N x_i - K \right)^2 = \gamma (1 - 2K) \sum_{i=1}^N x_i + 2\gamma \sum_{i < j} x_i x_j + \gamma K^2$$

Negating $\text{Obj}_{\max}(x)$ and collecting linear and quadratic terms yields:
$$q_i = -U_i + \gamma (1 - 2K)$$
$$Q_{ij} = \lambda R_{ij} + \mu C_{ij} + 2\gamma \quad (\text{for } i < j)$$
$$\text{offset} = \gamma K^2$$

Where $C_{ij} = 1$ if $(i, j) \in \mathcal{C}$ or $(j, i) \in \mathcal{C}$, else $0$.

---

### 1.3 Exact Mapping to Ising Spin Hamiltonian
To execute optimization on quantum hardware or simulators, binary variables $x_i \in \{0, 1\}$ are mapped to Pauli-$Z$ spin operators $Z_i \in \{+1, -1\}$ via the change of basis:
$$x_i = \frac{I - Z_i}{2}$$

Substituting this transformation into the bilinear products:
$$x_i x_j = \left( \frac{I - Z_i}{2} \right) \left( \frac{I - Z_j}{2} \right) = \frac{1}{4} I - \frac{1}{4} Z_i - \frac{1}{4} Z_j + \frac{1}{4} Z_i Z_j$$

Expanding the full Hamiltonian:
$$H(Z) = \sum_{i=1}^N q_i \left( \frac{I - Z_i}{2} \right) + \sum_{i < j} Q_{ij} \left( \frac{1}{4} I - \frac{1}{4} Z_i - \frac{1}{4} Z_j + \frac{1}{4} Z_i Z_j \right) + \text{offset} \cdot I$$

Grouping into standard Ising form:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + c_0 I$$

We obtain the exact coupling coefficients:
$$h_i = -\frac{1}{2} q_i - \frac{1}{4} \sum_{j \neq i} Q_{ij}^{\text{symm}}$$
$$J_{ij} = \frac{1}{4} Q_{ij} \quad (\text{for } i < j)$$
$$c_0 = \text{offset} + \frac{1}{2} \sum_{i=1}^N q_i + \frac{1}{4} \sum_{i < j} Q_{ij}$$

Where $Q^{\text{symm}} = Q + Q^T$ is the symmetrized quadratic interaction matrix.

---

## 2. Empirical Equivalence & Floating-Point Error Audit

To verify that the mapping introduces zero algebraic or numerical distortion, `harness_mathematical_validation.py` evaluated the absolute error:
$$\Delta(x) = |H_{\text{QUBO}}(x) - H_{\text{Ising}}(1 - 2x)|$$
across 500 randomly sampled bitstrings per graph size for $N \in \{6, 8, 10, 12, 14\}$ (2,500 total states):

| Graph Size ($N$) | Number of States Checked | Max Absolute Error ($\max \Delta$) | Mean Absolute Error | Equivalence to $10^{-9}$ |
|:---:|:---:|:---:|:---:|:---:|
| $N = 6$ | 500 | $5.33 \times 10^{-15}$ | $1.78 \times 10^{-15}$ | **PASS (Exact)** |
| $N = 8$ | 500 | $1.42 \times 10^{-14}$ | $4.21 \times 10^{-15}$ | **PASS (Exact)** |
| $N = 10$ | 500 | $2.13 \times 10^{-14}$ | $7.65 \times 10^{-15}$ | **PASS (Exact)** |
| $N = 12$ | 500 | $5.68 \times 10^{-14}$ | $1.89 \times 10^{-14}$ | **PASS (Exact)** |
| $N = 14$ | 500 | $5.68 \times 10^{-14}$ | $2.12 \times 10^{-14}$ | **PASS (Exact)** |

**Mathematical Verdict**: The mapping between QUBO and Ising formulation is algebraically exact, with errors bounded strictly by 64-bit IEEE-754 floating-point machine precision ($\sim 10^{-14}$), well below the verification threshold of $10^{-9}$.

---

## 3. Objective Non-Separability Proof & Top-$K$ Failure

### 3.1 Separability Condition
A binary optimization problem $\min_x f(x)$ subject to $\sum_i x_i = K$ is **separable** if and only if:
$$\frac{\partial^2 f(x)}{\partial x_i \partial x_j} = 0 \quad \forall i \neq j$$
Under separability, $f(x) = \sum_i f_i(x_i)$. The global optimum is obtained trivially in $\mathcal{O}(N \log N)$ by sorting variables by $f_i(1) - f_i(0)$ and selecting the top $K$ items.

### 3.2 Non-Separability under Redundancy & Contradictions
For the biological QUBO:
$$\frac{\partial^2 H(x)}{\partial x_i \partial x_j} = Q_{ij} = \lambda R_{ij} + \mu C_{ij} + 2\gamma$$
- When $\lambda = 0$ and $\mu = 0$, $Q_{ij} = 2\gamma$ is a rank-1 uniform interaction matrix. The budget constraint preserves variable ordering, meaning Top-$K$ ranking by utility $U_i$ recovers the exact constrained optimum.
- When $\lambda > 0$ or $\mu > 0$, $Q_{ij}$ is non-uniform and depends on the graph topology. The Hessian matrix is full-rank and non-separable, mapping directly to an NP-hard Ising spin glass.

### 3.3 Empirical Demonstration of Top-$K$ Heuristic Failure
`harness_mathematical_validation.py` constructed a 4-variable problem with budget $K=2$:
- Candidates:
  - $e_0$: $U_0 = 1.00$ (Highest single utility)
  - $e_1$: $U_1 = 0.95$ (Second highest utility)
  - $e_2$: $U_2 = 0.80$ (Third highest utility)
  - $e_3$: $U_3 = 0.75$ (Fourth highest utility)
- Redundancy & Contradiction: $e_0$ and $e_1$ are collinear ($R_{01} = 0.90, \lambda = 0.80$) and mutually contradictory ($(0, 1) \in \mathcal{C}, \mu = 2.50$).

**Results**:
- **Naive Top-$K$ Selection**: Selects $[e_0, e_1]$ based on raw utility.
  - Solution: $x = [1, 1, 0, 0]$
  - Energy: $+1.2700$
  - Feasibility: **False** (Violates mutual contradiction: $e_0 \wedge e_1 = 1$)
- **Exact Global Optimum**: Selects $[e_0, e_2]$.
  - Solution: $x = [1, 0, 1, 0]$
  - Energy: **$-1.8000$**
  - Feasibility: **True** (0 contradictions, exactly $K=2$ edges)
- **Optimality Gap**: $\Delta E = 1.2700 - (-1.8000) = \mathbf{3.0700}$.

**Conclusion**: Naive Top-$K$ ranking fails catastrophically on non-separable biological graphs, proving that quadratic graph optimization is mathematically necessary to resolve collinearity and biological contradictions.

---

## 4. Constraint Penalty Scaling Analysis

To guarantee that the unconstrained global minimum of $H(x)$ is strictly feasible:
1. **Contradiction Penalty Condition**:
   If an algorithm activates a contradictory pair $(i, j) \in \mathcal{C}$, it gains utility at most $U_i + U_j$. The penalty incurred is $\mu$. For the state to have higher energy than a non-contradictory state:
   $$\mu > \max_{(i,j) \in \mathcal{C}} (U_i + U_j)$$
   - In PPMI, typical single utilities range from $0.10$ to $0.60$ ($\max U_i \approx 0.85$, $\max (U_i + U_j) \approx 1.70$).
   - Current parameter: $\mu = 2.50 > 1.70$ (Safety margin: $+0.80$).

2. **Budget Penalty Condition**:
   If an algorithm selects $K+1$ variables, it gains at most $\max_i U_i$ in utility, but incurs a penalty $\gamma \cdot [(K+1) - K]^2 = \gamma$. To guarantee budget fidelity:
   $$\gamma > \max_i U_i$$
   - Current parameter: $\gamma = 1.50 > 0.85$ (Safety margin: $+0.65$).

**Audit Verdict**: Both penalty parameters ($\gamma=1.50, \mu=2.50$) satisfy the theoretical lower bounds for hard constraint enforcement at the global optimum.
