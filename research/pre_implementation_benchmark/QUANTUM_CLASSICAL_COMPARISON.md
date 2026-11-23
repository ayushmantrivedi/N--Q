# NĀḌĪ-Q Quantum vs Classical Optimization Benchmark
## Systematic Head-to-Head Comparison: Exact, Tabu, Greedy, Simulated Annealing, and QAOA ($p=1, 2$)

**Date**: November 23, 2026  
**Auditing Specialist**: Quantum Optimization Researcher & Systems Architect  
**Empirical Datasets**: `research/pre_implementation_benchmark/results/ppmi_discovery_results.csv`, `synthetic_benchmark_results.csv`  

---

## 1. Systematic Solver Comparison Matrix

The table below compiles empirical performance across all evaluated solvers on the authoritative 15-fold PPMI multimodal graph selection benchmark ($N=10$ candidate relationships, budget $K=4$, $\lambda=0.5, \mu=2.5, \gamma=1.5$):

| Metric | M0: Exact Exhaustive | M3: Tabu Search | M2: Greedy Selection | M1: Simulated Annealing | QAOA $p=1$ | QAOA $p=2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mean Minimization Energy** | **-0.6273** | **-0.6262** | -0.6011 | -0.5662 | +1.0391 | +0.0319 |
| **Optimality Gap vs Exact** | **0.0000** | **0.0011** | 0.0262 | 0.0611 | 1.6664 | 0.6593 |
| **Constraint Feasibility Rate** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 53.3% | 80.0% |
| **Budget Error ($|\sum x - K|$)** | **0.00** | **0.00** | **0.00** | **0.00** | 0.47 | 0.20 |
| **Mean Contradictions** | **0.00** | **0.00** | **0.00** | **0.00** | 0.20 | 0.00 |
| **Held-Out Test Replication** | 6.67% | 6.67% | 6.67% | 6.67% | 8.78% | 7.33% |
| **Effect Direction Consistency**| 56.67% | 56.67% | 58.33% | 61.67% | 57.44% | 63.00% |
| **Selection Stability (Jaccard)**| 0.4656 | 0.4249 | **0.4964** | 0.2944 | 0.3181 | 0.3167 |
| **Mean Runtime (ms)** | **31.4 ms** | **74.8 ms** | **45.4 ms** | **67.0 ms** | 7,196.5 ms | 12,930.3 ms |
| **Relative Speed vs QAOA $p=2$**| **$411\times$ faster**| **$173\times$ faster**| **$285\times$ faster**| **$193\times$ faster**| $1.8\times$ faster | Baseline ($1\times$) |
| **Evaluations per Run** | 1,024 | 1,501 | 34 | 501 | 75 | 75 |
| **Quantum Circuit Depth** | 0 | 0 | 0 | 0 | 4 | 7 |
| **Total Gate Count** | 0 | 0 | 0 | 0 | 110 | 210 |

---

## 2. Quantum Circuit Specifications & Gate Count Audit

QAOA circuits were compiled using PennyLane's `default.qubit` simulator with standard parameterized ansatzes:
- **Cost Hamiltonian Unitary**: $U(H_C, \gamma) = e^{-i \gamma H_C} = \prod_i e^{-i \gamma h_i Z_i} \prod_{i < j} e^{-i \gamma J_{ij} Z_i Z_j}$
- **Mixer Unitary**: $U(H_M, \beta) = e^{-i \beta H_M} = \prod_i e^{-i \beta X_i}$

### Quantum Gate Breakdown ($N=10$ Qubits)

| Circuit Layer | 1-Qubit Hadamard ($H$) | 2-Qubit Entangling ($\text{CNOT}$) | Parameterized Rotation ($R_Z$) | Parameterized Mixer ($R_X$) | Total Gate Count | Circuit Depth |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Initial State Preparation** | 10 | 0 | 0 | 0 | 10 | 1 |
| **QAOA Layer 1 ($p=1$)** | 0 | 45 | 45 | 10 | 100 | 3 |
| **QAOA Layer 2 ($p=2$)** | 0 | 45 | 45 | 10 | 100 | 3 |
| **Total QAOA $p=1$** | **10** | **45** | **45** | **10** | **110** | **4** |
| **Total QAOA $p=2$** | **10** | **90** | **90** | **20** | **210** | **7** |

---

## 3. The "Scale Inadequacy Barrier" for Quantum Advantage

A foundational question for investment is whether QAOA offers, or can offer, any computational advantage for biological relationship discovery.

### 3.1 Problem Size vs Classical Search Capacity
- In the NĀḌĪ-Q multimodal schema (Molecular $\times$ Structural MRI $\times$ Motor), the candidate relationship pool comprises $N = 10$ to $N = 20$ plausible pairwise associations.
- At $N = 10$, the total binary configuration space is:
  $$2^{10} = 1,024 \text{ states}$$
  A single modern CPU core evaluates the exact energy of all 1,024 states in **31 milliseconds** using vectorized NumPy/BLAS operations.
- At $N = 20$, the configuration space is:
  $$2^{20} \approx 1.05 \times 10^6 \text{ states}$$
  Vectorized integer quadratic programming or branch-and-bound solvers (e.g. SCIP, Gurobi, or Tabu search) resolve this in **less than 0.5 seconds**.

### 3.2 Computational Overhead of Quantum Simulation
- On classical simulators, evaluating expectation values $\langle \psi | H_C | \psi \rangle$ requires simulating the full $2^{10}$-dimensional statevector ($1,024$ complex amplitudes). Each step of gradient descent performs multiple forward/backward passes.
- Consequently, QAOA $p=2$ requires **12.9 seconds** on local CPU—which is **173 times slower than Tabu search** and **411 times slower than exhaustive search**.

### 3.3 Physical QPU Execution Constraints
Even if executed on physical superconducting quantum hardware (e.g. IBM Quantum or Rigetti):
1. **Compilation & Communication Latency**: Submitting jobs, queuing, calibration, and readout latency on current cloud QPUs takes between 10 and 60 seconds per job.
2. **Measurement Shot Noise**: Evaluating expectation values requires $M \ge 1,000$ measurement shots per circuit execution to achieve $\mathcal{O}(1/\sqrt{M})$ statistical error. Across 25 optimization steps with 3 restarts, this requires 75,000 circuit executions.
3. **Two-Qubit Gate Error**: QAOA $p=2$ requires **90 CNOT gates**. On state-of-the-art QPUs with two-qubit gate error rates of $\epsilon \approx 0.5\% - 1.0\%$, the overall circuit fidelity is approximately $(1 - \epsilon)^{90} \approx 0.995^{90} \approx 63.6\%$ (meaning $>36\%$ of circuit runs suffer depolarizing errors).

**Mathematical & Systems Verdict**: Claiming a computational speedup or search quality advantage for QAOA on graphs of size $N \le 20$ is mathematically and practically impossible. Classical solvers (Exact, Tabu, Greedy) strictly dominate QAOA in solution quality, runtime, and constraint feasibility.
