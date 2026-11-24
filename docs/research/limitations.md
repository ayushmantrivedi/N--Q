# Methodological & Technical Limitations — NĀḌĪ-Q

> This document is a first-class scientific research artifact. Honest, rigorous documentation of system limitations is an essential prerequisite for biomedical AI and quantum computing credibility.

---

## 1. Quantum Computing Limitations

### 1.1 Simulator Execution vs Physical QPUs [A]
All quantum circuits within NĀḌĪ-Q currently execute on high-performance classical simulators (PennyLane `default.qubit` and `lightning.qubit`).
- **Implication**: While mathematical state-vector simulation is noise-free and exact, it does not demonstrate quantum speedup or physical device resilience.
- **Physical QPU Gap**: Real NISQ devices (e.g. superconducting transmons or trapped ions) suffer from cumulative two-qubit gate errors ($0.5\%\text{--}2.0\%$), thermal relaxation ($T_1$), dephasing ($T_2$), and readout crosstalk.
- **Mitigation in Architecture A**: By keeping clinical inference purely classical, physical QPU errors, timeouts, or simulator crashes have zero influence on patient progression predictions ($\Delta_{\max} = 0.00\text{e}{+}00$).

### 1.2 Combinatorial Scale & Classical Solver Dominance [A/C]
In Plane 1, the biological interaction space across current multimodal features consists of $N=15$ candidate edges.
- **Empirical Reality**: At $N=15$, Classical Tabu Search ($M_3$) solves the QUBO to global optimality in **127 ms**, whereas PennyLane QAOA ($p=1, 2$) requires several seconds on CPU simulation.
- **Scientific Honesty**: Quantum QAOA does not provide a practical runtime advantage at $N=15$. It serves as a proof-of-concept optimization sidecar for future high-dimensional biological networks ($N > 50\text{--}100$).

### 1.3 Barren Plateaus and Expressibility Trade-Offs [A]
Deep variational circuits are mathematically vulnerable to barren plateaus (exponentially vanishing gradients). QVBE-v4 mitigates this via shallow depth ($L=2$) and Fubini-Study regularization, but scaling to $\ge 20$ qubits will require local cost functions and structured warm-starts.

---

## 2. Biomedical Data & Clinical Validation Limitations

### 2.1 Sample Size Constraints ($N=341$) [A]
While NĀḌĪ-Q uses the authoritative longitudinal PPMI cohort, the effective cohort size is $N=341$ participants with 159 progression events.
- **Statistical Power**: With $N=341$, the minimum detectable effect size (MDE) at $\alpha=0.05$ and $80\%$ power is Pearson $|r| \ge 0.151$. Very weak biological signals ($|r| < 0.15$) cannot be reliably resolved after multiple-testing correction.
- **High-Dimensional Risk**: Feeding wide multimodal matrices into over-parameterized models risks empirical overfitting. Strict 5-fold cross-validation and regularized baselines are enforced to prevent over-optimistic claims.

### 2.2 Endpoint Definitions & Motor Scale Noise [A]
The primary clinical endpoint is Task T2 (progression $\ge 5.0$ points on MDS-UPDRS Part III from Month 12 to Month 60).
- **Subjectivity**: While MDS-UPDRS Part III is the gold-standard motor rating scale, it possesses intra-rater and inter-rater variability of $\approx 2\text{--}4$ points.
- **Medication Effects**: Patient state (ON vs OFF medication) introduces variance in clinical scores. PPMI protocols attempt to standardize this, but clinical noise persists.

### 2.3 Generalizability & External Cohort Validation [A]
- **Single-Study Bias**: Current models are validated within PPMI. Performance across external clinical cohorts (e.g., ADNI, DATATOP, or routine outpatient EHRs) has not yet been demonstrated.
- **Exploratory Labeling**: In Plane 1, all discovered biological relationships are explicitly designated as **`Provisional/Explanatory`** and must not be used for therapeutic claims without independent prospective laboratory validation.

---

## 3. Regulatory & Clinical Deployment Scope

- **Non-Diagnostic Tool**: NĀḌĪ-Q is an investigational research platform, not an FDA/EMA-cleared Software as a Medical Device (SaMD).
- **Decision Support Only**: Output probabilities are intended solely to provide structured risk stratification for clinical researchers and trial enrollment, not autonomous medical decision-making.
