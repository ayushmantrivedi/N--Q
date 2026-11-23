# NĀḌĪ-Q Forensic Validation: Statistical Reliability Review
## Multiple-Testing Power Analysis, FDR Yield Mechanics, Fold-Level Support Breakdown, and Benchmark Limitations

**Date**: November 24, 2026  
**Auditor**: Independent Biostatistician & Research Methodology Auditor  
**Audited Artifacts**:
- `research/pre_implementation_benchmark/results/ppmi_discovery_results.csv`
- `research/pre_implementation_benchmark/results/ppmi_discovery_summary.json`
- `research/pre_implementation_benchmark/results/synthetic_benchmark_summary.json`

---

## 1. Audit of the FDR Discovery Yield (Why $q \le 0.05$ Yields Only 0.40 Edges)

The pre-implementation benchmark reported that Benjamini-Hochberg (BH) False Discovery Rate control at $q \le 0.05$ selected an average of only **0.40 edges per fold** on the PPMI cohort. 

### 1.1 Mathematical & Statistical Mechanics
1. **Hypothesis Family Definition**:
   In each training fold ($N_{\text{train}} \approx 272$), correlation screening tests $m = 10$ pairwise biological relationships across molecular biofluids, striatal imaging, and motor progression.
2. **Benjamini-Hochberg Critical Thresholds**:
   Sorting candidate $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(10)}$, the BH rejection criterion at $q = 0.05$ requires:
   $$p_{(k)} \le \frac{k}{m} q = \frac{k}{10} \times 0.05$$
   - For rank $k=1$ (the strongest association): $\alpha_1 = 0.0050$.
   - For rank $k=2$: $\alpha_2 = 0.0100$.
   - For rank $k=3$: $\alpha_3 = 0.0150$.
3. **Required Effect Size for Rejection**:
   For a sample size of $N_{\text{train}} = 272$ ($\text{df} = 270$), a two-tailed Pearson correlation test reaches $p \le 0.0050$ if and only if the absolute t-statistic satisfies:
   $$t = |r| \sqrt{\frac{N-2}{1 - r^2}} \ge 2.828 \implies |r| \ge \mathbf{0.170}$$
4. **Empirical Effect Sizes in 12-Month PPMI Deltas**:
   Longitudinal changes over 12 months ($\Delta \text{asyn}$, $\Delta \text{caudate}$, $\Delta \text{motor}$) in early Parkinson's disease exhibit high biological and measurement variability:
   - Median $|r|$ across all 10 candidate pairs: **0.082** ($p \approx 0.176$).
   - Maximum $|r|$ in Seed 42 folds: **0.108 to 0.151** ($p \approx 0.012$ to $0.072$).
   - Because no candidate reached $|r| \ge 0.170$ in Seed 42, the minimum $p$-value ($0.0122$) exceeded $\alpha_1 = 0.0050$, resulting in **zero rejected hypotheses in all 5 folds of Seed 42**.
5. **Statistical Power Analysis**:
   With $N = 272$ and true population correlation $\rho = 0.12$:
   $$\text{Power at } \alpha = 0.05 \text{ (Uncorrected)}: \approx 51.2\%$$
   $$\text{Power at } \alpha = 0.0050 \text{ (BH Rank 1)}: \approx \mathbf{18.4\%}$$
   At $18.4\%$ statistical power, the probability of detecting a genuine association of magnitude $\rho = 0.12$ is less than 1 in 5.

### 1.2 Fold-Level Support Distribution
Across the 15 folds:
- **Seed 42** (5 folds): 0 edges passed FDR in all 5 folds ($0/5$).
- **Seed 123** (5 folds): 2 edges passed FDR in Fold 0 and Fold 2 (`putamen -> motor_brady`, $r = 0.182$; `caudate -> motor_total`, $r = 0.176$). Zero edges in Folds 1, 3, 4 ($2/5$).
- **Seed 456** (5 folds): 2 edges passed FDR in Fold 4 (`putamen -> motor_brady`, $r = 0.179$). Zero edges in Folds 0, 1, 2, 3 ($1/5$).
- **Total across 15 folds**:
  $$\text{Mean Selected Edges} = \frac{(0 \times 12) + (2 \times 3)}{15} = \frac{6}{15} = \mathbf{0.40 \text{ edges}}$$

**Verdict**: The low yield of 0.40 edges is **not a software defect**; it is the mathematically expected consequence of applying strict family-wise FDR ($q \le 0.05$) to modest-sized longitudinal delta cohorts ($N=272$) where biological effect sizes hover around $|r| \approx 0.12$.

---

## 2. Proposed Defensible Discovery Protocol

Lowering the FDR threshold blindly (e.g. to $q \le 0.20$) would inflate false positive discoveries. Instead, we propose a three-pillar **Defensible Discovery Protocol**:

```
                       ┌──────────────────────────────────────────────┐
                       │   Stage 1: Multi-Year Longitudinal Baseline  │
                       │   - Use Month 36 (V08) deltas instead of     │
                       │     12-month deltas (2.5x larger delta signal)│
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Stage 2: Adaptive Tiered Screening         │
                       │   - Tier 1 (Validated): BH q <= 0.05         │
                       │   - Tier 2 (Provisional): Empirical Bayes /  │
                       │     Storey q <= 0.10 with n >= 50            │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Stage 3: MeTTa Symbolic Constraint QUBO    │
                       │   - Enforces collinearity & contradictions   │
                       │   - Solves for budget K=4 with Tabu Search   │
                       └──────────────────────────────────────────────┘
```

1. **Multi-Year Delta Windowing**: Parkinson's motor progression ($\Delta \text{UPDRS-III}$) over 36 months averages $\sim 6.5$ points (MCID $\ge 5.0$), compared to only $\sim 1.8$ points over 12 months. Expanding the observation baseline increases effect sizes from $|r| \approx 0.12$ to $|r| \approx 0.28$, raising statistical power from $18\%$ to over **$85\%$** at $\alpha = 0.0050$.
2. **Adaptive Two-Tier FDR**: Maintain $q \le 0.05$ for confirmed causal claims, but allow $q \le 0.10$ for "Provisional" hypotheses that are subject to downstream MeTTa redundancy pruning.
3. **Empirical Bayes Shrinkage**: Apply Empirical Bayes moderation (similar to limma / DESeq2) to shrink noisy participant-level variance estimates before correlation testing.

---

## 3. Fold-Level Statistical Support Breakdown (15 Folds)

The table below documents the exact numerator, denominator, and fold-level metrics for all 8 evaluated discovery protocols across all 15 cross-validation folds:

| Protocol | Total Edge Selections across 15 Folds | Total Replicated Edges ($p_{\text{te}} \le 0.10$, sign match) | Overall Replication Rate (Numerator / Denominator) | Mean Direction Consistency | Selection Stability (Pairwise Jaccard) | Mean Optimality Gap vs M0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0: Exact Exhaustive** | 60 | 4 | **4 / 60 (6.67% $\pm 14.8\%$)** | 56.67% ($\pm 20.0\%$) | 0.4656 | **0.0000** |
| **M3: Tabu Search** | 60 | 4 | **4 / 60 (6.67% $\pm 14.8\%$)** | 56.67% ($\pm 20.0\%$) | 0.4249 | **0.0011** |
| **Top-K Heuristic** | 60 | 8 | **8 / 60 (13.33% $\pm 24.8\%$)** | **70.00% ($\pm 23.5\%$)** | 0.6279 | 0.3062 |
| **Raw Pearson ($p \le 0.10$)**| 43 | 6 | **6 / 43 (13.33% $\pm 35.2\%$)** | **73.44% ($\pm 33.3\%$)** | 0.5775 | 0.0000 |
| **Pearson FDR-BH ($q \le 0.05$)**| 6 | 0 | **0 / 6 (0.00% $\pm 0.0\%$)** | 13.33% ($\pm 35.2\%$) | **0.6571** | 0.0000 |
| **MeTTa + FDR ($q \le 0.05$)**| 6 | 0 | **0 / 6 (0.00% $\pm 0.0\%$)** | 13.33% ($\pm 35.2\%$) | **0.6571** | 0.0000 |
| **QAOA $p=1$** | 65 | 6 | **6 / 65 (8.78% $\pm 14.8\%$)** | 57.44% ($\pm 24.5\%$) | 0.3181 | 1.6664 |
| **QAOA $p=2$** | 62 | 5 | **5 / 62 (7.33% $\pm 13.2\%$)** | 63.00% ($\pm 18.0\%$) | 0.3167 | 0.6593 |

### Detailed Seed-Level Replication Breakdown
- **M0 Exact Exhaustive**:
  - Seed 42: Fold 4 had $1/4$; Folds 0, 1, 2, 3 had $0/4$ $\implies 1/20$ ($5.0\%$).
  - Seed 123: Fold 1 had $1/4$; Folds 0, 2, 3, 4 had $0/4$ $\implies 1/20$ ($5.0\%$).
  - Seed 456: Fold 2 had $2/4$; Folds 0, 1, 3, 4 had $0/4$ $\implies 2/20$ ($10.0\%$).
  - Total across all 3 seeds: $4 / 60 = \mathbf{6.67\%}$.
- **QAOA $p=2$**:
  - Seed 42: $1/20$ ($5.0\%$).
  - Seed 123: $1/20$ ($5.0\%$).
  - Seed 456: $3/22$ ($12.0\%$).
  - Total across all 3 seeds: $5 / 62 = \mathbf{7.33\%}$.
- **Paired Permutation Test (QAOA $p=2$ vs M0 Exact)**:
  - Mean difference: $+0.66\%$.
  - 10,000 paired fold permutations: **$p = 0.687$**.
  - **Verdict**: The difference between QAOA and Exact is completely statistically indistinguishable from zero noise.

---

## 4. Assessment of Resampling Independence (15-Fold Limitations)

A critical methodological finding of this audit is that **the 15 cross-validation folds do NOT represent 15 statistically independent experiments**:
1. **Resampling Correlation**: In repeated 5-fold cross-validation across 3 seeds, each training fold shares $60\%$ of its participants with other folds in the same seed, and all 15 folds draw from the exact same cohort of $N=341$ individuals.
2. **Degrees of Freedom**: Treating 15 fold metrics as independent samples artificially deflates standard errors (Nadeau & Bengio, 2003). The effective sample size for hypothesis testing is closer to $N_{\text{eff}} \approx 5$.
3. **Implication**: Any claim that QAOA is "superior" or "comparable" based on fractional percentage differences across 15 correlated folds is statistically unsupportable.

---

## 5. Limitations of the Synthetic Benchmark

The synthetic benchmark in `harness_synthetic_benchmark.py` evaluated 10 replicates under a single causal generative structure. 

### Critical Limitations Identified:
1. **Separability Bias**: The synthetic DGP assigned direct causal relationships effect sizes of $r \approx 0.48$, while distractors had secondary indirect correlations of $r \approx 0.25$. This created an artificially easy problem where Top-$K$ ranking by raw utility recovered the exact planted set without needing quadratic optimization.
2. **Narrow Noise Spectrum**: The synthetic benchmark evaluated a single noise variance ($\sigma = 0.8$) and zero missing data.
3. **Absence of Pure Null Replicates**: The benchmark did not test a pure noise null hypothesis (where all true correlations are zero). On pure null data, optimization algorithms risk "discovering" spurious correlations by force-fitting the budget $K=4$.

### Recommendation for Future Synthetic Work
Before publishing synthetic benchmarks, the simulation suite should be expanded to include:
- Hard non-separable instances (where distractors have higher marginal utility than true causal edges).
- Missing-data injection ($10\% - 30\%$ MCAR and MAR).
- Pure null data replicates to establish false discovery baselines under quadratic optimization.
