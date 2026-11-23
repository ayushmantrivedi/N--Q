# NĀḌĪ-Q PPMI Longitudinal Evidence Discovery Benchmark Report
## Cross-Fold Biological Replication, Multiple-Testing FDR Control, and Modality Coverage on PPMI Cohort (N=341)

**Date**: November 23, 2026  
**Auditing Specialist**: Biomedical Data Scientist & Clinical Machine Learning Researcher  
**Artifacts**: `research/pre_implementation_benchmark/results/ppmi_discovery_results.csv`, `ppmi_discovery_summary.json`  

---

## 1. Clinical Cohort Protocol & Discovery Methods

The PPMI evidence discovery benchmark evaluated whether graph optimization or quantum algorithms improve the discovery of reproducible biological relationships from real longitudinal patient data.

### 1.1 Longitudinal Multimodal PPMI Cohort ($N=341$)
- **Data Source**: `F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet`, `curated_cut_cache.parquet`.
- **Primary Endpoint**: Task T2 (Month-12-to-Month-60 future motor progression $\ge 5.0$, 159 progressors / 46.6%).
- **Cross-Validation**: 3 Random Seeds ($42, 123, 456$) $\times$ 5-Fold Stratified K-Fold = 15 total folds, strictly partitioned by participant `PATNO`.
- **Candidate Feature Deltas**:
  1. Molecular: CSF $\alpha$-synuclein, CSF p-Tau181, CSF Amyloid-$\beta$.
  2. Striatal Imaging: Bilateral Caudate volume, Bilateral Putamen volume.
  3. Motor Progression: MDS-UPDRS Part III Total, Bradykinesia, Gait.
- **Candidate Pool ($N=10$)**:
  - $e_0$: `asyn` $\to$ `caudate` (molecular $\to$ structural)
  - $e_1$: `ptau` $\to$ `caudate` (molecular $\to$ structural)
  - $e_2$: `abeta` $\to$ `caudate` (molecular $\to$ structural)
  - $e_3$: `asyn` $\to$ `putamen` (molecular $\to$ structural)
  - $e_4$: `ptau` $\to$ `putamen` (molecular $\to$ structural)
  - $e_5$: `caudate` $\to$ `motor_total` (structural $\to$ motor)
  - $e_6$: `putamen` $\to$ `motor_total` (structural $\to$ motor)
  - $e_7$: `putamen` $\to$ `motor_brady` (structural $\to$ motor)
  - $e_8$: `caudate` $\to$ `motor_gait` (structural $\to$ motor)
  - $e_9$: `asyn` $\to$ `motor_total` (molecular $\to$ motor shortcut)

---

## 2. Experimental Benchmark Results (15 Folds)

Each protocol selected a set of candidate relationships strictly using training participants. The selected relationships were then evaluated on the **completely unobserved test fold participants**:
- **Held-Out Replication**: Relationship must achieve $p_{\text{test}} \le 0.10$ and have matching correlation sign ($\text{sign}(r_{\text{test}}) == \text{sign}(r_{\text{train}})$).
- **Direction Consistency**: Fraction of selected relationships with matching sign.
- **Multimodal Coverage**: Number of distinct modality transition types represented in the selected graph (e.g. molecular-to-structural, structural-to-motor).
- **Selection Stability**: Pairwise Jaccard similarity across the 15 folds ($\binom{15}{2} = 105$ pairs).

### Summary Results Table (15 Folds, 120 Total Runs)

| Discovery Protocol | Mean Edges Selected | Test Replication Rate | Direction Consistency | Modality Coverage | Selection Stability | Optimality Gap | Mean Runtime (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0: Exact Exhaustive QUBO** | 4.00 | 6.67% | 56.67% | **2.20** | 0.4656 | **0.0000** | 31.4 ms |
| **M3: Tabu Local Search** | 4.00 | 6.67% | 56.67% | **2.20** | 0.4249 | **0.0011** | 74.8 ms |
| **Top-K Heuristic** | 4.00 | 13.33% | 70.00% | 2.00 | 0.6279 | 0.3062 | 0.089 ms |
| **Raw Pearson ($p \le 0.10$)** | 2.87 | 13.33% | **73.44%** | 1.47 | 0.5775 | 0.0000 | **0.006 ms** |
| **Pearson FDR-BH ($q \le 0.05$)** | 0.40 | 0.00% | 13.33% | 0.20 | **0.6571** | 0.0000 | 0.045 ms |
| **MeTTa + FDR ($q \le 0.05$)** | 0.40 | 0.00% | 13.33% | 0.20 | **0.6571** | 0.0000 | 82.6 ms |
| **QAOA $p=1$** | 4.33 | 8.78% | 57.44% | 2.00 | 0.3181 | 1.6664 | 7,196.5 ms |
| **QAOA $p=2$** | 4.13 | 7.33% | 63.00% | **2.20** | 0.3167 | 0.6593 | 12,930.3 ms |

---

## 3. Key Scientific Findings

### 3.1 The Low Replication Reality in Longitudinal PPMI Biomarkers
- Across all methods, the biological replication rate on held-out test participants was **under 15%** (ranging from $0.0\%$ to $13.33\%$).
- **Statistical Root Cause**: 12-month longitudinal delta measurements for CSF $\alpha$-synuclein and MRI striatal volume have high measurement noise and biological heterogeneity. In sample sizes of $N \approx 272$ (training fold), many observed correlations hover between $|r| = 0.10$ and $0.18$ ($p \approx 0.01$ to $0.09$). When tested on the remaining $N \approx 68$ participants in the test fold, statistical power is low ($< 35\%$), causing most nominal correlations to fail replication ($p_{\text{test}} > 0.10$).

### 3.2 Impact of Multiple-Testing FDR Correction
- Standard uncorrected screening (`Raw_Pearson_p10`) selected an average of $2.87$ edges per fold.
- When applying rigorous **Benjamini-Hochberg False Discovery Rate (FDR)** control at $q \le 0.05$, the number of accepted relationships plummeted to **0.40 edges per fold**:
  - In 12 out of 15 folds, **zero (0) edges passed FDR**.
  - In only 3 folds did any relationships reach FDR significance (specifically `caudate -> motor_total` and `putamen -> motor_brady`).
- **Conclusion**: A significant fraction of candidate edges extracted by naive correlation screening are false positive associations driven by multiple testing. Integrating Benjamini-Hochberg FDR control into MeTTa's symbolic compiler is essential to prevent knowledge graph pollution.

### 3.3 Zero Evidence of Quantum Advantage in Biological Replication
- **Replication Rates**:
  - Exact Exhaustive: $6.67\%$ ($\pm 14.8\%$)
  - Tabu Search: $6.67\%$ ($\pm 14.8\%$)
  - QAOA $p=1$: $8.78\%$ ($\pm 14.8\%$)
  - QAOA $p=2$: $7.33\%$ ($\pm 13.2\%$)
- A paired two-sided permutation test comparing QAOA $p=2$ vs Exact yields $p = 0.687$ (statistically indistinguishable).
- **Selection Stability**: QAOA achieved the **lowest stability** ($0.3167$ vs $0.4656$ for Exact, $0.6279$ for Top-K). The stochastic nature of shallow quantum circuit measurements introduces spurious selection variance across cross-validation folds.

**Conclusion**: QAOA provides zero measurable improvement in biological relationship discovery, replication rate, or selection stability over classical solvers.
