# NĀḌĪ-Q — Scientific Methodology & Statistical Validation Protocol
## Experimental Design, Modality Decoupling, Classical Benchmarking & Discovery Framework

**Project**: NĀḌĪ-Q (Smart India Hackathon 2026, Submission SIH26139)  
**Methodology Document**: Authoritative Research Protocol  
**Authors**: Lead Quantum ML Researcher, Biomedical AI Researcher & Senior Systems Architect  
**Date**: November 14, 2026  
**Status**: APPROVED PROTOCOL  

---

## 1. Ethical Governance & Core Research Principles

The NĀḌĪ-Q research pipeline is governed by five foundational principles:
1. **Zero Data Fabrication**: All analyses are conducted exclusively on authenticated clinical, neuroimaging, and biospecimen records from the Parkinson's Progression Markers Initiative (PPMI). No synthetic noise injections, simulated clinical scores, or hallucinated trajectories are permitted.
2. **Strict Longitudinal Integrity**: Models predicting 5-year progression utilize clinical and biomarker data collected strictly at or prior to Visit 04 (Month 12). Future outcomes from Visit 12 (Month 60) are accessed solely as ground-truth evaluation labels.
3. **Participant-Clustered Partitioning**: Data splits are strictly partitioned by unique participant identifiers (`PATNO`). All records for a given patient are assigned exclusively to either the training set or the test set:
   $$\text{Train} \cap \text{Test} = \emptyset$$
4. **Data Preprocessing Leakage Prevention**: All imputation parameters (medians, means) and feature scaling statistics (means, standard deviations, quantile ranges) are computed strictly on the training partition and applied out-of-fold to test partitions.
5. **Multiple Testing Correction**: Every statistical hypothesis test across canonical correlations, partial correlations, and subgroup comparisons is subjected to Benjamini-Hochberg False Discovery Rate (FDR) correction at an authoritative threshold of $q < 0.05$.

---

## 2. Longitudinal Cohort Construction & Endpoint Definitions

### 2.1 The Multimodal Benchmark Cohort ($N = 341$)
From the broader PPMI dataset ($N=5,423$), the multimodal longitudinal progression cohort comprises $N=341$ verified patients who satisfy:
1. Baseline 3D T1w structural MRI with validated FreeSurfer 7.1 parcellation (204 regional IDPs).
2. Documented baseline CSF $\alpha$-synuclein Seed Amplification Assay (SAA) qualitative status and kinetics.
3. Quantitative baseline fluid biomarkers: Plasma/CSF Neurofilament Light Chain (NfL), Glial Fibrillary Acidic Protein (GFAP), phosphorylated tau ($\text{pTau}_{181}$), total tau ($\text{tTau}$), and amyloid-beta ($\text{A}\beta_{42}$).
4. Longitudinal MDS-UPDRS Part III motor scores across baseline ($t=0$), Month 6 ($t=0.5\text{ yr}$), Month 12 ($t=1.0\text{ yr}$), and Month 60 ($t=5.0\text{ yr}$).

### 2.2 Primary Prognostic Endpoint: 5-Year MCID Motor Progression
In accordance with Movement Disorder Society (MDS) clinical consensus guidelines, the primary longitudinal target is defined as:
$$Y = \mathbb{I}\left( \text{NP3TOT}(V12) - \text{NP3TOT}(BL) \ge 5.0 \right)$$
where $\text{NP3TOT}$ denotes the MDS-UPDRS Part III total motor score, and $5.0\text{ points}$ represents the established Minimal Clinically Important Difference (MCID).
* Event Rate: $177 / 341$ participants ($51.9\%$) experienced clinically meaningful motor worsening at 5 years.
* Continuous Velocity Endpoint:
  $$\Delta \text{NP3TOT}_{5\text{yr}} = \text{NP3TOT}(V12) - \text{NP3TOT}(BL)$$

---

## 3. Modality Decoupling & Representation Formulation

To eliminate target-construction overlap and ensure architectural modularity, the input feature space is cleanly decoupled into three orthogonal branches:

```
Total Modality Input Space
├── Structural MRI Branch: X_mri in R^204 (FreeSurfer 7.1 Morphometry)
├── Molecular Biofluid Branch: X_mol in R^20 (Fluid Assays + SAA Kinetics + Genetics)
└── Functional Temporal Branch: X_temp in R^(3 x 35) (Months 0, 6, 12 Motor Sequences)
```

### 3.1 Structural MRI Branch ($X_{\text{mri}} \in \mathbb{R}^{204}$)
* 68 regional cortical thicknesses (`cth_*`)
* 40 subcortical segmentations (`aseg_*`)
* 68 regional cortical surface areas (`sa_*`)
* 28 regional white matter hypointensity volumes (`wm_*`)

### 3.2 Molecular Biofluid & Genetic Branch ($X_{\text{mol}} \in \mathbb{R}^{20}$)
* Fluid Assays: `PLASMA_NFL`, `PLASMA_GFAP`, `CSF_PTAU181`, `CSF_TTAU`, `CSF_ABETA42`, `CSF_ALPHASYN`, `CSF_NFL`, `CSF_GFAP`
* SAA Kinetics: `SAA_positive`, `SAA_Fmax_24h`, `SAA_SlopeMax_24h`, `SAA_AUC_24h`, `SAA_TTT_24h`
* Genetics & PRS: `is_LRRK2_carrier`, `is_GBA_carrier`, `is_APOE_e4_carrier`, `PATHVAR_COUNT`, `GP2_PGS`, `META5_PGS`, `META5_excl_LRRK2_GBA_PGS`

### 3.3 Functional Temporal Branch ($X_{\text{temp}} \in \mathbb{R}^{3 \times 35}$)
* Visit 0 (Baseline, $t=0.0\text{ yr}$)
* Visit 02 (Month 6, $t=0.5\text{ yr}$)
* Visit 04 (Month 12, $t=1.0\text{ yr}$)
* Each visit contains the total motor score, 4 subscores (tremor, rigidity, bradykinesia, axial/gait), and 30 individual motor examination items.

---

## 4. Quantum Variational Biomarker Embedding (QVBE-v2)

### 4.1 Circuit Topology & Parameterization
QVBE-v2 operates on an 8-qubit register ($\mathcal{H} = (\mathbb{C}^2)^{\otimes 8}$, $\dim \mathcal{H} = 256$):
1. **Angle Encoding**: Input features $\mathbf{x}$ are projected into qubit rotation angles $\mathbf{\theta}_0 = \pi \cdot \text{MinMax}(\mathbf{x}) \in [0, \pi]^8$ and applied via single-qubit $R_Y(\theta_{0, i})$ gates:
   $$|\psi_0\rangle = \bigotimes_{i=1}^8 R_Y(\theta_{0, i}) |0\rangle^{\otimes 8}$$
2. **Parameterized $U_3$ Rotations**: Across $L=2$ variational layers, each qubit undergoes a 3-parameter single-qubit rotation:
   $$U_3(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\ e^{i\phi} \sin(\theta/2) & e^{i(\phi+\lambda)} \cos(\theta/2) \end{pmatrix}$$
3. **Cyclic Entanglement**: Nearest-neighbor CNOT gates create cyclic entanglement across the register:
   $$\text{CNOT}_{i, (i \bmod 8) + 1} \quad \text{for } i \in \{1, \dots, 8\}$$

### 4.2 Multi-Basis Pauli Observable Extraction
To maximize informational extraction from the quantum state $|\psi\rangle$, QVBE-v2 extracts $M=23$ expectation values across multiple non-commuting bases:
1. **Computational Basis (Z-basis)**: Single-qubit longitudinal Pauli-Z projections:
   $$\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \quad (i = 1, \dots, 8)$$
2. **Transverse Basis (X-basis)**: Single-qubit transverse Pauli-X projections:
   $$\langle X_i \rangle = \langle \psi | X_i | \psi \rangle \quad (i = 1, \dots, 8)$$
3. **Two-Qubit Correlation Basis (ZZ-basis)**: Nearest-neighbor two-qubit correlation projections:
   $$\langle Z_i Z_{i+1} \rangle = \langle \psi | Z_i \otimes Z_{i+1} | \psi \rangle \quad (i = 1, \dots, 7)$$
Total observable representation: $Q = [\langle Z_i \rangle, \langle X_i \rangle, \langle Z_i Z_{i+1} \rangle] \in \mathbb{R}^{23}$.

---

## 5. Matched Classical Controls Protocol

To prevent unsubstantiated claims of quantum superiority, every quantum representation ($Q_{\text{mol}}, Q_{\text{mri}}$) is systematically benchmarked against four matched classical controls:
1. **Tier 1 (Raw Input Control)**: The normalized input feature matrices ($X_{\text{mol}} \in \mathbb{R}^{20}$, $X_{\text{mri}} \in \mathbb{R}^{204}$).
2. **Tier 2 (Linear PCA Control)**: Dimensionality reduction via Principal Component Analysis matched to $k=8$ latent components (matching the 8-qubit register size).
3. **Tier 3 (Non-Linear Kernel PCA Control)**: Classical non-linear manifold projection via Radial Basis Function (RBF) Kernel PCA matched to $k=8$ components ($\gamma = 1/d$).
4. **Tier 4 (Random Gaussian Projection Control)**: Johnson-Lindenstrauss random orthogonal projection ($d \to 8$), controlling for arbitrary dimension compression effects.

---

## 6. Statistical Significance & Discovery Verification Suite

### 6.1 Canonical Correlation Analysis (CCA)
Cross-modal linear and non-linear alignment is evaluated between modality pairs $(A, B)$ by finding projection vectors $\mathbf{a}, \mathbf{b}$ that maximize the Pearson correlation:
$$r_1 = \max_{\mathbf{a}, \mathbf{b}} \text{corr}(\mathbf{a}^T A, \mathbf{b}^T B)$$
Subsequent orthogonal canonical vectors $(r_2, r_3)$ capture residual shared variance.

### 6.2 Clustered Bootstrap Confidence Intervals
To assess sample variability without parametric assumptions, we compute $B = 1,000$ non-parametric bootstrap resamples with replacement. Empirical $95\%$ confidence intervals are determined via the 2.5th and 97.5th percentiles:
$$\text{CI}_{95\%} = \left[ q_{0.025}(\{r_1^{(b)}\}_{b=1}^B), \, q_{0.975}(\{r_1^{(b)}\}_{b=1}^B) \right]$$

### 6.3 Permutation Null Hypothesis Testing
To test whether observed cross-modal alignments exceed chance, participant identifiers for modality $B$ are randomly permuted $P = 1,000$ times while holding modality $A$ fixed:
$$p_{\text{perm}} = \frac{1}{P} \sum_{p=1}^P \mathbb{I}\left( r_1^{(\pi_p)} \ge r_1^{\text{observed}} \right)$$

### 6.4 Partial Correlation Analysis
To verify whether biological signatures predict functional worsening independently of baseline severity, partial Pearson correlations are computed controlling for baseline motor score ($\text{NP3TOT}_{\text{BL}}$):
$$r_{x, y \cdot z} = \frac{r_{xy} - r_{xz} r_{yz}}{\sqrt{(1 - r_{xz}^2)(1 - r_{yz}^2)}}$$
All $p$-values are adjusted using the Benjamini-Hochberg procedure.

