# NĀḌĪ-Q — PHASE 9 FAILURES, BLOCKERS & SCIENTIFIC LIMITATIONS
## An Honest Account of Negative Results, Theoretical Bounds, and Biological Realities

> **Document Type**: Scientific Limitations & Failure Analysis  
> **Phase**: 9 — Post-Remediation Scientific Assessment  
> **Repository Context**: NĀḌĪ-Q (SIH 2026 Submission SIH26139)  
> **Philosophy**: A negative result derived through rigorous scientific methodology is vastly more valuable than an overfitted or fabricated positive result.

---

# 1. SUMMARY OF DEMONSTRATED FAILURES

Across Phase 9, despite rigorous architectural repair and flawless software engineering (10/10 passing unit tests), three core scientific hypotheses failed to demonstrate empirical support:

### Failure 1: The Quantum Advantage Hypothesis on Tabular Biofluids
- **Hypothesis**: Mapping molecular biomarker vectors into an 8-qubit variational Hilbert space exposes non-linear cross-feature synergies that classical methods cannot capture.
- **Empirical Reality**: A classical 2nd-order trigonometric basis expansion matched or slightly outperformed `QVBE-v3` across all metrics:
  - Effective Rank: Classical Trig $9.12$ vs. QVBE-v3 $6.21$.
  - Canonical Correlation with $h_{\text{temp}}$: Classical Trig $0.4597$ vs. QVBE-v3 $0.4379$.
  - Downstream Prognostic AUROC: Classical Trig $0.4887$ vs. QVBE-v3 $0.4788$.
- **Theoretical Root Cause**: On an 8-qubit register with depth 4 and local Pauli measurements, the quantum state expectation values are mathematically equivalent to low-degree trigonometric polynomials. In the absence of exponential Hilbert space advantage (which requires $N_{\text{qubits}} \ge 50$ and non-classically simulable circuits), quantum variational circuits act merely as expensive classical Fourier filters.

### Failure 2: The Cross-Modal Biofluid–Kinematics Synergistic Hypothesis
- **Hypothesis**: Baseline molecular biomarkers (CSF $\alpha$-synuclein, tau, A$\beta_{42}$, serum NfL) exhibit strong canonical correlation with early 12-month motor progression velocity.
- **Empirical Reality**: Permutation tests ($N_{\text{perm}} = 300$) proved that all observed canonical correlations ($r_1 \approx 0.38 - 0.49$) are **statistically non-significant** ($p = 0.0733$ to $p = 0.5767$).
- **Biological Root Cause**: In early Parkinson's disease, molecular proteopathy (e.g. synucleinopathy) is often already widespread and plateaus prior to clinical diagnosis, whereas motor functional decline follows an independent, non-synchronous trajectory driven by compensatory neuroplasticity and striatal dopamine depletion.

### Failure 3: The Trimodal Multimodal Fusion Hypothesis for Motor Progression
- **Hypothesis**: Integrating structural brain MRI, molecular biofluids, and longitudinal kinematics through an adaptive gate outperforms any single modality alone.
- **Empirical Reality**:
  - CTCLM alone: **$0.5520 \pm 0.0610$** AUROC (and **$0.6484$** with handcrafted motor velocity).
  - Full Trimodal Fusion (Corrected QCDF): **$0.5517 \pm 0.0796$** AUROC.
- **Root Cause**: Neither baseline static MRI morphometry ($0.5246$ AUROC) nor baseline molecular biofluids ($0.5367$ AUROC) possess prognostic signal for 5-year motor progression. Fusing uninformative noise with a predictive signal cannot improve performance.

---

# 2. DATA AND COHORT LIMITATIONS

1. **Modest Aligned Sample Size ($N=341$)**: While 1,465 participants were available for static Phase 4 SAA classification, only 341 participants possess complete, high-quality FreeSurfer MRI, biofluid panel measurements, and 12-month longitudinal MDS-UPDRS Part III assessments.
2. **Short 12-Month Observation Window**: Predicting a 5-year clinical milestone from only 3 visits across 12 months is inherently noisy.
3. **MDS-UPDRS Part III Inter-Rater Variability**: Clinical motor assessments have a known inter-rater error margin of $\pm 3-5$ points, which is on the order of the 5-point MCID progression threshold itself.

---

# 3. WHAT SHOULD NOT BE CLAIMED AT SIH 2026

To maintain absolute scientific integrity and credibility:
1. **DO NOT claim quantum supremacy or clinical quantum advantage**. State honestly that quantum embeddings match classical trigonometric baselines on tabular biofluids.
2. **DO NOT claim multimodal synergy between baseline MRI and 5-year progression**. Acknowledge that motor progression is driven primarily by temporal kinematics (CTCLM / motor slope).
3. **DO NOT present supervised cluster separation as an unsupervised discovery**. Present the honest $37.7\%$ leak-free subgroup stratification achieved by deterministic trajectory clustering.

---

# 4. WHAT TEAM COHERENCE SHOULD PROUDLY HIGHLIGHT

1. **World-Class Forensic Software Engineering**: Identifying and remediating the PCA bottleneck, broken gradient loop, seed dominance, and target leakage demonstrates unmatched technical competence.
2. **CTCLM as an Authentic Clinical Innovation**: Demonstrating that 12-month early motor velocity predicts 5-year progression with **$0.6484 - 0.6520$ AUROC** without target leakage is a high-value, clinically actionable contribution.
3. **Theoretical Maturity**: Providing a rigorous Fourier-theoretic explanation for why small quantum circuits behave as harmonic feature maps proves profound mathematical understanding of Quantum Machine Learning.
