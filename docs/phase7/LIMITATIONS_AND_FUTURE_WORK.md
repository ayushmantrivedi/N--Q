# NĀḌĪ-Q — Scientific Limitations, Hardware Realities & Future Roadmap
## Critical Appraisal of Quantum-Classical Integration, Cohort Constraints, and Next-Generation Milestones

**Project**: NĀḌĪ-Q (Smart India Hackathon 2026, Submission SIH26139)  
**Document**: Limitations, Hardware Boundaries & Future Work  
**Authors**: Lead Quantum ML Researcher, Biomedical AI Researcher & Senior Systems Architect  
**Date**: November 14, 2026  
**Status**: APPROVED SCIENTIFIC DOCUMENT  

---

## 1. Clinical, Cohort & Endpoint Limitations

### 1.1 Multimodal Cohort Scale and Selection Attrition
While the curated PPMI database spans 5,423 participants and 39,755 clinical visits, enforcing strict multimodal completeness across 5 years—requiring high-resolution 3D T1w structural MRI with validated FreeSurfer parcellations, baseline CSF Seed Amplification Assays, quantitative fluid biomarkers (NfL, GFAP, pTau), and continuous 5-year paired motor outcomes—restricted the primary longitudinal cohort to $N = 341$ participants ($N = 515$ in the pure clinical motor cohort). 
* **Implication**: Although $N=341$ represents one of the largest deeply phenotyped multimodal Parkinson's cohorts in the literature, statistical power for detecting subtle higher-order cross-modal interactions remains constrained.
* **Mitigation**: All findings are backed by 1,000 bootstrap resamples, permutation tests, and False Discovery Rate (FDR) corrections ($q < 0.05$).

### 1.2 Dopaminergic Symptomatic Medication Confounding
MDS-UPDRS Part III motor scores are intrinsically modulated by symptomatic dopaminergic therapy (levodopa, dopamine agonists, MAO-B inhibitors). 
* In PPMI, participants are examined in standardized conditions (OFF medication where feasible for de novo cohorts, or ON medication during longitudinal maintenance). 
* Initiation or escalation of dopaminergic therapy between Month 12 and Year 5 can mask underlying neurodegenerative progression, producing apparent motor stabilization or temporary improvement that reflects pharmacological response rather than disease arrest.

### 1.3 Scope of the SAA Molecular Biomarker
The $\alpha$-synuclein Seed Amplification Assay (SAA) is an ultra-sensitive qualitative and semi-quantitative measure of protein misfolding and aggregation in CSF.
* SAA positivity confirms the presence of pathogenic synuclein seeds with $>95\%$ sensitivity in sporadic PD.
* However, SAA positivity alone does not linearly correlate with clinical disease severity or the rate of motor decline. As demonstrated in our Question 2 and Question 4 analyses, SAA-positive patients exhibit widely divergent 5-year progression rates ($28.4\%$ vs $73.8\%$), governed primarily by secondary neuroaxonal damage (NfL) and tau/amyloid co-pathology.

---

## 2. Quantum Computing & Physical Hardware Realities

### 2.1 Statevector Simulation vs. Noisy Intermediate-Scale Quantum (NISQ) Execution
The empirical results reported in Phases 4–7 were generated using PennyLane's statevector simulation (`default.qubit`), representing an idealized, noise-free Hilbert space.
* **Coherence Time Limits**: On existing physical QPUs (e.g., superconducting transmon qubits or trapped-ion systems), qubit coherence times ($T_1$ relaxation, $T_2$ dephasing) are typically on the order of $100 - 300\ \mu\text{s}$.
* **Gate Fidelity**: Two-qubit CNOT gate error rates on commercial QPUs range between $0.5\%$ and $1.5\%$. In an 8-qubit, $L=2$ cyclic architecture ($16$ CNOT gates), cumulative circuit fidelity without error mitigation drops to $\sim 85 - 90\%$.
* **Finite Shot Sampling Variance**: Physical quantum hardware evaluates expectation values via finite projective measurement counts ($S$ shots). Phase 5 established that at $S = 1,024$ shots, Poisson measurement noise degrades standalone AUROC by nearly $0.08$ points ($0.8090 \to 0.7217$). Maintaining high-fidelity observable extraction requires shot budgets of $S \ge 8,192$, demanding optimized shot allocation protocols in clinical production.

### 2.2 Barren Plateaus and Dimension Scaling
In variational quantum algorithms, deep random parameterizations suffer from exponential concentration of measure (barren plateaus), where gradient variances scale as $\mathcal{O}(2^{-n})$.
* NĀḌĪ-Q mitigates this by maintaining a shallow depth ($L=2$), local Hamiltonian observables ($\langle Z_i \rangle, \langle X_i \rangle$), and bounded angle initializations.
* However, scaling beyond 8–16 qubits to ingest hundreds of raw features simultaneously without classical dimensionality reduction remains intractable without symmetry-preserving, physically structured quantum circuits.

---

## 3. Methodological Boundaries & Negative Findings

To maintain absolute scientific transparency, NĀḌĪ-Q explicitly documents the following negative findings:
1. **No Standalone Quantum Supremacy on Flat Tabular Metrics**: Variational quantum circuits do not out-classify classical Gradient-Boosted Decision Trees (LightGBM) on tabular biomarker metrics. Classical LightGBM achieved $0.8615$ AUROC, whereas Standalone 8-qubit QVBE achieved $0.8220$. The value of QVBE is as a complementary, non-linear geometric representation engine, not a replacement for classical ensembles.
2. **Failure of Naive Concatenation**: Concatenating raw static features, quantum observables, and temporal embeddings degraded performance ($0.5794$ AUROC) relative to temporal modeling alone ($0.6417$ AUROC). Gated fusion (QCDF) with availability masking is strictly necessary to prevent uninformative modalities from acting as noise sources.
3. **Cross-Sectional Structural MRI Insufficiency**: Baseline cross-sectional FreeSurfer morphometry alone achieved an AUROC of $0.4818$ (equivalent to random guessing) for predicting 5-year functional motor decline. Structural brain changes lag years behind molecular synucleinopathy and dynamic motor velocity in early disease stages.

---

## 4. Next-Generation Roadmap & Future Directions

### 4.1 Phase 8: Physical Hardware Deployment & Error Mitigation
* Implement Zero-Noise Extrapolation (ZNE) and Clifford Readout Error Mitigation (TREX) on physical QPUs (e.g. IBM Quantum Heron / Eagle architectures).
* Deploy grouped Pauli measurement scheduling via qubit-wise commutativity (QWC) to reduce the required circuit execution passes for the 23 observables from 3 to 2.

### 4.2 Molecular & Imaging Modality Expansion
* **DaTscan SPECT Integration**: Incorporate quantitative Dopamine Transporter SPECT imaging (striatal binding ratios in anterior caudate and posterior putamen) into Pillar 1, providing direct visualization of presynaptic nigrostriatal denervation.
* **Plasma pTau217 & Peripheral SAA**: Expand Pillar 2 to evaluate blood-based pTau217 (high diagnostic specificity for Alzheimer's co-pathology) and peripheral tissue seeding assays (skin biopsy or olfactory mucosa RT-QuIC).
* **Longitudinal MRI Morphometric Velocity**: Upgrade Pillar 1 from cross-sectional baseline scans to longitudinal FreeSurfer percent-change maps ($\Delta \text{cortical thickness}$ across Months 0–12), capturing early neurostructural atrophy rates.

### 4.3 Multi-Center Prospective Validation
* Validate NĀḌĪ-Q representations on external independent cohorts:
  - Parkinson's Disease Biomarkers Program (PDBP, $N \approx 1,500$)
  - BioFinder Study (Lund University, Sweden)
  - Alzheimer's Disease Neuroimaging Initiative (ADNI, for cross-disease synuclein/amyloid specificity).

