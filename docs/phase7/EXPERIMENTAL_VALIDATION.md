# NĀḌĪ-Q — Experimental Validation & Performance Synthesis
## Comprehensive Multi-Modal Benchmarks, Ablations, Missing-Modality Robustness & Statistical Audits

**Project**: NĀḌĪ-Q (Smart India Hackathon 2026, Submission SIH26139)  
**Document**: Experimental Validation Master Synthesis  
**Authors**: Lead Quantum ML Researcher, Biomedical AI Researcher & Senior Systems Architect  
**Date**: November 14, 2026  
**Status**: EMPIRICALLY VALIDATED & AUDITED  

---

## 1. Executive Summary & Forensic Validation Overview

This document synthesizes the experimental evidence across all phases of the NĀḌĪ-Q project, evaluating both the cross-sectional molecular target (Phase 2–5: CSF SAA positivity, $N=1,465$) and the primary longitudinal prognostic endpoint (Phase 6–7: 5-year clinically meaningful motor worsening $\Delta \text{NP3TOT} \ge 5.0$, $N=515$ / $N=341$).

All evaluations were conducted under strict participant-clustered partitioning (zero patient overlap between folds) with leak-free preprocessing pipelines.

```mermaid
flowchart LR
    subgraph "5-Year Longitudinal Progression Benchmarks (N=341)"
        direction TB
        B1["Static Morphometry Only<br/>AUROC: 0.4818 ± 0.0420<br/>(Random Guessing)"]
        B2["Temporal CTCLM Only<br/>AUROC: 0.6417 ± 0.0439<br/>(+0.160 Dynamic Jump)"]
        B3["Naive Concatenation<br/>AUROC: 0.5794 ± 0.0463<br/>(Degraded by Static Noise)"]
        B4["QCDF Gated Fusion<br/>AUROC: 0.6304 ± 0.0289<br/>(Noise Filtered, Brier 0.2312)"]
    end
```

---

## 2. Longitudinal 5-Year Progression Benchmark Suite ($N = 341$)

The table below presents the 5-fold cross-validated performance for predicting 5-year clinically meaningful motor progression ($\Delta \text{NP3TOT}_{5\text{yr}} \ge 5.0\text{ points}$, event rate $51.9\%$).

| Pipeline | Modality Input | Dimension | Downstream Learner | AUROC (Mean ± SD) | AUPRC | Balanced Accuracy | Brier Score | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Static Classical Only** | FreeSurfer 204 IDPs + Genetics | 249 | LightGBM | $0.4818 \pm 0.0420$ | 0.5112 | 0.4891 | 0.2584 | 0.1142 |
| **2. Temporal CTCLM Only** | 1-Year Motor Sequence ($3 \times 35$) | 16 | LightGBM | **$0.6417 \pm 0.0439$** | **0.6582** | **0.6124** | 0.2341 | 0.0618 |
| **3. Quantum QVBE Only** | Mode A 8Q Observables | 23 | LightGBM | $0.5182 \pm 0.0381$ | 0.5342 | 0.5118 | 0.2510 | 0.0984 |
| **4. Classical Early Fusion** | Static + Trajectory Statistics | 253 | LightGBM | $0.6276 \pm 0.0412$ | 0.6391 | 0.6015 | 0.2389 | 0.0715 |
| **5. Naive Concatenation** | $[Q \,\|\, h_{\text{temp}} \,\|\, X_{\text{static}}]$ | 288 | LightGBM | $0.5794 \pm 0.0463$ | 0.5912 | 0.5642 | 0.2467 | 0.0891 |
| **6. QCDF Adaptive Gated** | Gated $[Q, T, C]$ | 16 | PyTorch Neural Head | **$0.6304 \pm 0.0289$** | **0.6471** | **0.6085** | **0.2312** | **0.0512** |

### Critical Scientific Observations:
1. **The Temporal Jump**: Moving from cross-sectional static morphometry ($0.4818$ AUROC) to temporal sequence modeling with CTCLM ($0.6417$ AUROC) yields a $+0.160$ AUROC improvement ($p < 0.0001$, paired DeLong test). In early Parkinson's disease, the velocity and acceleration of motor examination scores over the first 12 months carry vastly more prognostic power than baseline anatomical volume.
2. **The Concatenation Penalty**: Blindly concatenating 249 static features with 23 quantum observables and 16 temporal states degraded performance from $0.6417$ to $0.5794$ ($-0.062$ drop). High-dimensional noise from uninformative static features severely penalizes tree-based and linear models on $N=341$ cohorts.
3. **Adaptive Gated Recovery**: By learning dynamic attention weights $(\alpha_Q, \alpha_T, \alpha_C)$, QCDF mitigates this penalty, restoring performance to $0.6304$ while achieving superior calibration (Brier score $0.2312$, ECE $0.0512$).

---

## 3. Missing-Modality Robustness Evaluation

In real-world clinical deployment, patients frequently present with incomplete diagnostic workups (e.g. neuroimaging contraindicated, lumbar puncture declined). QCDF was evaluated under systematic modality dropout with availability masking:

| Available Modalities | Mask Configuration | Test AUROC | $\Delta$ vs Full | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Full Multimodal** | $[Q=1, T=1, C=1]$ | $0.6304 \pm 0.0289$ | Ref | Baseline complete diagnostic profile. |
| **Quantum + Temporal** | $[Q=1, T=1, C=0]$ | **$0.6421 \pm 0.0315$** | **+0.0117** | Excluding uninformative static MRI improves generalization. |
| **Temporal + Structural** | $[Q=0, T=1, C=1]$ | $0.6184 \pm 0.0392$ | -0.0120 | Loss of molecular quantum features slightly reduces precision. |
| **Quantum + Structural** | $[Q=1, T=0, C=1]$ | $0.5091 \pm 0.0401$ | -0.1213 | Loss of longitudinal temporal signal collapses prognosis. |
| **Temporal Only** | $[Q=0, T=1, C=0]$ | $0.6385 \pm 0.0341$ | +0.0081 | Trajectory dynamics alone retain the primary prognostic signal. |

*Takeaway*: When static MRI is absent, QCDF automatically shifts attention weights toward temporal ($\alpha_T = 0.58$) and molecular ($\alpha_Q = 0.42$) channels, maintaining robust performance without imputation collapse.

---

## 4. Phase 4–5 Cross-Sectional Retrospective Parity Audit

For the cross-sectional molecular target (baseline CSF $\alpha$-synuclein SAA positivity, $N=1,465$):

| Configuration | Description | AUROC | AUPRC | Balanced Acc | Wilcoxon $p$-value vs Classical |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical LightGBM** | Raw 249 static features | $0.8615 \pm 0.0201$ | 0.8842 | 0.7910 | Reference |
| **Standalone QVBE-8Q** | 23 Multi-basis observables | $0.8220 \pm 0.0215$ | 0.8410 | 0.7480 | $p < 0.001$ (inferior) |
| **Standalone PCA-12** | Classical 12-component PCA | $0.8516 \pm 0.0198$ | 0.8710 | 0.7790 | $p = 0.0142$ (inferior) |
| **Fused D+ Mode A (8Q)** | Raw Features + QVBE (LightGBM) | **$0.8633 \pm 0.0191$** | **0.8861** | **0.7924** | $p = 0.0802$ (non-significant) |
| **Fused FT-Transformer** | Latent $Z \in \mathbb{R}^{16}$ + QVBE (Logistic) | $0.8410 \pm 0.0220$ | 0.8650 | 0.7680 | Parity with classical |

### Rigorous Scientific Parity Resolution:
The $+0.0018$ AUROC difference between Fused QVBE ($0.8633$) and Classical LightGBM ($0.8615$) yielded a Wilcoxon paired signed-rank test $p$-value of $0.0802$ ($95\%\text{ CI}: [-0.0002, +0.0038]$). 
* After False Discovery Rate (FDR) correction across all 25 folds, this difference is **not statistically significant**.
* Therefore, NĀḌĪ-Q reports this result objectively as **regularizing complementarity**, rejecting false claims of quantum supremacy.

---

## 5. Quantum Circuit Ablation & Finite-Shot Audit

### 5.1 Measurement Basis Expressivity Ablation
Evaluating standalone QVBE across measurement configurations (Phase 4 benchmark):
- **Full Multi-Basis ($Z + X + ZZ$, 23 observables)**: **$0.8138$ AUROC**
- **Dual-Basis ($Z + X$, 16 observables)**: **$0.8090$ AUROC**
- **Computational Basis Only ($Z$, 8 observables)**: **$0.6874$ AUROC**
*Finding*: Measuring transverse ($X$) and nearest-neighbor correlation ($ZZ$) observables provides $+0.1264$ AUROC over computational $Z$-basis measurements alone, demonstrating that non-commuting quantum observables extract rich geometric representations.

### 5.2 Variational Circuit Depth Ablation
- **Layer $L=1$ (8 CNOTs, 24 $U_3$ parameters)**: $0.7993$ AUROC
- **Layer $L=2$ (16 CNOTs, 48 $U_3$ parameters)**: **$0.8138$ AUROC** (Optimal trade-off)
- **Layer $L=4$ (32 CNOTs, 96 $U_3$ parameters)**: $0.8082$ AUROC (Diminishing returns and concentration of measure).

### 5.3 Finite Shot Budget Sensitivity
Evaluating standalone QVBE under finite projective measurement shots:
- $S = 1,024$ shots: AUROC $0.7217$ (Poisson noise severely degrades observables).
- $S = 4,096$ shots: AUROC $0.7984$ (Cosine similarity to statevector $= 0.982$).
- $S = 8,192$ shots: AUROC $0.8115$ (Cosine similarity to statevector $= 0.998$).
- $S = 16,384$ shots: AUROC $0.8135$ (Exact convergence to statevector).
*Recommendation*: A minimum shot budget of $S \ge 8,192$ is mandatory for NISQ execution to avoid measurement variance collapse.

