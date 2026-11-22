# NĀḌĪ-Q — QVBE-v3 Learning Objective Specification and Mathematics

> **Document Type**: Formal Objective Specification  
> **Phase**: 9 — QVBE-v2 Remediation → QVBE-v3 Development  
> **Created**: 2026-11-18  
> **Status**: Authoritative specification — governs QVBE-v3 training objective implementation  
> **Evidentiary Standards**: `[VERIFIED FROM SOURCE CODE]` | `[VERIFIED FROM EXPERIMENT OUTPUT]` | `[MATHEMATICAL DERIVATION]` | `[REASONABLE INFERENCE]` | `[UNTESTED HYPOTHESIS]`

---

## 0. Context and Motivation

QVBE-v2 was shown by Phase 8 deep diagnostics to have zero variational optimisation in Mode A: parameters are fixed after a single Ridge classifier warm-start and are never updated via gradient descent on a principled self-supervised objective. `[VERIFIED FROM EXPERIMENT OUTPUT — test6_seed_to_patient_var_ratio = 1.541]`

The result is that parameter initialisation variance (0.0152) **exceeds** patient biological variance (0.0098), making the quantum embedding unreliable as a biomarker representation. `[VERIFIED FROM EXPERIMENT OUTPUT]`

QVBE-v3 requires a principled self-supervised learning objective that:

1. Can be optimised by gradient descent (differentiable end-to-end).  
2. Encourages the quantum representation to encode cross-modal biological relationships.  
3. Is honest about the limitations imposed by the available dataset size (N = 341 total; N_pair ≈ 100–150 participants with both molecular biofluid and temporal motor data).

---

## 1. Notation and Problem Statement

| Symbol | Definition |
|--------|-----------|
| $N$ | Total participants, N = 341 `[VERIFIED FROM EXPERIMENT OUTPUT]` |
| $N_\text{pair}$ | Participants with both molecular biofluid AND temporal motor data, $N_\text{pair} \approx 100$–$150$ `[REASONABLE INFERENCE from PPMI cohort structure]` |
| $x \in \mathbb{R}^d$ | Input feature vector (molecular biofluid or multi-omic tabular features) |
| $q_\theta(x) \in \mathbb{R}^{23}$ | QVBE-v2 output (23 Pauli expectation values); $\mathbb{R}^{31}$ for QVBE-v3 (31 observables) |
| $h_\text{temp} \in \mathbb{R}^H$ | Temporal motor representation from CTCLM-discovery (self-supervised, label-free) |
| $\tau$ | InfoNCE temperature hyperparameter |
| $\lambda$ | Reconstruction regularisation weight |
| $B$ | Training mini-batch size |

**Research objective**: Discover latent cross-modal relationships between molecular biofluid data and temporal motor sequences in Parkinson's disease participants, without using diagnostic labels. `[MATHEMATICAL DERIVATION]`

---

## 2. Candidate A — Masked Feature Reconstruction

### 2.1 Mathematical Formulation

Let $x \in \mathbb{R}^d$ be the input feature vector after min-max rescaling to $[0, 1]^d$.

**Masking strategy**: A binary mask $m \in \{0, 1\}^d$ is sampled independently per training step, with each element:

$$m_i \sim \text{Bernoulli}(0.70), \quad i = 1, \ldots, d$$

so that 30% of features are set to zero in expectation. `[REASONABLE INFERENCE — consistent with MAE masking ratios (He et al. 2022)]`

The masked input is:

$$x_\text{masked} = m \odot x$$

where $\odot$ denotes element-wise multiplication.

**Encoder → Quantum Circuit Pipeline**:

$$x \xrightarrow{\text{Trainable Adapter}} z = Wx \in \mathbb{R}^8 \xrightarrow{\text{Angle Encode + HEA}} \rho(\theta, z) \xrightarrow{\text{Pauli Measure}} q_\theta(x) \in \mathbb{R}^{31}$$

Steps:
1. **Trainable Input Adapter**: Linear projection $W \in \mathbb{R}^{8 \times d}$, trained jointly with quantum parameters $\theta$. Replaces fixed PCA from QVBE-v2.  `[MATHEMATICAL DERIVATION]`
2. **Data Re-uploading Encoding** (Pérez-Salinas et al. 2020): Qubit $i$ receives encoding angle $\phi_i^{(l)} = z_i$ at each of $L=4$ circuit layers, interleaved with variational layers.  
3. **Hardware-Efficient Ansatz (HEA)**: Depth-4 alternating $R_y$–$R_z$ rotation layers with cyclic CNOT entanglement.  
4. **Observable measurement**: Produces $q_\theta(x) \in \mathbb{R}^{31}$ (see Deliverable 2 / Measurement Design document).

**Decoder**: A lightweight MLP decoder $g_\phi: \mathbb{R}^{31} \to \mathbb{R}^d$, two hidden layers (64, 128 units), ReLU activations:

$$\hat{x} = g_\phi(q_\theta(x_\text{masked}))$$

**Reconstruction Loss**:

$$\mathcal{L}_\text{reconstruction} = \frac{1}{d} \left\| x - g_\phi\!\left(q_\theta(x_\text{masked})\right) \right\|_2^2$$

Note: the target is the **unmasked** original $x$; the input to the encoder is the masked $x_\text{masked}$. The representation $q_\theta(x)$ must therefore preserve enough information to reconstruct the missing 30%.

### 2.2 Information Preservation Requirement

The quantum representation $q_\theta(x)$ must preserve sufficient information from the full input $x$ to allow linear or near-linear reconstruction of the masked features. Because the encoder sees the masked $x_\text{masked}$ but must predict the original $x$, it is implicitly pressured to capture cross-feature correlations. `[MATHEMATICAL DERIVATION]`

### 2.3 Risk Analysis — Does Reconstruction Guarantee Biologically Useful Representations?

**No.** `[MATHEMATICAL DERIVATION]`

Reconstruction loss is minimised by any representation that captures maximum variance in $x$. PCA is a linear method that **also** minimises mean squared reconstruction error:

$$\text{PCA}: \hat{x} = \mu + U_k U_k^\top (x - \mu), \quad \text{where} \ U_k \in \mathbb{R}^{d \times k} \ \text{are top-}k \ \text{eigenvectors of Cov}(X)$$

PCA achieves $\mathcal{L}_\text{reconstruction} = \sum_{i=k+1}^d \lambda_i$ (sum of discarded eigenvalues), which is a provably optimal reconstruction. A quantum encoder that merely learns the same subspace as PCA provides no advantage.

**Key limitation**: QVBE-v2 Phase 8 diagnostics showed that PCA on molecular biofluid data already explains 99.99% of variance with 8 components. `[VERIFIED FROM EXPERIMENT OUTPUT — test1_pca_bio_explained_variance_8 = 0.9999]` A reconstruction objective would almost certainly converge to this same PCA subspace.

**What additional constraint could distinguish quantum from classical?**  
Adding a cross-modal alignment term (see Candidate B) forces the representation to encode information that is *not* accessible to univariate PCA — namely, the shared latent structure with temporal motor sequences. This is the distinguishing motivation for the combined loss in §4.

---

## 3. Candidate B — Cross-Modal InfoNCE Alignment

### 3.1 Mathematical Formulation

**Positive pair definition**: A positive pair $(x_i, s_i)$ is formed when participant $i$ has **both**:
- Molecular biofluid feature vector $x_i \in \mathbb{R}^d$ (proteomics / metabolomics / CSF biomarkers), AND  
- Temporal motor sequence $s_i$ represented as hidden state $h_\text{temp,i} = \text{CTCLM-discovery}(s_i) \in \mathbb{R}^H$

Pairs from the same participant are positive; pairs from different participants (within the batch) are negative.

**Critical dependency**: $h_\text{temp}$ **must** originate from CTCLM operating in discovery (self-supervised) mode — with NO diagnostic labels. `[VERIFIED FROM SOURCE CODE — CTCLMConfig supports label-free temporal encoding]` Using supervised CTCLM outputs would constitute label leakage into the self-supervised pre-training stage.

**Negative sampling**: In-batch negatives. For a batch of size $B$, each molecular embedding $q_{\theta,i}$ is contrasted against $B - 1$ negative temporal embeddings $h_{\text{temp},j}$, $j \neq i$.

**Similarity function**: Cosine similarity after L2 normalisation:

$$\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \cdot \|v\|_2}$$

Both $q_{\theta,i}$ and $h_{\text{temp},i}$ are L2-normalised before computing $\text{sim}(\cdot, \cdot)$.

**InfoNCE Loss** (Oord et al. 2018):

$$\mathcal{L}_\text{InfoNCE} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\!\left(\text{sim}(q_{\theta,i},\, h_{\text{temp},i}) \,/\, \tau\right)}{\displaystyle\sum_{j=1}^{B} \exp\!\left(\text{sim}(q_{\theta,i},\, h_{\text{temp},j}) \,/\, \tau\right)}$$

This objective is a lower bound on mutual information $I(q_\theta(X); H_\text{temp})$, specifically:

$$\mathcal{L}_\text{InfoNCE} \geq \log(B) - I(q_\theta(X);\, H_\text{temp}) \cdot C$$

for some constant $C > 0$ that depends on the distribution. `[MATHEMATICAL DERIVATION — Poole et al. 2019]`

Minimising $\mathcal{L}_\text{InfoNCE}$ therefore maximises a lower bound on this mutual information. `[MATHEMATICAL DERIVATION]`

**Temperature justification (τ = 0.07)**: `[REASONABLE INFERENCE]`

The temperature $\tau$ controls the sharpness of the similarity distribution:

- Low $\tau$ (< 0.1): Hard negatives are heavily penalised; the model learns to form tightly clustered representations. This is desirable when negative pairs are genuinely dissimilar.  
- High $\tau$ (> 0.5): Gradient signal is weak; loss approximates a soft version of average similarity.

$\tau = 0.07$ follows the CLIP (Radford et al. 2021) and SimCLR (Chen et al. 2020) empirical findings that temperature near 0.07 yields the best representation quality for contrastive objectives. With a small dataset, low $\tau$ is preferable to avoid degenerate collapse (all embeddings mapping to the same point) while maintaining discriminative pressure. `[REASONABLE INFERENCE — requires empirical tuning; $\tau \in \{0.05, 0.07, 0.10, 0.20\}$ should be swept]`

### 3.2 Risk Analysis — Sample Size and InfoNCE Variance

**This is the most critical limitation of Candidate B.** `[MATHEMATICAL DERIVATION]`

With $N_\text{pair} \approx 100$–$150$ participants having both modalities, the maximum possible batch size is $B \leq N_\text{pair}$.

For a typical batch of $B = 32$:
- Number of negatives per anchor: $B - 1 = 31$
- The InfoNCE estimator at batch size $B$ is known to have variance proportional to $1 / (B-1)$ `[MATHEMATICAL DERIVATION — Wang & Isola 2020]`

For small $B$ (e.g. $B = 32$), this means:

$$\text{Var}(\hat{I}) \propto \frac{1}{31} \approx 0.032 \text{ per anchor}$$

This is **high variance**. In the SimCLR literature, batch sizes of 256–4096 are typical precisely to reduce this variance. At $B = 32$ with $N_\text{pair} = 100$, the estimator will be noisy. `[MATHEMATICAL DERIVATION]`

**Practical implication** `[HONEST LIMITATION — N = 341 is insufficient for InfoNCE at standard settings]`:

| Batch size $B$ | Effective negatives | InfoNCE variance regime | Verdict |
|---|---|---|---|
| 32 | 31 | High — gradient is noisy | Marginal; use gradient accumulation |
| 64 | 63 | Moderate | Maximum feasible with $N_\text{pair} \approx 100$ |
| 128 | 127 | Lower | Infeasible without full paired set |
| 341 | 340 | Low | Only possible with all participants + both modalities |

**Recommended mitigation**: Use gradient accumulation to simulate $B_\text{effective} = 64$ using $B_\text{physical} = 32$. This requires careful handling of the in-batch negative set construction. `[REASONABLE INFERENCE]`

---

## 4. Candidate C — Self-Supervised Stability (Siamese/Dropout Augmentation)

### 4.1 Mathematical Formulation

Two augmented views of the same input are generated:

$$x^{(1)} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\, \sigma^2 I), \quad \sigma = 0.05$$
$$x^{(2)} = \text{FeatureDropout}(x,\, p=0.10)$$

where FeatureDropout independently zeroes each feature with probability $p = 0.10$.

**Stability Loss**:

$$\mathcal{L}_\text{stability} = \left\| q_\theta(x^{(1)}) - q_\theta(x^{(2)}) \right\|_2^2$$

Alternatively, using cosine distance for scale invariance:

$$\mathcal{L}_\text{stability}^{\cos} = 1 - \frac{q_\theta(x^{(1)})^\top q_\theta(x^{(2)})}{\|q_\theta(x^{(1)})\|_2 \cdot \|q_\theta(x^{(2)})\|_2}$$

### 4.2 Biomedical Justification for Augmentations

**Gaussian noise ($\sigma = 0.05$)**: Models **measurement noise** in molecular assays. CSF biomarker measurements (e.g., α-synuclein, Aβ42, tau) have coefficient of variation (CV) of approximately 5–15% in PPMI-class cohorts. `[REASONABLE INFERENCE — consistent with PPMI assay documentation]` Noise with $\sigma = 0.05$ on [0,1]-rescaled data corresponds to ~5% relative noise, which is at the lower bound of realistic measurement error.

**Feature Dropout (p = 0.10)**: Models **biological variability** and **missing modality** conditions. In a real cohort, up to 10–15% of biomarker measurements may be missing or below the assay detection limit. `[REASONABLE INFERENCE]`

### 4.3 Risk Analysis — Does Stability Mean Biological Relevance?

**No, not in general.** `[MATHEMATICAL DERIVATION]`

A representation is stable under augmentation if and only if the augmentation changes the input along directions the encoder deems unimportant. If the augmentation is arbitrary (noise uncalibrated to measurement precision), the encoder may:

1. Learn to **ignore** entire features that carry biological signal (a noisy feature is easier to discard than to encode faithfully).  
2. Converge to a **constant representation** $q_\theta(x) = c$ for all $x$, which is trivially stable but encodes zero information. This is the **representation collapse** failure mode. `[MATHEMATICAL DERIVATION]`

**Specific concern for NĀḌĪ-Q**: Phase 8 showed that QVBE-v2 effective rank is only 8.68 out of 23 observables. `[VERIFIED FROM EXPERIMENT OUTPUT — test2_effective_rank = 8.679]` A stability objective could encourage the model to collapse further toward a rank-1 or rank-0 representation.

**Conclusion**: Candidate C is **REJECTED** as a primary objective. `[MATHEMATICAL DERIVATION]`

---

## 5. Selection, Recommendation, and Combined Loss

### 5.1 Decision Summary

| Candidate | Role | Decision | Justification |
|-----------|------|----------|---------------|
| **B — InfoNCE** | **PRIMARY** | ✅ Adopted | Directly targets cross-modal biological relationships; theoretically grounded as MI lower bound |
| **A — Reconstruction** | **REGULARISER** | ✅ Adopted (λ = 0.1) | Prevents trivial constant embeddings; adds information-preservation pressure |
| **C — Stability** | — | ❌ Rejected | Risk of representation collapse; arbitrary augmentations may destroy clinically relevant signal |

### 5.2 Combined Loss Function

$$\boxed{\mathcal{L}_\text{total} = \mathcal{L}_\text{InfoNCE} + \lambda \cdot \mathcal{L}_\text{reconstruction}, \quad \lambda = 0.1}$$

Expanded form:

$$\mathcal{L}_\text{total} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\!\left(\text{sim}(q_{\theta,i},\, h_{\text{temp},i}) / \tau\right)}{\sum_{j=1}^{B} \exp\!\left(\text{sim}(q_{\theta,i},\, h_{\text{temp},j}) / \tau\right)} + \frac{0.1}{d} \left\| x - g_\phi(q_\theta(x_\text{masked})) \right\|_2^2$$

### 5.3 Coefficient Justifications

| Coefficient | Value | Justification |
|-------------|-------|---------------|
| $\tau$ (temperature) | 0.07 | Follows CLIP/SimCLR empirical optimum for contrastive objectives; low $\tau$ necessary for discriminative signal at small batch sizes. `[REASONABLE INFERENCE — requires ablation sweep over {0.05, 0.07, 0.10, 0.20}]` |
| $\lambda$ (reconstruction weight) | 0.1 | Reconstruction is a regulariser, not the primary objective. $\lambda = 0.1$ ensures it contributes ~10% of the total gradient signal. This is consistent with auxiliary loss weighting in multi-task literature (Kendall et al. 2018). `[REASONABLE INFERENCE]` |
| Masking rate | 0.30 | Masks 30% of features, forcing the quantum representation to capture inter-feature correlations. Lower rates (<0.15) provide insufficient pressure; higher rates (>0.50) make reconstruction too difficult for small quantum representations. `[REASONABLE INFERENCE — consistent with MAE literature]` |
| $\sigma$ (augmentation noise — if used for data variation) | — | Not used in primary objective; Candidate C rejected |
| $B$ (batch size) | 32 (physical), 64 (effective via accumulation) | Limited by $N_\text{pair} \approx 100$–$150$. Gradient accumulation over 2 mini-batches simulates $B_\text{eff} = 64$. `[HONEST LIMITATION]` |

### 5.4 Training Pipeline Dependency Graph

```
Data Sources
├── Molecular Biofluid (x_i)        ─┐
│   └─→ Trainable Adapter (W)        │
│       └─→ QVBE-v3 Circuit (θ)      ├─→ L_InfoNCE (PRIMARY)
│           └─→ q_mol_i              │
│                                    │
└── Temporal Motor Sequences (s_i)   │
    └─→ CTCLM-discovery (NO LABELS)  │
        └─→ h_temp_i ───────────────┘

Masked x_i ─→ QVBE-v3 (θ) ─→ q_masked ─→ Decoder (φ) ─→ x_hat ─→ L_reconstruction (REGULARISER)

L_total = L_InfoNCE + 0.1 * L_reconstruction
Gradients flow through: θ (QVBE-v3 parameters), W (Trainable Adapter), φ (Decoder)
CTCLM parameters: FROZEN during QVBE-v3 training
```

**[REASONABLE INFERENCE]**: CTCLM parameters are frozen during QVBE-v3 training. This follows the standard practice in contrastive cross-modal learning where one encoder is treated as a fixed "teacher" (here, the temporal encoder pre-trained on CTCLM-discovery).

### 5.5 Honest Summary of Limitations

> **[HONEST LIMITATION]** N = 341 is **insufficient** for InfoNCE at the batch sizes (256+) where it performs optimally in computer vision and NLP applications. With $N_\text{pair} \approx 100$–$150$ and $B = 32$, the estimator has high variance and results should be interpreted cautiously. We recommend reporting 95% confidence intervals on all downstream evaluation metrics via bootstrap resampling. This is an acknowledged constraint of the PPMI-class cohort; it cannot be circumvented by algorithm design alone.

---

## 6. References

| Reference | Used For |
|-----------|---------|
| Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748. | InfoNCE derivation |
| Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* (CLIP) | τ = 0.07 justification |
| Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* (SimCLR) | Batch size analysis |
| Pérez-Salinas et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum 4, 226. | Data Re-uploading encoding |
| He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners.* (MAE) | Masking ratio |
| Kendall et al. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses.* | λ weight |
| Wang & Isola (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity.* | InfoNCE variance analysis |
| Poole et al. (2019). *On Variational Bounds of Mutual Information.* | MI lower bound proof |

---

*Document version: 1.0 — 2026-11-18. Status: Authoritative specification for QVBE-v3 training objective.*
