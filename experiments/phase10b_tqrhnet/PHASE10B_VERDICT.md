# NĀḌĪ-Q — PHASE 10B FINAL SCIENTIFIC VERDICT
## TQ-RHNet: Time-Conditioned Quantum Relational Hypergraph Network

**Date**: November 22, 2026  
**Status**: EXPERIMENT COMPLETED & RIGOROUSLY AUDITED  
**Cohort**: Authoritative Longitudinal PPMI Multimodal Cohort ($N=341$ participants, 5-year progression follow-up)  
**Execution Environment**: Python 3.12.4, PennyLane 0.41.1 (`default.qubit` simulator), PyTorch 2.4.1+cpu  
**Machine-Readable Experiment Artifacts**:
- `TQRHNET_RESULTS.csv`
- `TQRHNET_ABLATION_RESULTS.csv`
- `TQRHNET_NODE_DRIFT_RESULTS.json`
- `DYNAMIC_LANDMARK_PREDICTIONS.csv`
- `EVENT_CHAINS_RESULTS.json`

---

## 1. CRITICAL AUDIT: TARGET LEAKAGE IN PHASE 10A BASELINE

Before evaluating Phase 10B architectures, a forensic audit of the Phase 10A handcrafted motor velocity baseline was executed.

- **Audited Metric**: Phase 10A reported `annualized_velocity_5yr` with an AUROC of $1.0000$ and Brier score of $0.0000$.
- **Mathematical Audit Finding**:
  $$\text{annualized\_velocity\_5yr} = \frac{\text{NP3TOT}_{\text{V12}} - \text{NP3TOT}_{\text{BL}}}{5.0}$$
  $$\text{target\_progression\_5yr\_mcid5} = \mathbb{I}\left(\text{NP3TOT}_{\text{V12}} - \text{NP3TOT}_{\text{BL}} \ge 5.0\right)$$
  The predictor utilized observations from visit V12 (the 60-month follow-up visit). The condition $\text{annualized\_velocity\_5yr} \ge 1.0$ is **mathematically identical** to the target label.
- **Verdict on Phase 10A Baseline**: `LEAKAGE CONFIRMED (CONTAMINATED)`.
- **Corrective Action**: `annualized_velocity_5yr` is permanently excluded from all valid benchmark comparisons.
- **Legitimate Replacement**: Replaced by `early_trajectory_delta_1yr` ($\text{NP3TOT}_{\text{V04}} - \text{NP3TOT}_{\text{BL}}$, Months 0 to 12). This represents a strictly prospective, non-leaked clinical motor baseline ($r = 0.3149$).

---

## 2. FIVE-ARCHITECTURE CONTROLLED BENCHMARK RESULTS (Strict 5-Fold CV, $N=341$)

All models evaluated on the identical 5-fold cross-validation split manifest (`PARTICIPANT_SPLIT_MANIFEST_PHASE6.csv`) using identical input features across 4 longitudinal visits (BL, V04, V06, V08):

| Architecture / Baseline | Pooled AUROC (95% CI) | Pooled AUPRC | Pooled Brier (95% CI) | ECE | Calib Slope | Sens / Spec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A: Pairwise Temporal Graph** | 0.5335 [0.4712, 0.6003] | 0.5428 | 0.3791 [0.3359, 0.4225] | 0.3395 | 0.0351 | 0.462 / 0.541 |
| **Variant B: Classical Hypergraph MLP** | **0.5657** [0.4995, 0.6320] | **0.5749** | **0.3605** [0.3178, 0.4039] | **0.3220** | **0.0744** | 0.402 / 0.651 |
| **Variant C: Phase 10A Static Quantum** | 0.5108 [0.4461, 0.5760] | 0.5213 | 0.4432 [0.3920, 0.4907] | 0.4333 | 0.0091 | 0.556 / 0.488 |
| **Variant D: TQ-RHNet (2-Body Time-Cond.)** | 0.5527 [0.4911, 0.6141] | 0.5361 | 0.4118 [0.3645, 0.4555] | 0.4081 | 0.0424 | 0.497 / 0.599 |
| **Variant E: TQ-RHNet Full (2B+3B+Drift)** | 0.5493 [0.4874, 0.6096] | 0.5281 | 0.4201 [0.3730, 0.4650] | 0.4079 | 0.0378 | 0.497 / 0.564 |
| **Ablation E: TQ-RHNet (No Drift)** | 0.5524 [0.4925, 0.6135] | 0.5338 | 0.4220 [0.3751, 0.4681] | 0.4160 | 0.0386 | 0.515 / 0.547 |
| **Baseline 1: Clean Early Motor Delta** | **0.6809** [0.6237, 0.7331] | **0.6502** | **0.2254** [0.2120, 0.2405] | **0.0429** | **0.9754** | **0.580 / 0.651** |
| **Baseline 2: CTCLM Functional Latent** | 0.5346 [0.4747, 0.5952] | 0.5456 | 0.2657 [0.2484, 0.2834] | 0.1321 | 0.2014 | 0.491 / 0.535 |

---

## 3. DYNAMIC LANDMARK PREDICTIONS (Months 0, 12, 24, 36)

Evaluating 5-year progression dynamically across sequential observation windows:

| Clinical Landmark | Nominal Time | TQ-RHNet Full (Variant E) | Classical Hypergraph (Variant B) | CTCLM Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Month 0 (BL)** | $T=1$ visit | AUROC = 0.5672, Brier = 0.3227 | AUROC = 0.5648, Brier = 0.3255 | AUROC = 0.6046, Brier = 0.2421 |
| **Month 12 (V04)** | $T=2$ visits | AUROC = 0.5465, Brier = 0.3619 | **AUROC = 0.6213**, Brier = 0.2984 | AUROC = 0.5619, Brier = 0.2519 |
| **Month 24 (V06)** | $T=3$ visits | AUROC = 0.5356, Brier = 0.3753 | **AUROC = 0.5860**, Brier = 0.3274 | AUROC = 0.5377, Brier = 0.2590 |
| **Month 36 (V08)** | $T=4$ visits | AUROC = 0.5622, Brier = 0.3612 | **AUROC = 0.5999**, Brier = 0.3258 | AUROC = 0.5280, Brier = 0.2624 |

*Key Empirical Finding*: The **Classical Hypergraph MLP consistently outperformed TQ-RHNet Full at every multi-visit landmark** (e.g. Month 12: $0.6213$ vs $0.5465$, $\Delta = +0.0748$; Month 24: $0.5860$ vs $0.5356$; Month 36: $0.5999$ vs $0.5622$).

---

## 4. NODE ACTIVATION & QUANTUM RELATIONAL DRIFT VERIFICATION

Tested on representative participant PATNO 3001 across Baseline and Month 12:
- **Quantum State Space**: $2^8 = 256$ complex amplitudes.
- **Quantum Relational Drift**: $D_Q(t) = 1 - |\langle \psi_{\text{BL}} | \psi_{\text{V04}} \rangle|^2 = 0.171383$ (Fidelity $\mathcal{F}_Q = 0.828617$).
- **Classical Comparison Distances**: Euclidean = $5.5135$, Cosine = $0.0078$, RBF = $0.2187$.
- **Drift Perturbation Sensitivity**: Adding $+1.0$ std perturbation to the Molecular node shifted drift by $\Delta D_Q = 0.020729$ (from $0.171383$ to $0.192112$).
- **Incidence Structure Verification**:
  - Perturbing Molecular node updated incident hyperedges: $\|\Delta e_{\text{cross}}\| = 0.274142$, $\|\Delta e_{\text{intra\_mol}}\| = 1.405204$.
  - Non-incident hyperedges prior to message passing remained strictly zero:
    $$\|\Delta e_{\text{intra\_struct}}\|_2 = 0.00000000 \quad \text{and} \quad \|\Delta e_{\text{intra\_func}}\|_2 = 0.00000000$$
  - Graph message passing routed information to connected Structural ($\|\Delta m\| = 0.085756$) and Functional ($\|\Delta m\| = 0.102373$) nodes, updating patient hidden state by $\|\Delta h\| = 0.135078$.

---

## 5. DIRECT ANSWERS TO THE MANDATED SCIENTIFIC QUESTIONS

### 1. Does time-conditioned entanglement improve cross-modal representation?
**Answer**: **Partially (`VERIFIED`)**, but strictly within the quantum domain.  
Time-conditioning coupling strength ($\theta_0 + \theta_t \Delta t$) improved the quantum model from AUROC $0.5108$ (Phase 10A static) to $0.5527$ (Variant D). However, it remains inferior to the Classical Hypergraph MLP (AUROC $0.5657$). Time-conditioning mitigates static phase noise, but does not overcome classical representation capacity.

### 2. Does quantum relational drift add information beyond classical distances?
**Answer**: **No (`FAILED`)**.  
Ablation analysis confirms that removing quantum relational drift ($D_Q = 0$) resulted in an AUROC of $0.5524$, compared to $0.5493$ with drift enabled ($\Delta = -0.0031$). The drift metric $D_Q$ did not provide complementary prognostic information beyond classical Euclidean and Cosine rate-of-change vectors.

### 3. Do three-body interactions help?
**Answer**: **No (`FAILED`)**.  
Comparing Variant D (2-body $R_{ZZ}$ only, AUROC $0.5527$) against Ablation E (2-body + 3-body $R_{ZZZ}$, AUROC $0.5524$) shows that introducing 3-body entangling unitaries and 3-body connected cumulants slightly degraded predictive accuracy ($\Delta = -0.0003$). The 3-body parameterization adds rotational dispersion without capturing usable higher-order clinical signal.

### 4. Does the model improve five-year prediction?
**Answer**: **No (`FAILED`)**.  
TQ-RHNet Full achieved pooled AUROC $0.5493$ (95% CI [0.4874, 0.6096]), which is:
- Inferior to Classical Hypergraph MLP ($0.5657$)
- Inferior to CTCLM Functional Baseline at Baseline ($0.6046$)
- Markedly inferior to the simple, clean 1-year early motor delta baseline ($0.6809$).

### 5. Does it discover reproducible event sequences?
**Answer**: **No (`FAILED`)**.  
Longitudinal event sequence analysis across 2,206 visits confirms:
- Forward Trimodal Cascade ($\text{Mol} \to \text{Struct} \to \text{Motor}$): 34 patients, Odds Ratio = $1.51$, $p = 0.2818$ (not significant).
- Reverse Control Sequence ($\text{Motor} \to \text{Mol}$): 10 patients, Odds Ratio = $1.55$, $p = 0.5397$ (not significant).
The reverse sequence exhibits a comparable Odds Ratio to the forward cascade, demonstrating that observed orderings reflect irregular clinic visit intervals rather than invariant pathogenic causality.

### 6. Should TQ-RHNet enter the final NĀḌĪ-Q architecture?
**Answer**: **NO (`REJECTED`)**.  
TQ-RHNet should **NOT** enter the production NĀḌĪ-Q diagnostic architecture. It introduces severe computational simulation overhead, degrades probability calibration (Brier $0.4201$ vs $0.2254$), and fails to outperform simpler classical models or standard clinical rate features.

---

## 6. FINAL SUMMARY TABLE

| Hypothesis / Dimension | Phase 10B Result | Final Scientific Assessment |
| :--- | :---: | :--- |
| **Leakage Audit** | `VERIFIED` | Phase 10A velocity baseline leaked 5-year target; permanently removed. |
| **Incidence Message Passing** | `VERIFIED` | Non-incident edges strictly zero; message passing functions as derived. |
| **Time-Conditioned $R_{ZZ}$** | `VERIFIED` | Outperformed static quantum (+0.0419 AUROC); underperformed classical. |
| **Quantum Relational Drift $D_Q$** | `FAILED` | Added zero predictive value; classical distances suffice. |
| **3-Body Interaction $R_{ZZZ}$** | `FAILED` | Cumulants provided no advantage; increased variance. |
| **Overall 5-Year Prognosis** | `FAILED` | Classical baseline (0.6809) and Classical MLP (0.5657) beat TQ-RHNet (0.5493). |
| **Causal Event Chains** | `FAILED` | Not statistically significant ($p = 0.2818$); reverse sequence equivalent ($p = 0.5397$). |

*All source code, models, and empirical artifacts are preserved in `experiments/phase10b_tqrhnet/` and `F:\NADI-Q_AUDIT_OUTPUT\PHASE_10B\`.*
