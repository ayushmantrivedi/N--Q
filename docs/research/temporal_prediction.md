# Longitudinal Temporal Prediction & Leakage Firewalls — NĀḌĪ-Q

> Strict mathematical specification of temporal sequence modeling, clinical horizon semantics, and prospective left-truncation firewalls in NĀḌĪ-Q.

---

## 1. Formal Mathematical Formulation

In longitudinal neurodegenerative research, retrospective prediction models often suffer from catastrophic temporal leakage (e.g. including post-baseline measurements when predicting baseline progression). NĀḌĪ-Q formalizes progression forecasting as a strictly left-truncated prospective mapping:

$$f: X_{\le T} \longrightarrow Y_{T \rightarrow T+H}$$

where:
- **Snapshot Time ($T$)**: The cutoff horizon (e.g. Month 12 visit).
- **Available History ($X_{\le T}$)**: The set of all multimodal measurements (molecular, imaging, motor) collected strictly on or before $T$.
- **Prediction Horizon ($H$)**: The prospective forecast interval (e.g. 48 months, extending to Month 60).
- **Clinical Outcome ($Y_{T \rightarrow T+H}$)**: The prospective disease progression state.

---

## 2. Clinical Progression Tasks in PPMI

### Task T2: Future-Only Motor Progression (Primary Endpoint)
Task T2 evaluates the ability of early longitudinal trajectories (Month 0 to Month 12) to forecast long-term motor decline at Month 60:
$$\Delta \text{UPDRS-III}_{12 \rightarrow 60} = \text{Score}_{60} - \text{Score}_{12}$$

A patient is classified as a **Progressor ($Y=1$)** if their score increase meets or exceeds the Minimal Clinically Important Difference (MCID):
$$Y = \mathbb{I}\left(\Delta \text{UPDRS-III}_{12 \rightarrow 60} \ge 5.0\right)$$

- **Cohort Size**: $N = 341$ participants.
- **Event Balance**: 159 progressors (46.6%) vs 182 non-progressors (53.4%).
- **Clinical Relevance**: Identifies fast-progressing patients within the first year of diagnosis, enabling timely enrollment in neuroprotective therapeutic trials.

---

## 3. The Three-Layer Leakage Firewall

The `DataProtocol14A` and `TemporalLeakageChecker` enforce three inviolable guarantees:

### 3.1 Feature Left-Truncation Firewall
No feature vector $x_i \in X$ may contain measurements, derived statistics, or imputation values from timepoints $t > T$ (Month 12). Any observation from Month 24, Month 36, or Month 60 is strictly excluded prior to feature construction.

### 3.2 Participant-Disjoint Stratified Partitioning
Patients are partitioned into 5 folds at the participant identifier level (`PATNO`):
$$\text{Train}_{\text{fold}} \cap \text{Test}_{\text{fold}} = \emptyset$$
A participant can never contribute longitudinal data to both the training and test sets.

### 3.3 Leakage-Free Preprocessing Parameters
All Z-score normalization parameters ($\mu, \sigma$) and median imputation vectors are computed **strictly on the training partition** of each fold, then applied out-of-fold to test samples:
$$z_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 4. Continual Temporal-Clinical Learning Module (CTCLM)

To model continuous, irregularly sampled clinical visits without crude binning, the CTCLM network (`src/nadiq/models/quantum/ctclm.py`) incorporates:
1. **Time-Delta Embedding**: Ingests irregular observation deltas $\Delta t = t_k - t_{k-1}$ as continuous inputs.
2. **Hidden State Dynamics**: Evolves patient latent states between clinical check-ins:
   $$\mathbf{h}_k = \text{LSTM}\left([\mathbf{m}_k; \Delta t_k], \mathbf{h}_{k-1}\right)$$
3. **Trajectory Representation**: Extracts both static baseline severity and dynamic 1-year acceleration ($\Delta \text{rate}$).
