# PPMI Longitudinal Cohort — Data Specification & Governance

**Dataset Identifier**: `PPMI` (Parkinson's Progression Markers Initiative)  
**Cohort Size**: $N = 341$ participants with complete or longitudinal follow-up  
**Follow-up Duration**: Up to 60 Months (5 Years)  
**Target Modalities**: Molecular Biofluids (CSF), Structural Neuroimaging (3D T1 MRI), Clinical Motor Scales (MDS-UPDRS Part III)  
**Data Governance**: Protected Health Information (PHI). Restricted local storage; strictly excluded from version control.

---

## 1. Study Overview & Cohort Assembly

The Parkinson's Progression Markers Initiative (PPMI) is an observational, international study aimed at identifying comprehensive biomarkers of Parkinson's disease progression. Within NĀḌĪ-Q, the PPMI cohort is assembled to evaluate prospective machine learning architectures for longitudinal progression prediction.

### Participant Demographics & Cohort Characteristics
- **Total Analyzed Participants**: $N = 341$
- **Diagnostic Distribution**: De novo, untreated early Parkinson's disease participants and matched healthy controls.
- **Progression Event Rate**: 159 progression events (46.6% positive class balance on Task T2).
- **Observation Timepoints**: Baseline (Month 0), Month 12, Month 24, Month 36, and Month 60.

---

## 2. Clinical Prediction Endpoints

### Task T2: Future-Only Motor Progression (Primary Endpoint)
* **Clinical Horizon**: Baseline / Month 12 $\rightarrow$ Month 60 (5-year prospective progression).
* **Clinical Target**: Minimal Clinically Important Difference (MCID) $\ge 5.0$ points increase on the MDS-UPDRS Part III motor score:
  $$\Delta \text{UPDRS-III}_{12 \rightarrow 60} = \text{Score}_{60} - \text{Score}_{12} \ge 5.0$$
* **Strict Temporal Left-Truncation Firewall**:
  All features input to the prediction models are gathered **strictly at or before Month 12**. No clinical, imaging, or fluid measurements from Month 24, 36, or 60 are accessible during feature construction, guaranteeing mathematical zero-leakage.

### Task T1: Baseline-to-Month-60 Progression
* Prospective progression assessed across the entire 5-year span from baseline.

---

## 3. Multimodal Feature Inventory & Reference Distributions

The authoritative $N=341$ cohort features six key clinical and biological variables spanning three modalities:

| Modality | Variable Name | Clinical Description | PPMI Mean | PPMI Std | Clinical Unit |
|:---|:---|:---|:---:|:---:|:---:|
| **Molecular** | `asyn` | CSF $\alpha$-synuclein concentration | 1820.0 | 750.0 | pg/mL |
| **Molecular** | `ptau` | CSF phospho-Tau181 concentration | 15.5 | 6.2 | pg/mL |
| **Structural MRI** | `caudate` | Bilateral caudate nucleus volume | 3050.0 | 520.0 | mm³ |
| **Structural MRI** | `putamen` | Bilateral putamen volume | 3950.0 | 630.0 | mm³ |
| **Functional Motor**| `motor_total` | MDS-UPDRS Part III Baseline Total | 21.8 | 9.5 | points |
| **Functional Motor**| `motor_subscore`| 1-Year Early Motor Rate Delta | 2.2 | 3.8 | pts/yr |

### Feature Normalization Protocol
To prevent data leakage during model training and evaluation:
- Normalization parameters (mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$) are computed **strictly on training fold participants**.
- Missing measurements are imputed using training-fold medians.
- The standard Z-score transformation is applied:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 4. Preprocessing & Temporal Leakage Firewalls

1. **Left-Truncation Enforcement**: Features observed at timestamps $t > T_{\text{snapshot}}$ (Month 12) are discarded.
2. **Patient-Disjoint Splitting**: Stratified 5-fold cross-validation splits participants at the patient identifier level (`PATNO`). A patient never appears simultaneously in training and test partitions.
3. **Repeated Measurements**: Longitudinal measurements are sequenced temporally with irregular visit interval offsets ($\Delta t$).

---

## 5. Ethical & Legal Data Governance

- **Local Storage Rule**: PPMI participant-level data is stored exclusively on local encrypted drives (`F:\NADI-Q data\` or local research directory).
- **Zero Remote Transmission**: No participant-level PPMI records are uploaded to external APIs, cloud services, or public repositories.
- **Git Protection**: Filtered automatically by [.gitignore](file:///c:/Users/ayush/OneDrive/Desktop/Nadi-Q/.gitignore) via `data/*.csv`, `data/raw/`, and `*participant_split_manifest*.csv`.
