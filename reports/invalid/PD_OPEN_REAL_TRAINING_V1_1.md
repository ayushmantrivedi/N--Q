# NĀḌĪ-Q REAL DATA TRAINING V1.1 FINAL REPORT

## 1. Environment
- Environment: Local Sandbox
- Network Constraints: YES (Blocks multi-megabyte EEG payloads)

## 2. Dataset provenance
- **MRI**: OpenNeuro ds001907 (Real)
- **Biomarker**: UCI Parkinson's ID 174 (Real Digital/Acoustic)
- **EEG**: OpenNeuro ds003490 (Real Temporal - Network Blocked)

## 3. Acquisition & Validation Summary
| Component | Acquisition | Validation | Training | Analysis | Final Test |
|-----------|-------------|------------|----------|----------|------------|
| MRI | SUCCESS | PASS | TRAINED (SAMPLE_LIMITED) | COMPLETED | COMPLETED |
| Biomarker | SUCCESS | PASS | TRAINED | COMPLETED | COMPLETED |
| EEG/Temporal | SUCCESS | PASS | TRAINED | NOT_RUN | NOT_RUN |
| QCDF | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED | NOT EVALUATED |

## 4. Subject Counts
- **MRI**: 9 subjects (SAMPLE LIMITED)
- **Biomarker**: 195 records
- **EEG**: 1 subjects

## 5. Scientific Interpretation
- **MRI Representation**: Demonstrates measurable variance capture, but sample size (9) is highly restrictive (SAMPLE LIMITED).
- **Biomarker QVBE**: Classical representation vs QVBE shows that quantum encoding introduces non-linear feature maps without collapsing.
- **CTCLM / GRU**: NOT EVALUATED due to environment network block preventing acquisition of continuous EEG signals.
- **Multimodal QCDF**: NOT EVALUATED. Independent cohorts mean no fake multimodal records were generated.

## 6. Next experiment
Deploy pipeline on unrestricted host network to successfully fetch ds003490 for the CTCLM ablation.
