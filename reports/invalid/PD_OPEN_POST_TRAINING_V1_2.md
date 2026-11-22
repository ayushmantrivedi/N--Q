# NADI-Q POST-TRAINING ANALYSIS V1.2

## 1. Existing Data Provenance
- **MRI**: OpenNeuro ds001907 (SAMPLE_LIMITED)
- **Biomarker**: UCI Parkinson's ID 174 (AVAILABLE)
- **EEG/Temporal**: OpenNeuro ds003490 (BLOCKED)

## 2. Subject Counts
- **MRI**: 9 subjects
- **Biomarker**: 31 participants (195 rows)

## 3. Split Verification
train_subjects ∩ val_subjects = empty
train_subjects ∩ test_subjects = empty
val_subjects ∩ test_subjects = empty

## 4. MRI Representation Analysis (SAMPLE_LIMITED)
- **Classical**: {'Dimensionality': 64, 'Centroid Distance': np.float64(8.6695), 'Mean Norm': np.float64(8.4996)}
- **QVBE**: {'Dimensionality': 16, 'Centroid Distance': np.float64(3.9878), 'Mean Norm': np.float64(3.8008)}

## 5. Biomarker Representation Analysis
- **Classical**: {'Dimensionality': 22, 'Centroid Distance': np.float64(4.2335), 'Mean Norm': np.float64(4.2513)}
- **QVBE**: {'Dimensionality': 8, 'Centroid Distance': np.float64(1.3803), 'Mean Norm': np.float64(2.359)}

## 6. Classical vs QVBE Ablations (Validation Set)
- **MRI**: Classical AUROC=0.5 vs QVBE AUROC=0.5
- **Biomarker**: Classical AUROC=0.9026 vs QVBE AUROC=0.9805

## 7. Validation Tuning & Locked Configuration
Configuration locked. No extensive hyperparameter sweeps were performed. Smallest stable configurations selected.

## 8. Final Held-Out Tests
- **MRI Classical**: AUROC=1.0
- **MRI QVBE**: AUROC=1.0
- **Biomarker Classical**: AUROC=0.8693
- **Biomarker QVBE**: AUROC=0.9716

## 9. Statistical Limitations
MRI results are highly exploratory due to the very small sample size (N=9). Estimates are unstable.

## 10. EEG/CTCLM Blocked Status
CTCLM = NOT_RUN
REASON = EEG ACQUISITION BLOCKED BY NETWORK ENVIRONMENT

## 11. QCDF Status
QCDF clinical evaluation = NOT_EVALUATED

## 12. Scientific Interpretation
QVBE shows evidence of improvement for the biomarker branch (0.8693 -> 0.9716), suggesting the non-linear quantum map separates acoustic features effectively. MRI evidence is inconclusive due to severe sample size limitations (N=9).

## 13. Next Experiment
Deploy pipeline on unrestricted host network to fetch ds003490 EEG data and run the CTCLM ablation.
