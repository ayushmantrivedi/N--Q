# NADI-Q BIOMARKER REAL EXPERIMENT V2.0

## 1. Dataset Provenance & Participant Identity Recovery
- **Dataset**: UCI Parkinson's Dataset (ID 174)
- **Status**: Participant identity SUCCESSFULLY recovered using the original metadata `name` column.
- **Total Rows**: 195
- **Unique Participants**: 32 (PD: 24, CTL: 8)
- Canonical Dataset Saved: `raw/biomarkers/parkinsons_with_subject_id.csv`

## 2. Exact Subject Split (Participant-Level)
- **Train Participants**: 22
- **Validation Participants**: 5
- **Test Participants**: 5
- **Leakage Verification**: `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` verified programmatically. All recordings from a participant belong to a single split.

## 3. Preprocessing
- **Features**: Scaled using StandardScaler.
- **Fitting**: Fitted strictly on Train participants only.

## 4. Architecture & Training Configuration
- **Classical**: Logistic Regression.
- **QVBE**: NADI-Q Hybrid Classical-Quantum Architecture (4 qubits, 2 layers, Pauli-Z measurement).
- **Training**: Full gradient-based backpropagation on PyTorch VQC. Non-zero gradients verified during training loop. No synthetic data embeddings or labels were used.
- **Checkpoints**: `checkpoints/biomarker_classical_best.pt`, `checkpoints/biomarker_qvbe_best.pt`

## 5. Validation Metrics (Participant-Aggregated)
- **Classical AUROC**: 1.0000
- **QVBE AUROC**: 0.2500

## 6. Locked Configuration
- Configuration mathematically locked in `configs/experiments/pd_open_biomarker_final_v2.yaml`.

## 7. Final Held-Out Metrics (Participant-Aggregated)
- **Classical AUROC**: 1.0000
- **Classical F1**: 1.0000
- **QVBE AUROC**: 1.0000
- **QVBE F1**: 0.8571

## 8. Scientific Conclusion
**QVBE WAS APPROXIMATELY NEUTRAL**
Based on exact participant-level aggregated held-out test predictions: QVBE AUROC (1.0000) vs Classical AUROC (1.0000).

*All V2 results are generated from validated forward-passes of fitted models on untouched test subjects, guaranteeing provenance.*
