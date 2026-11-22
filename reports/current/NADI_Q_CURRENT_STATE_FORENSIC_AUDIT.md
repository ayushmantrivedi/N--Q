# NĀḌĪ-Q CURRENT-STATE FORENSIC AUDIT V1.3

## 1. Executive Summary
This report presents a complete, evidence-based inventory of the Nāḍī-Q repository and its datasets. The repository contains functional architectural components for Quantum Variational Biomarker Encoding (QVBE), 3D MRI β-VAE, Continual Temporal Clinical Latent Model (CTCLM), and Quantum Clinical Decision Fusion (QCDF). However, scientific validity currently extends ONLY to the Biomarker branch. The MRI and CTCLM training pipelines simulate data (via 	orch.randn) and are thus scientifically invalid. Cross-dataset subject overlap is impossible due to disjoint cohorts. Q-XAI is unimplemented. 

## 2. Full Repository Map
- **src/nadiq/models/quantum**: Contains actual PyTorch/PennyLane implementations for qvbe.py and ctclm.py.
- **src/nadiq/models/mri**: Contains vae.py and mri_encoder.py.
- **src/nadiq/models/biomarkers**: Contains iomarker_encoder.py.
- **src/nadiq/models/fusion**: Contains qcdf.py.
- **src/nadiq/explainability**: Empty.
- **scripts/**: Orchestrators and runners (
un_v2_biomarker_real.py, 	rain_mri.py, 	rain_ctclm.py). Many scripts are stale or simulate training.
- **F:\NADI-Q data\PD_OPEN**: External volume containing actual datasets (
aw/), checkpoints, and metrics.

## 3. Actual Architecture
- **MRI Branch**: 3D T1 MRI → 3D Conv Encoder → μ, logσ² → Reparameterization (Latent=16) → QVBE (4 qubits). 
- **Biomarker Branch**: Numeric tabular vector → Standard Scaler → QVBE (4 qubits) → Logistic Regression head. 
- **Temporal/Kinematics Branch**: Timestep vector + Time offset → Concatenation → GRU (batch_first) → h_final.
- **Fusion (QCDF)**: Modality embeddings (4, 4, 32 dims) → independent linear projections (dim=16) → concat with availability boolean indicators → MLP (hidden=32, num_classes=3).

## 4. Architecture Fidelity Audit
- **MRI**: MATCH. The code reflects the design.
- **Biomarker**: MATCH. The code reflects the design.
- **Temporal**: MATCH. The code reflects the design.
- **Fusion**: MATCH. The code reflects the design.
- **Q-XAI**: MISMATCH. Not implemented.

## 5. QVBE Audit
- **Implementation**: src/nadiq/models/quantum/qvbe.py
- **Details**: Generic feature vector → Linear Projection (ℝ^input_dim → ℝ^n_qubits) → Angle Encoding via tanh → PennyLane VQC (Ry, Rz, Ry rotations per layer + CNOT ring topology) → PauliZ Measurement.
- **U3 Status**: MISMATCH. The implementation uses explicit Ry/Rz/Ry rotations, not generic U3 gates.
- **Trained Branches**: Biomarker branch has real trained QVBE weights. MRI is simulated.

## 6. β-VAE Audit
- **Implementation**: src/nadiq/models/mri/bvae.py
- **Details**: 3D Conv Encoder, Linear Latent Space, 3D ConvTranspose Decoder.
- **Real Training**: NEVER TRAINED on real MRI. Checkpoint pd_mri_bvae_v1 is a simulated pilot checkpoint trained on 	orch.randn.

## 7. CTCLM Audit
- **Implementation**: src/nadiq/models/quantum/ctclm.py
- **Details**: Time encoder MLP, sequence concatenation, GRU processing.
- **Real Training**: NEVER TRAINED. The data acquisition (PhysioNet) failed. Checkpoint is a pilot trained on 	orch.randn.

## 8. QCDF Audit
- **Implementation**: src/nadiq/models/fusion/qcdf.py
- **Acceptance**: CAN accept the requested embeddings without code changes, utilizing zero-padding + availability indicators.
- **Real Training**: NEVER TRAINED. Same-subject multimodal data does not exist.

## 9. Q-XAI Audit
- **Status**: MISSING. src/nadiq/explainability/__init__.py is empty.

## 10. Dataset Inventory
- **MRI**: OpenNeuro ds001907. 18 NIfTI files, 9 subjects. Valid but sample-limited.
- **Biomarker**: UCI Parkinson\'s ID 174. 195 rows, 32 subjects (PD: 24, CTL: 8). Validated and split properly in V2.0.
- **Temporal**: Missing. Gaitpdb failed. OpenNeuro ds003490 blocked.

## 11. MRI Audit
- **Subjects**: 9 unique subjects (sub-RC4101, sub-RC4103, sub-RC4106, sub-RC4107, sub-RC4109, sub-RC4110, sub-RC4111, sub-RC4112, sub-RC4113).
- **Adequacy**: Insufficient for clinical classification or representation learning (N=9 is too small for 3D CNN). No multimodal subjects available.

## 12. Biomarker Audit
- **Rows**: 195
- **Unique Participants**: 32 (Extracted from name: S01-S50 mapping).
- **Integrity**: Participant identity preserved in canonical CSV and splits. 

## 13. Temporal/Kinematics Audit
- **Status**: FAILED. Files in quarantine (gaitpdb.zip.part). No real temporal dataset available locally.

## 14. Cross-Subject Overlap Audit
- **Overlap**: IMPOSSIBLE.
- **Reasoning**: Identifiers belong to entirely different cohorts (sub-RC... for MRI, S01... for Biomarker, GaCo... for Kinematics). Same-subject fusion is physically impossible without new paired data.

## 15. Checkpoint Audit
- **biomarker_qvbe_best.pt**: REAL. Trained on UCI data.
- **biomarker_classical_best.pt**: REAL. Trained on UCI data.
- **pd_mri_bvae_v1/pilot_checkpoint.pt**: SIMULATED. Trained on 	orch.randn.
- **pd_ctclm_v1/pilot_checkpoint.pt**: SIMULATED. Trained on 	orch.randn.

## 16. Embedding Audit
- **biomarker_qvbe_train/val/test.csv**: REAL_MODEL_OUTPUT. 
- All V1.2 embeddings: SIMULATED (using 
p.random.randn).

## 17. Metric Provenance Audit
- **V2.0 Biomarker Metrics**: VALID. Real data, real participant splits, actual predictions.
- **V1.2 Biomarker/MRI Metrics**: INVALID. Generated via 
p.random.randn offset manipulations.

## 18. Training Pipeline Audit
- **run_v2_biomarker_real.py**: VALID. Contains real stratified subject-level splitting, gradient checks, and no-leakage guarantees.
- **train_mri.py**: INVALID. Runs simulated inputs.
- **train_ctclm.py**: INVALID. Runs simulated inputs.

## 19. Split/Leakage Audit
- **Biomarker V2.0**: PARTICIPANT-LEVEL (Validated).
- **Historical Biomarker V1.2**: ROW-LEVEL (Leaked).

## 20. Test Suite Audit
- **Status**: The test suite guarantees SOFTWARE CORRECTNESS (tensor shapes, configurations), NOT scientific validity.

## 21. Configuration Audit
- **Authoritative**: configs/experiments/pd_open_biomarker_final_v2.yaml. Other configs contain mocked/simulated assumptions.

## 22. Environment Audit
- **Hardware**: Windows, CUDA unavailable (using CPU), ~7.4 GB RAM.

## 23. Documentation Consistency Audit
- **Contradiction**: Documentation claims U3 implementation; source code uses explicit Ry/Rz/Ry rotations.
- **Contradiction**: Reports claim multimodal integration readiness, but no paired subjects exist.
- **Contradiction**: V1.2 historical reports claim quantum advantage; metrics were manually hardcoded.

## 24. Capability Matrix
| Capability | Implemented | Tested | Real Data | Scientifically Validated | Notes |
|------------|-------------|--------|-----------|---------------------------|-------|
| MRI preprocessing | Yes | Soft | No | No | NIfTI loaders exist, untuned |
| β-VAE | Yes | Soft | No | No | Simulated inputs only |
| MRI QVBE | Yes | Soft | No | No | Simulated inputs only |
| Biomarker preprocessing | Yes | Yes | Yes | Yes | StandardScaler |
| Biomarker QVBE | Yes | Yes | Yes | Yes | Functional and validated |
| CTCLM | Yes | Soft | No | No | Simulated inputs only |
| EEG adapter | Yes | Soft | No | No | Missing datasets |
| QCDF | Yes | Soft | No | No | Missing multimodal subjects |
| Q-XAI | No | No | No | No | Empty init |
| Multimodal fusion | No | No | No | No | Missing paired data |

## 25. Data-to-Model Traceability
- **Biomarker**: RAW ✅ → preprocessing ✅ → representation ✅ → model ✅ → checkpoint ✅ → embedding ✅ → prediction ✅ → metric ✅
- **MRI**: RAW ✅ → preprocessing ❌ → β-VAE ❌ → training ❌ → checkpoint ❌ → QVBE ❌ → metric ❌
- **Temporal**: RAW ❌ → preprocessing ❌ → CTCLM ❌

## 26. What We Actually Have (Usable Now)
- Functional QVBE PyTorch/PennyLane encoder.
- Functional CTCLM GRU PyTorch component.
- Functional QCDF PyTorch fusion layer.
- Clean UCI Parkinson\'s Biomarker Dataset (Participant-level).

## 27. Missing Assets
- Multimodal Paired Dataset (The primary missing asset for QCDF).
- Real Temporal/Kinematics Data (PhysioNet blocked).
- Q-XAI implementations.
- Clinical-scale MRI dataset (N=9 is too small).

## 28. Blockers
- **DATA**: Insufficient MRI samples; absent Temporal samples.
- **IDENTITY**: Impossible dataset intersection for multimodal fusion.
- **SCIENTIFIC**: VQC architectures need hyperparameter tuning for larger datasets.

## 29. Interface Readiness
- MRI encoder → QCDF: READY
- Biomarker encoder → QCDF: READY
- CTCLM → QCDF: READY
- QCDF → prediction: READY
- prediction → Q-XAI: MISSING

## 30. Inputs Needed for Next Design Phase
- **Biomarker Vectors**: Shape (B, 4) from QVBE.
- **Current Fusion Protocol**: QCDF accepts vectors and binary availability masks. 
- **Requirement**: We need a strategy to source genuinely paired multimodal data (e.g., ADNI, PPMI) OR a design to evaluate QCDF using synthesized subjects or domain-adapted embeddings, since current public data cohorts are perfectly disjoint.

## 31. Master Summary
| Area | Current State | Evidence | Confidence |
|------|---------------|----------|------------|
| Architecture | Fully defined | Source Code | High |
| MRI | Simulated | Checkpoints | High |
| Biomarker | Validated | Checkpoints/Logs | High |
| Kinematics | Missing | Data Directory | High |
| EEG | Missing | Data Directory | High |
| QVBE | Functional | Code/Tests | High |
| β-VAE | Untrained | Code | High |
| CTCLM | Untrained | Code | High |
| QCDF | Untrained | Code | High |
| Q-XAI | Missing | Code | High |
| Checkpoints | 2 Real, 2 Fake| Files | High |
| Embeddings | V2 Real | Files | High |
| Splits | V2 Validated | Scripts/Manifests| High |
| Metrics | V2 Validated | Logs | High |
| Data pairing | Impossible | Identifiers | High |
| Compute | CPU Limited | Hardware | High |
