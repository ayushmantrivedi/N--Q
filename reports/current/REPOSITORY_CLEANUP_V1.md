# NĀḌĪ-Q Repository Cleanup V1

## 1. Overview
The repository has been successfully cleaned and reorganized according to the target structure. The scientific architecture remains fully intact, and strict boundaries have been placed around verified vs simulated/legacy code paths.

## 2. Structural Modifications

### Files Moved
- **Scripts**: All scripts were recategorized. Active data acquisition tools moved to `scripts/data/`, the functional biomarker trainer moved to `scripts/training/`, and evaluators to `scripts/evaluation/`.
- **Experiments**: Reorganized into `smoke/` (architecture validation), `unimodal/`, `multimodal/`, and `archived/`.
- **Configs**: The authoritative active configuration (`pd_open_biomarker_final_v2.yaml`) remains in `configs/experiments/`. Legacy simulated configurations were moved to `configs/archived/`.
- **Docs**: Markdown documentation was sorted into `architecture/`, `data/`, `experiments/`, `research/`, and `development/`.

### Quarantined / Invalid Files
The following items were identified as scientifically invalid (due to simulated `np.random`/`torch.randn` outputs or structural leakages) and have been strictly quarantined to prevent accidental reuse:
- **Scripts**: `train_mri.py` and `train_ctclm.py` moved to `scripts/archived/invalid/`. `execute_v1_2_analysis.py` renamed to `execute_v1_2_analysis_INVALID.py` in `scripts/archived/invalid/`.
- **Checkpoints**: `checkpoints/pd_mri_bvae_v1/` and `checkpoints/pd_ctclm_v1/` were moved to `checkpoints/invalid/` in the external data volume.
- **Reports**: V1.1 and V1.2 training and post-training analysis reports moved to `reports/invalid/`.

### Deleted Clutter
The following disposable items were permanently removed from the repository root to ensure a clean source tree:
- `.pytest_cache/`
- Root logs: `benchmark.log`, `ctclm_test.log`, `inspect.log`, `qvbe_test.log`, `test_baseline.log`, `test_final.log`
- Scratch/temporary scripts and reports from the `scratch/` directory.
- `inspect_report.json`

## 3. Reference Updates & Tests
- **Path Fixes**: Checked internal scripts and `local_prepare_and_train.ps1` (moved to `scripts/archived/`) to ensure the reorganization did not fundamentally break execution capabilities.
- **Test Results**: Executed `pytest` across the reorganized tree.
  - **Result**: `227 passed, 4 warnings in 23.36s`
  - All software structural tests, data contract evaluations, and scientific boundary tests remain functional. No tests were deleted.

## 4. Final Repository Tree (High-Level)
```
NADI-Q/
├── configs/          (data, models, experiments, archived)
├── data/
├── docs/             (architecture, data, experiments, research, development, archived)
├── experiments/      (smoke, unimodal, multimodal, archived)
├── reports/          (current, historical, invalid)
├── scripts/          (data, training, evaluation, analysis, utilities, archived/invalid)
├── src/nadiq/        (data, preprocessing, models/mri, models/quantum, models/fusion, evaluation, utils)
├── tests/            (unit, scientific, integration, data, archived)
├── cleanup.py
├── PROJECT_STATE.md
├── pyproject.toml
└── README.md
```
*(Checkpoints, raw data, and embeddings remain safely mapped to the `F:\NADI-Q data\PD_OPEN\` volume).*

## 5. Active NĀḌĪ-Q Surface
The following constitutes the **only** active and scientifically validated surface of the repository to be utilized in the next design phase:

1. **Modality Models**:
   - `src/nadiq/models/quantum/qvbe.py` (Quantum Variational Biomarker Encoder)
   - `src/nadiq/models/mri/bvae.py` (β-VAE Architecture - Untrained)
   - `src/nadiq/models/quantum/ctclm.py` (CTCLM Architecture - Untrained)
   - `src/nadiq/models/fusion/qcdf.py` (Fusion Layer - Untrained)
2. **Active Data Adapters**:
   - `src/nadiq/data/loaders/` (Contracts and base structures)
3. **Active Training Scripts**:
   - `scripts/training/run_v2_biomarker_real.py` (Validated Participant-level Splitter & Trainer)
4. **Active Configs**:
   - `configs/experiments/pd_open_biomarker_final_v2.yaml`
5. **Active Checkpoints**:
   - `F:\NADI-Q data\PD_OPEN\checkpoints\active\biomarker_qvbe_best.pt`
   - `F:\NADI-Q data\PD_OPEN\checkpoints\active\biomarker_classical_best.pt`
6. **Current Reports**:
   - `reports/current/NADI_Q_CURRENT_STATE_FORENSIC_AUDIT.md`
   - `reports/current/PD_OPEN_BIOMARKER_REAL_EXPERIMENT_V2.md`
   - `PROJECT_STATE.md`
