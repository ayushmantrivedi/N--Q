import os

REPORT_DIR = r"F:\NADI-Q data\PD_OPEN\reports"
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORT_DIR, "PD_OPEN_LOCAL_EXECUTION_REPORT.md")

md = """# NĀḌĪ-Q PD OPEN LOCAL EXECUTION REPORT

## 1. ORCHESTRATION STATUS
- **Runner**: `scripts/local_prepare_and_train.ps1`
- **Environment**: PyTorch CPU, ~7.4 GB RAM, no CUDA.
- **Status**: Halted gracefully at validation gate due to incomplete multi-gigabyte payload acquisition.

## 2. REAL DATA ACQUISITION
- **MRI** (`ds001907`): Attempted streaming S3 download. Failed (404/Network constraints).
- **Kinematics** (`gaitpdb`): Attempted PhysioNet zip extraction. Failed (file corrupt/not zip).
- **Biomarkers** (PMC Supp): Downloaded raw HTML/placeholder due to mocked discovery link.

## 3. REAL VALIDATION
- **MRI Gate**: Failed. Actual files = 0.
- **Kinematics Gate**: Failed. Actual recordings = 0.
- **Biomarkers Gate**: Failed. CSV invalid.

## 4. REAL TRAINING
- **Splits**: Generators implemented (`scripts/generate_subject_splits.py`), blocked by validation.
- **MRI Pilot Epoch**: Implemented (`scripts/train_mri.py`), blocked by validation.
- **CTCLM Pilot Epoch**: Implemented (`scripts/train_ctclm.py`), blocked by validation.
- **Biomarker Pilot Epoch**: Implemented (`scripts/train_biomarkers.py`), blocked by validation.

## 5. HARDWARE & LIMITATIONS
- We strictly enforced: "No synthetic training data. No fabricated metrics."
- The pipeline architecture remains identical and ready for deployment on a full node with data access.

## 6. FINAL COMMANDS
To execute the pipeline directly on the Windows machine once network limitations are resolved:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\local_prepare_and_train.ps1
```

Individual commands available:
```bash
python scripts/download_real_pd_data.py
python scripts/validate_pd_open_local.py
python scripts/train_mri.py
python scripts/train_biomarkers.py
python scripts/train_ctclm.py
```
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(md)
print(f"Generated {REPORT_PATH}")
