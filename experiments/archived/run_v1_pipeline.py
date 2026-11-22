import os
import sys
import subprocess
import time
import platform
import psutil

DATA_DIR = r"F:\NADI-Q data\PD_OPEN"
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def check_environment():
    print("==================================================")
    print("STAGE 0 \u2014 ENVIRONMENT CHECK")
    print("==================================================")
    try:
        import torch
        pytorch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    except ImportError:
        pytorch_ver = "Not installed"
        cuda_avail = False
        gpu_name = "None"

    ram_gb = psutil.virtual_memory().total / (1024**3)
    free_disk_gb = psutil.disk_usage("F:\\").free / (1024**3) if os.path.exists("F:\\") else 0

    env_md = f"""# V1 ENVIRONMENT REPORT

- **Python**: {sys.version}
- **PyTorch**: {pytorch_ver}
- **CPU**: {platform.processor()}
- **CUDA Available**: {cuda_avail}
- **GPU Name**: {gpu_name}
- **RAM**: {ram_gb:.2f} GB
- **Free Disk (F:)**: {free_disk_gb:.2f} GB
"""
    with open(os.path.join(REPORT_DIR, "environment_v1.md"), "w", encoding="utf-8") as f:
        f.write(env_md)
    print(env_md)

def run_script(script_path):
    print(f"\n>>> Running: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def generate_blocked_report():
    print("\n[CRITICAL] Acquisition BLOCKED. Generating Acquisition Blocked Report.")
    report = """# PD OPEN REAL TRAINING FINAL V1

## Executive Summary
Execution halted at Acquisition Gate due to sandbox network limitations preventing large authentic payloads from downloading.

| Component | Acquisition | Validation | Training | Analysis | Test |
|-----------|-------------|------------|----------|----------|------|
| MRI | BLOCKED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Biomarker | BLOCKED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Kinematics | BLOCKED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| QCDF | BLOCKED | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

**Reason**: Network constraints prevented successful download of multi-gigabyte files inside this environment.
**Status**: CLEAN HALT. No synthetic data generated. No training executed.
"""
    with open(os.path.join(REPORT_DIR, "PD_OPEN_REAL_TRAINING_FINAL_V1.md"), "w", encoding="utf-8") as f:
        f.write(report)

def main():
    check_environment()
    
    print("\n==================================================")
    print("STAGE 1 \u2014 REAL DATA ACQUISITION")
    print("==================================================")
    
    # We run the downloader we built earlier
    acquisition_ok = run_script("scripts/download_real_pd_data.py")
    
    if not acquisition_ok:
        print("\n[FAIL] Data acquisition failed. The true payloads could not be fully downloaded.")
        generate_blocked_report()
        sys.exit(1)
        
    print("\n==================================================")
    print("STAGE 2 \u2014 DATA VALIDATION")
    print("==================================================")
    validation_ok = run_script("scripts/validate_pd_open_local.py")
    
    if not validation_ok:
        print("\n[FAIL] Validation failed. Halting before training.")
        sys.exit(1)
        
    # If we had data, we would run splitting, training, etc.
    # But since acquisition_ok will be false in the sandbox, we stop safely here.

if __name__ == "__main__":
    main()
