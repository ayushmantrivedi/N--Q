<#
.SYNOPSIS
Orchestrates the entire NĀḌĪ-Q real-data training pipeline locally.
#>

$ErrorActionPreference = "Stop"

$PYTHON = "python"
$DATA_DIR = "F:\NADI-Q data\PD_OPEN"

Write-Host "=========================================================="
Write-Host " NĀḌĪ-Q LOCAL WINDOWS EXECUTION RUNNER "
Write-Host "=========================================================="

Write-Host "--- 1. Environment Check ---"
$py_ver = & $PYTHON -V
Write-Host "Python: $py_ver"
& $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
& $PYTHON -c "import psutil; print(f'RAM (GB): {psutil.virtual_memory().total / (1024**3):.2f}')"
$disk = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='F:'" | Select-Object -ExpandProperty FreeSpace
if ($disk) { Write-Host "Disk Free (F:): $([math]::Round($disk / 1GB, 2)) GB" }

Write-Host "`n--- 2. Data Download ---"
& $PYTHON scripts/download_real_pd_data.py
if ($LASTEXITCODE -ne 0) { Write-Host "Download script exited with error."; exit $LASTEXITCODE }

Write-Host "`n--- 3. Validation Gate ---"
& $PYTHON scripts/validate_pd_open_local.py
if ($LASTEXITCODE -ne 0) { Write-Host "Validation failed. Halting training pipeline."; exit $LASTEXITCODE }

Write-Host "`n--- 4. Subject Splits ---"
& $PYTHON scripts/generate_subject_splits.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n--- 5. MRI Training Pilot & Budget ---"
& $PYTHON scripts/train_mri.py
if ($LASTEXITCODE -ne 0) { Write-Host "MRI Training stopped cleanly or failed."; exit $LASTEXITCODE }

Write-Host "`n--- 6. Biomarker Training Pilot & Budget ---"
& $PYTHON scripts/train_biomarkers.py
if ($LASTEXITCODE -ne 0) { Write-Host "Biomarker Training stopped cleanly or failed."; exit $LASTEXITCODE }

Write-Host "`n--- 7. CTCLM Training Pilot & Budget ---"
& $PYTHON scripts/train_ctclm.py
if ($LASTEXITCODE -ne 0) { Write-Host "CTCLM Training stopped cleanly or failed."; exit $LASTEXITCODE }

Write-Host "`n--- 8. Evaluate & Report ---"
& $PYTHON scripts/evaluate_and_report_local.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=========================================================="
Write-Host " LOCAL EXECUTION PIPELINE COMPLETE"
Write-Host " Final report generated at F:\NADI-Q data\PD_OPEN\reports\PD_OPEN_LOCAL_EXECUTION_REPORT.md"
Write-Host "=========================================================="
