import os
import shutil
import glob

REPO_DIR = r"c:\Users\ayush\OneDrive\Desktop\Nadi-Q"
DATA_DIR = r"F:\NADI-Q data\PD_OPEN"

# Phase 1: Create all necessary directories
repo_dirs = [
    "scripts/data",
    "scripts/training",
    "scripts/evaluation",
    "scripts/analysis",
    "scripts/utilities",
    "scripts/archived/invalid",
    "experiments/smoke",
    "experiments/unimodal",
    "experiments/multimodal",
    "experiments/archived",
    "configs/data",
    "configs/models",
    "configs/experiments",
    "configs/archived",
    "tests/archived",
    "tests/data",
    "docs/architecture",
    "docs/data",
    "docs/experiments",
    "docs/research",
    "docs/development",
    "docs/archived",
    "reports/current",
    "reports/historical",
    "reports/invalid",
]

for d in repo_dirs:
    os.makedirs(os.path.join(REPO_DIR, d), exist_ok=True)

data_dirs = [
    "checkpoints/active",
    "checkpoints/archived",
    "checkpoints/invalid",
    "embeddings/active",
    "embeddings/archived",
    "embeddings/invalid",
]

for d in data_dirs:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

# Phase 2: Move Repo Files
repo_moves = [
    # Scripts
    ("scripts/acquire_kinematics_v3.py", "scripts/data/acquire_kinematics_v3.py"),
    ("scripts/acquire_pd_v2.py", "scripts/data/acquire_pd_v2.py"),
    ("scripts/adni_mri_candidates.py", "scripts/data/adni_mri_candidates.py"),
    ("scripts/download_real_pd_data.py", "scripts/data/download_real_pd_data.py"),
    ("scripts/generate_subject_splits.py", "scripts/data/generate_subject_splits.py"),
    ("scripts/validate_pd_open_local.py", "scripts/data/validate_pd_open_local.py"),
    ("scripts/tadpole_inspect.py", "scripts/data/tadpole_inspect.py"),
    ("scripts/run_v2_biomarker_real.py", "scripts/training/run_v2_biomarker_real.py"),
    ("scripts/train_biomarkers.py", "scripts/training/train_biomarkers.py"),
    ("scripts/local_prepare_and_train.ps1", "scripts/training/local_prepare_and_train.ps1"),
    ("scripts/evaluate_and_report_local.py", "scripts/evaluation/evaluate_and_report_local.py"),
    ("scripts/train_mri.py", "scripts/archived/invalid/train_mri.py"),
    ("scripts/train_ctclm.py", "scripts/archived/invalid/train_ctclm.py"),
    ("scripts/execute_v1_2_analysis.py", "scripts/archived/invalid/execute_v1_2_analysis_INVALID.py"),
    ("scripts/run_v1_pipeline.py", "experiments/archived/run_v1_pipeline.py"),
    ("scripts/run_v1_1_pipeline.py", "experiments/archived/run_v1_1_pipeline.py"),
    
    # Experiments
    ("experiments/architecture_smoke_test.py", "experiments/smoke/architecture_smoke_test.py"),
    ("experiments/ctclm_smoke_test.py", "experiments/smoke/ctclm_smoke_test.py"),
    ("experiments/qvbe_smoke_test.py", "experiments/smoke/qvbe_smoke_test.py"),
    ("experiments/synthetic_multimodal_smoke_test.py", "experiments/smoke/synthetic_multimodal_smoke_test.py"),
    ("experiments/run_experiment.py", "experiments/archived/run_experiment.py"),
    ("experiments/run_tadpole_benchmark.py", "experiments/archived/run_tadpole_benchmark.py"),
    
    # Configs
    ("configs/experiments/pd_open_final_v1_2.yaml", "configs/archived/pd_open_final_v1_2.yaml"),
    
    # Docs
    ("docs/architecture.md", "docs/architecture/architecture.md"),
    ("docs/quantum_design.md", "docs/architecture/quantum_design.md"),
    ("docs/data_contract.md", "docs/data/data_contract.md"),
    ("docs/experimental_protocol.md", "docs/experiments/experimental_protocol.md"),
    ("docs/evaluation_protocol.md", "docs/experiments/evaluation_protocol.md"),
    ("docs/research_questions.md", "docs/research/research_questions.md"),
    ("docs/temporal_prediction.md", "docs/research/temporal_prediction.md"),
    ("docs/limitations.md", "docs/research/limitations.md"),
    ("docs/architecture/engine_readiness_review.md", "docs/development/engine_readiness_review.md"),
]

for src, dst in repo_moves:
    s = os.path.join(REPO_DIR, src)
    d = os.path.join(REPO_DIR, dst)
    if os.path.exists(s):
        shutil.move(s, d)

# Docs Datasets and Experiments moves
# Avoid creating directories inside themselves
os.makedirs(os.path.join(REPO_DIR, "docs/data"), exist_ok=True)
for f in glob.glob(os.path.join(REPO_DIR, "docs/datasets/*.md")):
    if os.path.isfile(f):
        shutil.move(f, os.path.join(REPO_DIR, "docs/data", os.path.basename(f)))
        
os.makedirs(os.path.join(REPO_DIR, "docs/experiments"), exist_ok=True)
for f in glob.glob(os.path.join(REPO_DIR, "docs/experiments/*.md")):
    if os.path.isfile(f):
        # some of these might already be in place, or we are moving from a flat dir, wait, they were in docs/experiments
        pass # Actually, the structure was already docs/experiments/tadpole_baseline_v1.md. So no move needed.
        
# Move Reports
repo_reports = [
    ("reports/NADI_Q_CURRENT_STATE_FORENSIC_AUDIT.md", "reports/current/NADI_Q_CURRENT_STATE_FORENSIC_AUDIT.md"),
]
for src, dst in repo_reports:
    s = os.path.join(REPO_DIR, src)
    d = os.path.join(REPO_DIR, dst)
    if os.path.exists(s):
        shutil.move(s, d)

f_reports_src = os.path.join(DATA_DIR, "reports")
if os.path.exists(f_reports_src):
    for f in os.listdir(f_reports_src):
        fp = os.path.join(f_reports_src, f)
        if os.path.isfile(fp) and f.endswith(".md"):
            if f in ["PD_OPEN_BIOMARKER_REAL_EXPERIMENT_V2.md"]:
                dst = "reports/current/"
            elif f in ["PD_OPEN_POST_TRAINING_V1_2.md", "PD_OPEN_REAL_TRAINING_V1_1.md", "PD_OPEN_TRAINING_FINAL_REPORT.md"]:
                dst = "reports/invalid/"
            else:
                dst = "reports/historical/"
            shutil.move(fp, os.path.join(REPO_DIR, dst, f))

# Data Volume Checkpoints/Embeddings
data_moves = [
    ("checkpoints/biomarker_classical_best.pt", "checkpoints/active/biomarker_classical_best.pt"),
    ("checkpoints/biomarker_qvbe_best.pt", "checkpoints/active/biomarker_qvbe_best.pt"),
    ("checkpoints/pd_mri_bvae_v1", "checkpoints/invalid/pd_mri_bvae_v1"),
    ("checkpoints/pd_ctclm_v1", "checkpoints/invalid/pd_ctclm_v1"),
    ("embeddings/biomarker_qvbe_train.csv", "embeddings/active/biomarker_qvbe_train.csv"),
    ("embeddings/biomarker_qvbe_val.csv", "embeddings/active/biomarker_qvbe_val.csv"),
    ("embeddings/biomarker_qvbe_test.csv", "embeddings/active/biomarker_qvbe_test.csv"),
]

for src, dst in data_moves:
    s = os.path.join(DATA_DIR, src)
    d = os.path.join(DATA_DIR, dst)
    if os.path.exists(s):
        shutil.move(s, d)

# Clean Clutter
clutter = [
    "benchmark.log", "ctclm_test.log", "inspect.log", "qvbe_test.log", "test_baseline.log", "test_final.log",
    "inspect_report.json", "repo_files.txt", ".pytest_cache"
]
for c in clutter:
    p = os.path.join(REPO_DIR, c)
    if os.path.exists(p):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            try:
                os.remove(p)
            except Exception:
                pass

scratch_dir = os.path.join(REPO_DIR, "scratch")
if os.path.exists(scratch_dir):
    for f in os.listdir(scratch_dir):
        p = os.path.join(scratch_dir, f)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass

# Clean empty directories that we moved files out of
try:
    os.rmdir(os.path.join(REPO_DIR, "docs/datasets"))
except Exception:
    pass

try:
    os.rmdir(os.path.join(REPO_DIR, "docs/architecture")) 
    # Wait, architecture has new files. Don't remove.
except Exception:
    pass

print("Cleanup executed.")
