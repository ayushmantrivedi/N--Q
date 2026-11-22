import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3"

oof_df = pd.read_csv(os.path.join(OUTPUT_DIR, "OUT_OF_FOLD_PREDICTIONS.csv"))
kernel_diag_df = pd.read_csv(os.path.join(OUTPUT_DIR, "KERNEL_DIAGNOSTICS_DATA.csv"))
rep_df = pd.read_csv(os.path.join(OUTPUT_DIR, "REPRESENTATION_STABILITY_METRICS.csv"))

print(f"Loaded OOF predictions: {len(oof_df)} rows")
models = oof_df['model_name'].unique()
print(f"Unique models: {models}")

# ----------------- 1. PARTICIPANT-LEVEL POOLED BOOTSTRAP ANALYSIS ----------------- #
# For each participant, average predictions across the 5 seeds to get a single unified out-of-fold probability
# This adheres strictly to: "Confidence intervals using an appropriate participant-level procedure. Do not treat the 25 correlated fold scores as 25 independent experiments."

pt_oof = oof_df.groupby(['PATNO', 'model_name']).agg({
    'y_true': 'first',
    'y_prob': 'mean'
}).reset_index()

print(f"Participant-level OOF shape per model: {len(pt_oof[pt_oof['model_name'] == models[0]])}")

n_boot = 1000
rng = np.random.RandomState(42)

bootstrap_stats = []

pats_unique = pt_oof['PATNO'].unique()
n_pts = len(pats_unique)

print(f"\nRunning {n_boot} participant-level bootstrap resamples...")

# Pre-extract vectors
model_preds = {}
y_true_all = None
for m in models:
    sub = pt_oof[pt_oof['model_name'] == m].sort_values('PATNO')
    model_preds[m] = sub['y_prob'].values
    if y_true_all is None:
        y_true_all = sub['y_true'].values

# Pre-generate bootstrap index arrays
boot_indices = [rng.choice(n_pts, size=n_pts, replace=True) for _ in range(n_boot)]

boot_aurocs = {m: [] for m in models}
boot_auprcs = {m: [] for m in models}
boot_balaccs = {m: [] for m in models}

for idxs in boot_indices:
    y_b = y_true_all[idxs]
    # Check if both classes present
    if len(np.unique(y_b)) < 2:
        continue
    for m in models:
        prob_b = model_preds[m][idxs]
        boot_aurocs[m].append(roc_auc_score(y_b, prob_b))
        boot_auprcs[m].append(average_precision_score(y_b, prob_b))
        boot_balaccs[m].append(balanced_accuracy_score(y_b, (prob_b >= 0.5).astype(int)))

# Compile table
pt_summary = []
for m in models:
    auroc_vals = np.array(boot_aurocs[m])
    auprc_vals = np.array(boot_auprcs[m])
    balacc_vals = np.array(boot_balaccs[m])
    
    pt_summary.append({
        "Model": m,
        "AUROC_Point": roc_auc_score(y_true_all, model_preds[m]),
        "AUROC_95CI": f"[{np.percentile(auroc_vals, 2.5):.4f}, {np.percentile(auroc_vals, 97.5):.4f}]",
        "AUPRC_Point": average_precision_score(y_true_all, model_preds[m]),
        "AUPRC_95CI": f"[{np.percentile(auprc_vals, 2.5):.4f}, {np.percentile(auprc_vals, 97.5):.4f}]",
        "BalAcc_Point": balanced_accuracy_score(y_true_all, (model_preds[m] >= 0.5).astype(int)),
        "BalAcc_95CI": f"[{np.percentile(balacc_vals, 2.5):.4f}, {np.percentile(balacc_vals, 97.5):.4f}]"
    })

pt_summary_df = pd.DataFrame(pt_summary)
print("\n=== PARTICIPANT-LEVEL POOLED METRICS (N=1,465) WITH 95% BOOTSTRAP CIs ===")
print(pt_summary_df.to_string(index=False))

# ----------------- 2. PAIRED STATISTICAL DIFFERENCE: QUANTUM VS CLASSICAL ----------------- #
print("\n=== PAIRED STATISTICAL DIFFERENCES (QUANTUM vs CLASSICAL) ===")
comparisons = [
    ("PQK_8_Qubit", "Classical_RBF_PCA8", "Matched 8-dim PCA baseline"),
    ("PQK_12_Qubit", "Classical_RBF_PCA12", "Matched 12-dim PCA baseline"),
    ("PQK_8_Qubit", "Logistic_Regression_L1", "Classical linear baseline"),
    ("PQK_8_Qubit", "Random_Forest", "Classical ensemble baseline"),
    ("PQK_8_Qubit", "XGBoost", "Classical gradient boosted tree"),
    ("PQK_8_Qubit", "LightGBM", "Classical gradient boosted tree"),
    ("PQK_12_Qubit", "LightGBM", "Classical gradient boosted tree"),
    ("PQK_12_Qubit", "Classical_RBF_Raw", "Full raw classical RBF")
]

comp_rows = []
for q_mod, c_mod, desc in comparisons:
    diff_aurocs = np.array(boot_aurocs[q_mod]) - np.array(boot_aurocs[c_mod])
    mean_diff = np.mean(diff_aurocs)
    ci_low = np.percentile(diff_aurocs, 2.5)
    ci_high = np.percentile(diff_aurocs, 97.5)
    # Two-sided empirical p-value
    p_val = 2 * min(np.mean(diff_aurocs >= 0), np.mean(diff_aurocs <= 0))
    p_val = max(p_val, 1.0 / n_boot)
    
    comp_rows.append({
        "Quantum Model": q_mod,
        "Classical Benchmark": c_mod,
        "Description": desc,
        "Delta_AUROC": f"{mean_diff:+.4f}",
        "95% CI of Delta": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
        "p-value": f"{p_val:.4f}" if p_val >= 0.001 else "< 0.001",
        "Significant Difference?": "YES (Classical Superior)" if ci_high < 0 else "No"
    })

comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False))

# ----------------- 3. KERNEL DIAGNOSTICS & CONCENTRATION AUDIT ----------------- #
print("\n=== KERNEL DIAGNOSTICS SUMMARY ===")
kd_summary = kernel_diag_df.groupby('model_name').agg({
    'effective_rank': ['mean', 'std'],
    'condition_number': ['mean', 'std'],
    'off_diagonal_mean': ['mean', 'std'],
    'off_diagonal_variance': ['mean', 'std']
}).reset_index()
print(kd_summary.to_string())

# ----------------- 4. REPRESENTATION STABILITY & VARIANCE RETENTION ----------------- #
print("\n=== REPRESENTATION STABILITY (ACROSS 25 TRAINING FOLDS) ===")
rep_summary = rep_df.agg({
    'var_retained_pca8': ['mean', 'std', 'min', 'max'],
    'var_retained_pca12': ['mean', 'std', 'min', 'max'],
    'eff_rank_raw': ['mean', 'std'],
    'eff_rank_pca8': ['mean', 'std'],
    'eff_rank_pca12': ['mean', 'std']
})
print(rep_summary)

# Save all statistical outputs
pt_summary_df.to_csv(os.path.join(OUTPUT_DIR, "PARTICIPANT_LEVEL_BENCHMARK_STATS.csv"), index=False)
comp_df.to_csv(os.path.join(OUTPUT_DIR, "STATISTICAL_COMPARISON_PAIRED_TESTS.csv"), index=False)

