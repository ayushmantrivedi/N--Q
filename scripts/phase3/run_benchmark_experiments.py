import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import time
import tracemalloc
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import xgboost as xgb
import lightgbm as lgb
import pennylane as qml

# Import evaluation metrics from benchmark_engine
from benchmark_engine import evaluate_predictions, compute_effective_rank

DATA_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\ppmi_primary_benchmark_cohort_n1465.parquet"
MANIFEST_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\PARTICIPANT_SPLIT_MANIFEST.csv"
PARTITION_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\feature_partitions.json"
OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3"

print("Loading dataset, split manifest, and feature schema...")
df = pd.read_parquet(DATA_PATH)
manifest_df = pd.read_csv(MANIFEST_PATH)
with open(PARTITION_PATH, 'r') as f:
    partitions = json.load(f)

core_features = partitions['core_features_249']
target_col = partitions['target']
pats_all = df['PATNO'].values
y_all = df[target_col].values
X_all = df[core_features].values

print(f"Data shape: X = {X_all.shape}, y = {y_all.shape}")
print(f"Positive: {(y_all == 1).sum()}, Negative: {(y_all == 0).sum()}")

# Initialize PennyLane circuits
dev_8 = qml.device("default.qubit", wires=8)
@qml.qnode(dev_8)
def circ_8(x):
    for i in range(8):
        qml.RY(x[i], wires=i)
    for i in range(8):
        qml.CNOT(wires=[i, (i + 1) % 8])
    for i in range(8):
        qml.RZ(x[i], wires=i)
    for i in range(7):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(8)] + [qml.expval(qml.PauliX(i)) for i in range(8)]

dev_12 = qml.device("default.qubit", wires=12)
@qml.qnode(dev_12)
def circ_12(x):
    for i in range(12):
        qml.RY(x[i], wires=i)
    for i in range(12):
        qml.CNOT(wires=[i, (i + 1) % 12])
    for i in range(12):
        qml.RZ(x[i], wires=i)
    for i in range(11):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(12)] + [qml.expval(qml.PauliX(i)) for i in range(12)]

def extract_pqk(circuit, X_scaled, n_dim):
    n_samples = len(X_scaled)
    q_feats = np.zeros((n_samples, 2 * n_dim), dtype=np.float64)
    for i in range(n_samples):
        q_feats[i] = circuit(X_scaled[i])
    return q_feats

seeds = [42, 101, 2024, 777, 999]
n_splits = 5

classical_results = []
quantum_results = []
representation_results = []
kernel_diagnostics_records = []
oof_predictions = []

total_folds = len(seeds) * n_splits
fold_counter = 0

start_total_time = time.time()

print(f"\n================ STARTING BENCHMARK ACROSS {total_folds} FOLDS ================\n")

for seed in seeds:
    for fold in range(n_splits):
        fold_counter += 1
        t_fold_start = time.time()
        print(f"--- [Fold {fold_counter}/{total_folds}] Seed: {seed}, Fold: {fold} ---")
        
        # Test participants for this fold
        test_pats = set(manifest_df[(manifest_df['seed'] == seed) & (manifest_df['fold'] == fold)]['PATNO'])
        train_mask = np.array([p not in test_pats for p in pats_all])
        test_mask = np.array([p in test_pats for p in pats_all])
        
        X_train_raw = X_all[train_mask].copy()
        y_train = y_all[train_mask].copy()
        X_test_raw = X_all[test_mask].copy()
        y_test = y_all[test_mask].copy()
        test_pats_fold = pats_all[test_mask]
        
        # Fit imputer and scaler ONLY on training data
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # ---------------- TASK 4: REPRESENTATION EXPERIMENTS ---------------- #
        # Fit PCA(8) on train
        pca_8 = PCA(n_components=8, random_state=seed)
        X_train_pca8 = pca_8.fit_transform(X_train)
        X_test_pca8 = pca_8.transform(X_test)
        var_ratio_8 = float(np.sum(pca_8.explained_variance_ratio_))
        
        # Fit PCA(12) on train
        pca_12 = PCA(n_components=12, random_state=seed)
        X_train_pca12 = pca_12.fit_transform(X_train)
        X_test_pca12 = pca_12.transform(X_test)
        var_ratio_12 = float(np.sum(pca_12.explained_variance_ratio_))
        
        # Measure effective rank of representations
        cov_raw = np.cov(X_train, rowvar=False)
        eff_rank_raw = compute_effective_rank(cov_raw)
        cov_pca8 = np.cov(X_train_pca8, rowvar=False)
        eff_rank_pca8 = compute_effective_rank(cov_pca8)
        cov_pca12 = np.cov(X_train_pca12, rowvar=False)
        eff_rank_pca12 = compute_effective_rank(cov_pca12)
        
        representation_results.append({
            "seed": seed,
            "fold": fold,
            "var_retained_pca8": var_ratio_8,
            "var_retained_pca12": var_ratio_12,
            "eff_rank_raw": eff_rank_raw,
            "eff_rank_pca8": eff_rank_pca8,
            "eff_rank_pca12": eff_rank_pca12
        })
        
        # ---------------- TASK 3: CLASSICAL BENCHMARK MODELS ---------------- #
        classical_models = {
            "Logistic_Regression_L1": LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000, random_state=seed),
            "Logistic_Regression_L2": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=1000, random_state=seed),
            "Random_Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=seed, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.7503, random_state=seed, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.7503, random_state=seed, verbose=-1),
            "Classical_RBF_Raw": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
            "Classical_RBF_PCA8": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
            "Classical_RBF_PCA12": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed)
        }
        
        for name, model in classical_models.items():
            tracemalloc.start()
            t0 = time.time()
            
            # Select input representation
            if name == "Classical_RBF_PCA8":
                model.fit(X_train_pca8, y_train)
                y_prob = model.predict_proba(X_test_pca8)[:, 1]
            elif name == "Classical_RBF_PCA12":
                model.fit(X_train_pca12, y_train)
                y_prob = model.predict_proba(X_test_pca12)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                
            t1 = time.time()
            curr_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            metrics = evaluate_predictions(y_test, y_prob)
            metrics.update({
                "model_name": name,
                "seed": seed,
                "fold": fold,
                "train_time_sec": t1 - t0,
                "peak_memory_kb": peak_mem / 1024
            })
            classical_results.append(metrics)
            
            # Store OOF predictions
            for p, yt, yp in zip(test_pats_fold, y_test, y_prob):
                oof_predictions.append({
                    "PATNO": p,
                    "seed": seed,
                    "fold": fold,
                    "model_family": "classical",
                    "model_name": name,
                    "y_true": int(yt),
                    "y_prob": float(yp)
                })

        # ---------------- TASK 5: QUANTUM PROJECTED KERNEL BENCHMARK ---------------- #
        # MinMax scale PCA features to [0, pi]
        minmax_8 = MinMaxScaler(feature_range=(0, np.pi))
        X_train_q8_in = minmax_8.fit_transform(X_train_pca8)
        X_test_q8_in  = minmax_8.transform(X_test_pca8)
        
        minmax_12 = MinMaxScaler(feature_range=(0, np.pi))
        X_train_q12_in = minmax_12.fit_transform(X_train_pca12)
        X_test_q12_in  = minmax_12.transform(X_test_pca12)
        
        quantum_configs = [
            ("PQK_8_Qubit", circ_8, X_train_q8_in, X_test_q8_in, 8),
            ("PQK_12_Qubit", circ_12, X_train_q12_in, X_test_q12_in, 12)
        ]
        
        for q_name, q_circ, X_tr_q, X_te_q, n_wires in quantum_configs:
            tracemalloc.start()
            t_q0 = time.time()
            
            # 1. Extract quantum expectations:
            q_train_feats = extract_pqk(q_circ, X_tr_q, n_wires)
            q_test_feats  = extract_pqk(q_circ, X_te_q, n_wires)
            t_extract = time.time()
            
            # 2. Construct Projected Quantum Kernel:
            # gamma = 1.0 / feature_dimension
            gamma = 1.0 / (2 * n_wires)
            K_train = rbf_kernel(q_train_feats, q_train_feats, gamma=gamma)
            K_test  = rbf_kernel(q_test_feats, q_train_feats, gamma=gamma)
            t_kernel = time.time()
            
            # 3. Kernel diagnostics
            evals = np.linalg.eigvalsh(K_train)
            evals_pos = evals[evals > 1e-12]
            cond_num = float(np.max(evals_pos) / np.min(evals_pos)) if len(evals_pos) > 0 else np.nan
            k_eff_rank = compute_effective_rank(K_train)
            
            # Off-diagonal stats
            n_tr = len(K_train)
            off_diag_indices = ~np.eye(n_tr, dtype=bool)
            off_diag_vals = K_train[off_diag_indices]
            k_mean = float(np.mean(off_diag_vals))
            k_var  = float(np.var(off_diag_vals))
            
            kernel_diagnostics_records.append({
                "model_name": q_name,
                "seed": seed,
                "fold": fold,
                "effective_rank": k_eff_rank,
                "condition_number": cond_num,
                "off_diagonal_mean": k_mean,
                "off_diagonal_variance": k_var,
                "min_eigenvalue": float(np.min(evals)),
                "max_eigenvalue": float(np.max(evals)),
                "top_5_eigenvalues": [float(e) for e in evals[-5:]]
            })
            
            # 4. Train precomputed kernel SVM
            clf_q = SVC(kernel='precomputed', probability=True, C=1.0, random_state=seed)
            clf_q.fit(K_train, y_train)
            y_prob_q = clf_q.predict_proba(K_test)[:, 1]
            t_q_end = time.time()
            
            curr_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            q_metrics = evaluate_predictions(y_test, y_prob_q)
            q_metrics.update({
                "model_name": q_name,
                "seed": seed,
                "fold": fold,
                "circuit_evaluations": len(X_tr_q) + len(X_te_q),
                "feature_extraction_sec": t_extract - t_q0,
                "kernel_construction_sec": t_kernel - t_extract,
                "svm_train_sec": t_q_end - t_kernel,
                "total_quantum_time_sec": t_q_end - t_q0,
                "peak_memory_kb": peak_mem / 1024,
                "effective_rank": k_eff_rank,
                "condition_number": cond_num,
                "kernel_variance": k_var
            })
            quantum_results.append(q_metrics)
            
            # Store OOF predictions
            for p, yt, yp in zip(test_pats_fold, y_test, y_prob_q):
                oof_predictions.append({
                    "PATNO": p,
                    "seed": seed,
                    "fold": fold,
                    "model_family": "quantum",
                    "model_name": q_name,
                    "y_true": int(yt),
                    "y_prob": float(yp)
                })

        print(f"  Fold {fold_counter} complete in {time.time() - t_fold_start:.1f}s | "
              f"RF AUROC: {classical_results[-6]['auroc']:.4f} | "
              f"XGB AUROC: {classical_results[-5]['auroc']:.4f} | "
              f"PQK-8 AUROC: {quantum_results[-2]['auroc']:.4f} | "
              f"PQK-12 AUROC: {quantum_results[-1]['auroc']:.4f}")

# Convert results to DataFrames
classical_df = pd.DataFrame(classical_results)
quantum_df   = pd.DataFrame(quantum_results)
rep_df       = pd.DataFrame(representation_results)
kernel_diag_df = pd.DataFrame(kernel_diagnostics_records)
oof_df       = pd.DataFrame(oof_predictions)

# Save intermediate raw tables
classical_df.to_csv(os.path.join(OUTPUT_DIR, "CLASSICAL_BENCHMARK_RESULTS.csv"), index=False)
quantum_df.to_csv(os.path.join(OUTPUT_DIR, "QUANTUM_BENCHMARK_RESULTS.csv"), index=False)
oof_df.to_csv(os.path.join(OUTPUT_DIR, "OUT_OF_FOLD_PREDICTIONS.csv"), index=False)
rep_df.to_csv(os.path.join(OUTPUT_DIR, "REPRESENTATION_STABILITY_METRICS.csv"), index=False)
kernel_diag_df.to_csv(os.path.join(OUTPUT_DIR, "KERNEL_DIAGNOSTICS_DATA.csv"), index=False)

total_elapsed = time.time() - start_total_time
print(f"\n================ BENCHMARK FINISHED IN {total_elapsed/60:.2f} MINUTES ================\n")

# Print Summary Table
print("=== AGGREGATE SUMMARY METRICS (MEAN ± STD ACROSS 25 FOLDS) ===")
summary_rows = []
for name in classical_df['model_name'].unique():
    sub = classical_df[classical_df['model_name'] == name]
    summary_rows.append({
        "Model": name,
        "AUROC": f"{sub['auroc'].mean():.4f} ± {sub['auroc'].std():.4f}",
        "AUPRC": f"{sub['auprc'].mean():.4f} ± {sub['auprc'].std():.4f}",
        "Bal Acc": f"{sub['balanced_acc'].mean():.4f} ± {sub['balanced_acc'].std():.4f}",
        "Brier": f"{sub['brier'].mean():.4f} ± {sub['brier'].std():.4f}",
        "ECE": f"{sub['ece'].mean():.4f} ± {sub['ece'].std():.4f}",
        "Avg Runtime (s)": f"{sub['train_time_sec'].mean():.3f}s"
    })

for name in quantum_df['model_name'].unique():
    sub = quantum_df[quantum_df['model_name'] == name]
    summary_rows.append({
        "Model": name,
        "AUROC": f"{sub['auroc'].mean():.4f} ± {sub['auroc'].std():.4f}",
        "AUPRC": f"{sub['auprc'].mean():.4f} ± {sub['auprc'].std():.4f}",
        "Bal Acc": f"{sub['balanced_acc'].mean():.4f} ± {sub['balanced_acc'].std():.4f}",
        "Brier": f"{sub['brier'].mean():.4f} ± {sub['brier'].std():.4f}",
        "ECE": f"{sub['ece'].mean():.4f} ± {sub['ece'].std():.4f}",
        "Avg Runtime (s)": f"{sub['total_quantum_time_sec'].mean():.3f}s"
    })

sum_df = pd.DataFrame(summary_rows)
print(sum_df.to_string(index=False))

