import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss

from benchmark_engine import evaluate_predictions

DATA_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\ppmi_primary_benchmark_cohort_n1465.parquet"
MANIFEST_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\PARTICIPANT_SPLIT_MANIFEST.csv"
PARTITION_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\feature_partitions.json"
OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3"

def run_ablations():
    print("Loading data for ablation study...")
    df = pd.read_parquet(DATA_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(PARTITION_PATH, 'r') as f:
        partitions = json.load(f)

    mri_feats   = partitions['mri_features']
    motor_feats = partitions['motor_features']
    gen_feats   = partitions['genetics_features'] + partitions['prs_features']
    demo_feats  = partitions['demographics_features']

    ablation_suites = {
        "Morphometry_Only": mri_feats,
        "Clinical_Motor_Only": motor_feats,
        "Genetics_PRS_Only": gen_feats,
        "Morphometry_plus_Motor": mri_feats + motor_feats,
        "Morphometry_Motor_Genetics": mri_feats + motor_feats + gen_feats, # 249 core
        "Full_Approved_Suite": mri_feats + motor_feats + gen_feats + demo_feats # 252 full
    }

    target_col = partitions['target']
    pats_all = df['PATNO'].values
    y_all = df[target_col].values

    # Run on all 25 folds (5 seeds x 5 folds)
    seeds = [42, 101, 2024, 777, 999]
    n_splits = 5

    ablation_results = []
    
    print("\nStarting Modality Ablation Experiments...")
    for suite_name, feat_cols in ablation_suites.items():
        print(f"\n--- Evaluating Suite: {suite_name} (D = {len(feat_cols)} features) ---")
        X_suite = df[feat_cols].values
        
        t0_suite = time.time()
        for seed in seeds:
            for fold in range(n_splits):
                test_pats = set(manifest_df[(manifest_df['seed'] == seed) & (manifest_df['fold'] == fold)]['PATNO'])
                train_mask = np.array([p not in test_pats for p in pats_all])
                test_mask = np.array([p in test_pats for p in pats_all])
                
                X_tr = X_suite[train_mask]
                y_tr = y_all[train_mask]
                X_te = X_suite[test_mask]
                y_te = y_all[test_mask]
                
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_te = imputer.transform(X_te)
                
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                
                # Evaluate two top representative classical models:
                # 1. Random Forest (non-linear ensemble)
                # 2. LightGBM (gradient boosted trees)
                # 3. Logistic Regression L2 (linear baseline)
                models = {
                    "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=0.7503, random_state=seed, verbose=-1),
                    "Random_Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=seed, n_jobs=-1),
                    "Logistic_Regression_L2": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=1000, random_state=seed)
                }
                
                for m_name, model in models.items():
                    model.fit(X_tr, y_tr)
                    y_prob = model.predict_proba(X_te)[:, 1]
                    metrics = evaluate_predictions(y_te, y_prob)
                    metrics.update({
                        "ablation_suite": suite_name,
                        "feature_count": len(feat_cols),
                        "model_name": m_name,
                        "seed": seed,
                        "fold": fold
                    })
                    ablation_results.append(metrics)
                    
        print(f"Suite {suite_name} finished in {time.time() - t0_suite:.1f}s")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ABLATION_RESULTS.csv"), index=False)
    print("\nSaved ABLATION_RESULTS.csv!")

    # Summary table
    print("\n=== ABLATION SUMMARY (AUROC Mean ± Std Across 25 Folds) ===")
    summary = ablation_df.groupby(['ablation_suite', 'model_name'])['auroc'].agg(['mean', 'std']).reset_index()
    print(summary.to_string(index=False))

if __name__ == '__main__':
    run_ablations()

