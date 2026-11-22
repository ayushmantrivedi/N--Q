import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA\ppmi_primary_benchmark_cohort_n1465.parquet"
OUTPUT_MANIFEST = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\PARTICIPANT_SPLIT_MANIFEST.csv"

df = pd.read_parquet(DATA_PATH)
pats = df['PATNO'].values
y = df['target_saa_binary'].values

seeds = [42, 101, 2024, 777, 999]
n_splits = 5

manifest_rows = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(pats, y)):
        test_pats = pats[test_idx]
        test_y = y[test_idx]
        for p, target in zip(test_pats, test_y):
            manifest_rows.append({
                "PATNO": p,
                "seed": seed,
                "fold": fold,
                "split_role": "test",
                "target_saa_binary": target
            })

manifest_df = pd.DataFrame(manifest_rows)
print(f"Total split manifest rows: {len(manifest_df)} (expected 1465 * 5 = {1465 * 5})")
assert len(manifest_df) == 1465 * 5

# Check per seed: every participant appears exactly once
for seed in seeds:
    seed_sub = manifest_df[manifest_df['seed'] == seed]
    assert seed_sub['PATNO'].nunique() == 1465
    assert len(seed_sub) == 1465
    # Check fold distribution and class balance
    for fold in range(n_splits):
        fold_sub = seed_sub[seed_sub['fold'] == fold]
        pos_ratio = fold_sub['target_saa_binary'].mean()
        print(f"Seed {seed}, Fold {fold}: N = {len(fold_sub)}, Pos ratio = {pos_ratio:.4f}")

manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
print(f"Saved split manifest to: {OUTPUT_MANIFEST}")

