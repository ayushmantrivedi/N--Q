import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
CACHE_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"
OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_3\DATA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== STEP 1: LOAD COHORT DEFINITION ===")
cur_df = pd.read_parquet(CACHE_PATH)
cur_bl = cur_df[cur_df['EVENT_ID'] == 'BL'].copy()

# FreeSurfer 3-modality participants
fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", low_memory=False)
fs_sa  = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_SA_18Sep2026.csv", low_memory=False)
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", low_memory=False)

fs_1713 = set(fs_cth['PATNO']).intersection(set(fs_sa['PATNO'])).intersection(set(fs_vol['PATNO']))
print(f"FreeSurfer complete 3-modality participants: {len(fs_1713)}")

# Benchmark cohort filter
benchmark_bl = cur_bl[
    cur_bl['PATNO'].isin(fs_1713) &
    cur_bl['CSFSAA'].isin([0.0, 1.0, 3.0]) &
    cur_bl['updrs3_score'].notna()
].copy().sort_values('PATNO').reset_index(drop=True)

print(f"Primary Benchmark Cohort Size: {len(benchmark_bl)}")
pats = list(benchmark_bl['PATNO'])

# Target variable definition
benchmark_bl['target_saa_binary'] = benchmark_bl['CSFSAA'].map({0.0: 0, 1.0: 1, 3.0: 1}).astype(int)
pos_count = (benchmark_bl['target_saa_binary'] == 1).sum()
neg_count = (benchmark_bl['target_saa_binary'] == 0).sum()
print(f"Target distribution: Positive = {pos_count}, Negative = {neg_count}")
assert len(benchmark_bl) == 1465, f"Expected 1465 participants, got {len(benchmark_bl)}"
assert pos_count == 837, f"Expected 837 positives, got {pos_count}"
assert neg_count == 628, f"Expected 628 negatives, got {neg_count}"

print("\n=== STEP 2: ASSEMBLE FREESURFER 7.1 IDPs ===")
cth_cols = [c for c in fs_cth.columns if c not in ['PATNO', 'EVENT_ID']]
sa_cols  = [c for c in fs_sa.columns  if c not in ['PATNO', 'EVENT_ID']]
vol_cols = [c for c in fs_vol.columns if c not in ['PATNO', 'EVENT_ID']]

print(f"CTH columns: {len(cth_cols)}")
print(f"SA columns:  {len(sa_cols)}")
print(f"ASEG columns: {len(vol_cols)}")
print(f"Total FreeSurfer IDPs: {len(cth_cols) + len(sa_cols) + len(vol_cols)}")

# Filter to cohort
fs_cth_sub = fs_cth[fs_cth['PATNO'].isin(pats)].copy().sort_values('PATNO').reset_index(drop=True)
fs_sa_sub  = fs_sa[fs_sa['PATNO'].isin(pats)].copy().sort_values('PATNO').reset_index(drop=True)
fs_vol_sub = fs_vol[fs_vol['PATNO'].isin(pats)].copy().sort_values('PATNO').reset_index(drop=True)

# Prefix column names to guarantee unique, traceable naming
# CTH: cth_{roi}
# SA:  sa_{roi}
# VOL: aseg_{roi}
rename_cth = {c: f"cth_{c}" for c in cth_cols}
rename_sa  = {c: f"sa_{c}"  for c in sa_cols}
rename_vol = {c: f"aseg_{c}" for c in vol_cols}

fs_cth_renamed = fs_cth_sub[['PATNO'] + cth_cols].rename(columns=rename_cth)
fs_sa_renamed  = fs_sa_sub[['PATNO'] + sa_cols].rename(columns=rename_sa)
fs_vol_renamed = fs_vol_sub[['PATNO'] + vol_cols].rename(columns=rename_vol)

mri_merged = fs_cth_renamed.merge(fs_sa_renamed, on='PATNO').merge(fs_vol_renamed, on='PATNO')
print(f"Merged MRI shape: {mri_merged.shape}")
assert len(mri_merged) == 1465

print("\n=== STEP 3: ASSEMBLE CLINICAL MOTOR ITEMS (MDS-UPDRS PART III) ===")
updrs = pd.read_csv(f"{DATA_DIR}\\Motor___MDS-UPDRS\\MDS-UPDRS_Part_III_18Sep2026.csv", low_memory=False)
updrs_bl = updrs[(updrs['EVENT_ID'] == 'BL') & (updrs['PATNO'].isin(pats))].copy()

def pick_motor_row(group):
    if len(group) == 1:
        return group.iloc[0]
    off = group[group['PDSTATE'] == 'OFF']
    if len(off) > 0:
        return off.iloc[0]
    un = group[group['PDTRTMNT'] == 0]
    if len(un) > 0:
        return un.iloc[0]
    return group.iloc[0]

updrs_dedup = updrs_bl.groupby('PATNO', as_index=False).apply(pick_motor_row, include_groups=False)

motor_items = [
    'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
    'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
    'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT',
    'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML',
    'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL',
    'NP3RTALJ', 'NP3RTCON'
]

# Calculate domain subscores
# 1. Tremor subscore (10 items)
tremor_items = ['NP3PTRMR', 'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', 'NP3RTCON']
# 2. Rigidity subscore (5 items)
rigidity_items = ['NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL']
# 3. Bradykinesia subscore (11 items)
bradykinesia_items = ['NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3BRADY']
# 4. Axial / Gait subscore (7 items)
axial_gait_items = ['NP3SPCH', 'NP3FACXP', 'NP3RISNG', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR']

updrs_dedup['subscore_tremor'] = updrs_dedup[tremor_items].sum(axis=1)
updrs_dedup['subscore_rigidity'] = updrs_dedup[rigidity_items].sum(axis=1)
updrs_dedup['subscore_bradykinesia'] = updrs_dedup[bradykinesia_items].sum(axis=1)
updrs_dedup['subscore_axial_gait'] = updrs_dedup[axial_gait_items].sum(axis=1)

motor_cols_all = motor_items + ['subscore_tremor', 'subscore_rigidity', 'subscore_bradykinesia', 'subscore_axial_gait']
motor_df = updrs_dedup[['PATNO'] + motor_cols_all].copy()

# Note: 2 patients in benchmark_bl (out of 1,465) did not have a matching BL row in raw UPDRS III, but had updrs3_score in curated cut.
# Merge with benchmark_bl to ensure 1,465 rows, and impute those 2 from cohort median
motor_merged = benchmark_bl[['PATNO', 'updrs3_score']].merge(motor_df, on='PATNO', how='left')
print(f"Motor merged shape: {motor_merged.shape}")
n_missing_motor = motor_merged[motor_cols_all].isna().sum().max()
print(f"Missing rows in item-level motor: {n_missing_motor} (imputing missing 2 from median)")
for c in motor_cols_all:
    if motor_merged[c].isna().sum() > 0:
        motor_merged[c] = motor_merged[c].fillna(motor_merged[c].median())

# Total motor features: 33 items + 4 subscores + 1 total score = 38 features
motor_feature_names = ['updrs3_score'] + motor_cols_all
print(f"Total motor features: {len(motor_feature_names)}")

print("\n=== STEP 4: ASSEMBLE GENETICS & PRS ===")
gen_full = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", low_memory=False)
prs_full = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", low_memory=False)

gen_sub = benchmark_bl[['PATNO']].merge(gen_full, on='PATNO', how='left')

# LRRK2 carrier
gen_sub['is_LRRK2_carrier'] = gen_sub['LRRK2'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() in ['0', '0.0', ''] else 1)
# GBA carrier
gen_sub['is_GBA_carrier'] = gen_sub['GBA'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() in ['0', '0.0', ''] else 1)
# APOE e4 carrier
gen_sub['is_APOE_e4_carrier'] = gen_sub['APOE'].apply(lambda x: 1 if ('4' in str(x)) else 0)
# Pathogenic variant count
gen_sub['PATHVAR_COUNT'] = pd.to_numeric(gen_sub['PATHVAR_COUNT'], errors='coerce').fillna(0).astype(int)

gen_cols = ['is_LRRK2_carrier', 'is_GBA_carrier', 'is_APOE_e4_carrier', 'PATHVAR_COUNT']
gen_features = gen_sub[['PATNO'] + gen_cols]
print("Genetics features ready:", gen_cols)

# PRS
prs_cols = ['GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS']
prs_sub = benchmark_bl[['PATNO']].merge(prs_full[['PATNO'] + prs_cols], on='PATNO', how='left')
print("PRS features ready:", prs_cols)

print("\n=== STEP 5: ASSEMBLE DEMOGRAPHICS (OPTIONAL ABLATION GROUP) ===")
demo_cols = ['age', 'SEX', 'EDUCYRS']
demo_sub = benchmark_bl[['PATNO'] + demo_cols].copy()
# Encode SEX: 1=Male, 2=Female -> 0=Male, 1=Female
demo_sub['SEX_binary'] = demo_sub['SEX'].map({1: 0, 2: 1}).fillna(0).astype(int)
demo_sub['age_at_baseline'] = demo_sub['age']
demo_sub['education_years'] = demo_sub['EDUCYRS']
demo_features = demo_sub[['PATNO', 'age_at_baseline', 'SEX_binary', 'education_years']]
demo_feature_names = ['age_at_baseline', 'SEX_binary', 'education_years']
print("Demographics features ready:", demo_feature_names)

print("\n=== STEP 6: COMPILE MASTER FEATURE MATRIX ===")
master_df = benchmark_bl[['PATNO', 'target_saa_binary']].copy()
master_df = master_df.merge(mri_merged, on='PATNO')
master_df = master_df.merge(motor_merged[['PATNO'] + motor_feature_names], on='PATNO')
master_df = master_df.merge(gen_features, on='PATNO')
master_df = master_df.merge(prs_sub, on='PATNO')
master_df = master_df.merge(demo_features, on='PATNO')

print(f"Master DataFrame Shape: {master_df.shape}")
print(f"Participants count: {len(master_df)}")

# Verify feature counts
mri_cols_all = [c for c in master_df.columns if c.startswith('cth_') or c.startswith('sa_') or c.startswith('aseg_')]
print(f"MRI feature count: {len(mri_cols_all)} (expected 204)")
print(f"Motor feature count: {len(motor_feature_names)} (expected 38)")
print(f"Genetics feature count: {len(gen_cols)} (expected 4)")
print(f"PRS feature count: {len(prs_cols)} (expected 3)")
core_features = mri_cols_all + motor_feature_names + gen_cols + prs_cols
print(f"Core Multimodal Features (no demographics): {len(core_features)} (expected 249)")
full_features = core_features + demo_feature_names
print(f"Full Multimodal Features (with demographics): {len(full_features)} (expected 252)")

print("\n=== STEP 7: LEAKAGE QUARANTINE AUDIT ===")
leakage_reg = pd.read_csv(r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\LEAKAGE_EXCLUSION_REGISTER.csv")
banned_vars = set(leakage_reg['variable_name'])
print(f"Total banned leakage variables: {len(banned_vars)}")
present_banned = banned_vars.intersection(set(master_df.columns))
print(f"Banned variables present in master_df: {present_banned}")
assert len(present_banned) == 0, f"LEAKAGE DETECTED! Banned variables found: {present_banned}"

# Check for any biofluid variables (distinguishing FreeSurfer structural MRI ventricular volume 'aseg_CSF' in mm^3)
biofluid_vars = [c for c in master_df.columns if (not c.startswith('aseg_')) and any(k in c.lower() for k in ['csf', 'syn', 'tau', 'abeta', 'fmax', 'biomarker', 'protein'])]
print(f"Biofluid variables in feature matrix: {biofluid_vars}")
assert len(biofluid_vars) == 0, f"Biofluid variables found in feature set: {biofluid_vars}"

print("\n=== STEP 8: SAVE BENCHMARK DATASET ===")
parquet_out = os.path.join(OUTPUT_DIR, "ppmi_primary_benchmark_cohort_n1465.parquet")
csv_out     = os.path.join(OUTPUT_DIR, "ppmi_primary_benchmark_cohort_n1465.csv")

master_df.to_parquet(parquet_out, index=False)
master_df.to_csv(csv_out, index=False)

print(f"Saved parquet: {parquet_out} ({os.path.getsize(parquet_out) / 1024:.1f} KB)")
print(f"Saved csv:     {csv_out} ({os.path.getsize(csv_out) / 1024:.1f} KB)")

# Save feature partition dictionary
import json
partitions = {
    "target": "target_saa_binary",
    "id_column": "PATNO",
    "mri_features": mri_cols_all,
    "motor_features": motor_feature_names,
    "genetics_features": gen_cols,
    "prs_features": prs_cols,
    "demographics_features": demo_feature_names,
    "core_features_249": core_features,
    "full_features_252": full_features,
    "counts": {
        "n_total": len(master_df),
        "n_positive": int(pos_count),
        "n_negative": int(neg_count),
        "d_mri": len(mri_cols_all),
        "d_motor": len(motor_feature_names),
        "d_genetics": len(gen_cols),
        "d_prs": len(prs_cols),
        "d_demographics": len(demo_feature_names),
        "d_core": len(core_features),
        "d_full": len(full_features)
    }
}
with open(os.path.join(OUTPUT_DIR, "feature_partitions.json"), "w") as f:
    json.dump(partitions, f, indent=2)

print("Saved feature_partitions.json!")
print("TASK 1 DATASET ASSEMBLY COMPLETE AND VERIFIED!")
