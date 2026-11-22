import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
cache_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

# 1. FreeSurfer 7.1 IDPs
fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", low_memory=False)
fs_sa  = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_SA_18Sep2026.csv", low_memory=False)
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", low_memory=False)

cth_features = [c for c in fs_cth.columns if c not in ['PATNO', 'EVENT_ID']]
sa_features  = [c for c in fs_sa.columns  if c not in ['PATNO', 'EVENT_ID']]
vol_features = [c for c in fs_vol.columns if c not in ['PATNO', 'EVENT_ID']]

print(f"FreeSurfer Features:")
print(f"  Cortical Thickness: {len(cth_features)} regions (units: mm)")
print(f"  Surface Area:       {len(sa_features)} regions (units: mm^2)")
print(f"  Subcortical Volume: {len(vol_features)} structures (units: mm^3)")
print(f"  Total Imaging IDPs: {len(cth_features) + len(sa_features) + len(vol_features)}")

# 2. MDS-UPDRS Part III
updrs = pd.read_csv(f"{DATA_DIR}\\Motor___MDS-UPDRS\\MDS-UPDRS_Part_III_18Sep2026.csv", nrows=10)
motor_items = [
    'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
    'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
    'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT',
    'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML',
    'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL',
    'NP3RTALJ', 'NP3RTCON'
]
print(f"\nMDS-UPDRS Part III Item Scores: {len(motor_items)} items (scale: 0-4 per item)")

# 3. Genetics & PRS
gen = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", nrows=10)
gen_cols = ['LRRK2', 'GBA', 'APOE', 'PATHVAR_COUNT']
prs = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", nrows=10)
prs_cols = ['GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS']
print(f"Genetics Consensus features: {gen_cols}")
print(f"PRS features: {prs_cols}")

# 4. Demographics
demo_cols = ['age', 'SEX', 'EDUCYRS']
print(f"Demographics features: {demo_cols}")

# Now examine missingness across our primary benchmark cohort
cur_df = pd.read_parquet(cache_path)
cur_bl = cur_df[cur_df['EVENT_ID'] == 'BL'].copy()

# Cohort of FS participants with conclusive SAA
fs_1713 = set(fs_cth['PATNO']).intersection(set(fs_sa['PATNO'])).intersection(set(fs_vol['PATNO']))
cohort_bl = cur_bl[cur_bl['PATNO'].isin(fs_1713) & cur_bl['CSFSAA'].isin([0.0, 1.0, 3.0]) & cur_bl['updrs3_score'].notna()].copy()
print(f"\nBenchmark Cohort Size: {len(cohort_bl)}")

# Merge with FreeSurfer IDPs
mri_bl = fs_cth[fs_cth['PATNO'].isin(cohort_bl['PATNO'])].merge(
    fs_sa[fs_sa['PATNO'].isin(cohort_bl['PATNO'])], on=['PATNO', 'EVENT_ID'], suffixes=('_cth', '_sa')
).merge(
    fs_vol[fs_vol['PATNO'].isin(cohort_bl['PATNO'])], on=['PATNO', 'EVENT_ID']
)
print(f"MRI merged shape for cohort: {mri_bl.shape}")

# Missingness in FreeSurfer
mri_missing = mri_bl.drop(columns=['PATNO', 'EVENT_ID']).isna().sum().sum()
print(f"Total missing values across all {len(mri_bl)} x {mri_bl.shape[1]-2} MRI IDPs: {mri_missing} ({mri_missing / (len(mri_bl) * (mri_bl.shape[1]-2)):.4%})")

# Demographics missingness
print("Demographics missingness in cohort:")
for c in ['age', 'SEX', 'EDUCYRS']:
    n_miss = cohort_bl[c].isna().sum()
    print(f"  {c}: {n_miss} missing ({n_miss / len(cohort_bl):.2%})")

# Genetics missingness
gen_full = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", low_memory=False)
prs_full = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", low_memory=False)

cohort_gen = cohort_bl[['PATNO']].merge(gen_full[['PATNO'] + gen_cols], on='PATNO', how='left')
print("Genetics missingness in cohort:")
for c in gen_cols:
    n_miss = cohort_gen[c].isna().sum()
    print(f"  {c}: {n_miss} missing ({n_miss / len(cohort_bl):.2%})")

cohort_prs = cohort_bl[['PATNO']].merge(prs_full[['PATNO'] + prs_cols], on='PATNO', how='left')
print("PRS missingness in cohort:")
for c in prs_cols:
    n_miss = cohort_prs[c].isna().sum()
    print(f"  {c}: {n_miss} missing ({n_miss / len(cohort_bl):.2%})")

