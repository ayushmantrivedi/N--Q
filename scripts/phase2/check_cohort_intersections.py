import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

# Load FreeSurfer IDs
DATA_DIR = r"F:\NADI-Q data\PPMI"
fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", usecols=['PATNO', 'EVENT_ID'])
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", usecols=['PATNO', 'EVENT_ID'])
fs_pts_1716 = set(fs_cth['PATNO'])
fs_pts_1713 = set(fs_cth['PATNO']).intersection(set(fs_vol['PATNO']))

# Load Curated cut BL
excel_path = f"{DATA_DIR}\\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
df_cur = pd.read_excel(excel_path, sheet_name='20260810')
cur_bl = df_cur[df_cur['EVENT_ID'] == 'BL'].copy()

# Merge FS with Curated Cut at BL
cur_fs_1716 = cur_bl[cur_bl['PATNO'].isin(fs_pts_1716)].copy()
cur_fs_1713 = cur_bl[cur_bl['PATNO'].isin(fs_pts_1713)].copy()

print(f"Curated cut participants matching FS 1716 at BL: {len(cur_fs_1716)}")
print(f"Curated cut participants matching FS 1713 at BL: {len(cur_fs_1713)}")

print("\n--- SAA Distribution in Curated Cut for FreeSurfer Cohort (N=1,713) ---")
print(cur_fs_1713['CSFSAA'].value_counts(dropna=False))
print("\nCSFSAA Decode Breakdown:")
saa_map = {0.0: 'Negative', 1.0: 'Positive LBD-like', 2.0: 'Inconclusive', 3.0: 'Positive MSA-like', np.nan: 'Missing'}
print(cur_fs_1713['CSFSAA'].map(saa_map).value_counts(dropna=False))

print("\n--- Diagnostic COHORT in Curated Cut for FreeSurfer Cohort (N=1,713) ---")
cohort_map = {1: 'Parkinson (PD)', 2: 'Healthy Control (HC)', 3: 'SWEDD', 4: 'Prodromal'}
print(cur_fs_1713['COHORT'].map(cohort_map).value_counts(dropna=False))

print("\n--- Cross-Tab: Diagnostic Cohort vs CSFSAA ---")
print(pd.crosstab(cur_fs_1713['COHORT'].map(cohort_map), cur_fs_1713['CSFSAA'].map(saa_map), dropna=False))

# Check MDS-UPDRS Part III in Curated Cut at BL (NP3TOT)
print("\n--- MDS-UPDRS Part III (NP3TOT) in Curated Cut for FS Cohort ---")
print("NP3TOT non-null count:", cur_fs_1713['NP3TOT'].notna().sum())
print("NP3TOT missing count:", cur_fs_1713['NP3TOT'].isna().sum())

# Check Genetics & PRS in Curated Cut or Raw files
gen = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", usecols=['PATNO', 'LRRK2', 'GBA', 'APOE'])
prs = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", usecols=['PATNO', 'GP2_PGS', 'META5_PGS'])

gen_fs = set(cur_fs_1713['PATNO']).intersection(set(gen['PATNO']))
prs_fs = set(cur_fs_1713['PATNO']).intersection(set(prs['PATNO']))
print(f"Genetics consensus matches in FS 1,713: {len(gen_fs)}")
print(f"PRS matches in FS 1,713: {len(prs_fs)}")

# Complete multimodal intersection at baseline
has_conclusive_saa = cur_fs_1713['CSFSAA'].isin([0.0, 1.0, 3.0])
has_np3tot = cur_fs_1713['NP3TOT'].notna()
cur_valid = cur_fs_1713[has_conclusive_saa & has_np3tot]
print(f"\nFS 1,713 with Conclusive SAA (0, 1, 3) AND MDS-UPDRS Part III (NP3TOT): {len(cur_valid)}")
cur_valid_gen = cur_valid[cur_valid['PATNO'].isin(gen_fs)]
print(f"FS 1,713 with Conclusive SAA, NP3TOT, AND Genetics: {len(cur_valid_gen)}")
cur_valid_prs = cur_valid[cur_valid['PATNO'].isin(prs_fs)]
print(f"FS 1,713 with Conclusive SAA, NP3TOT, AND PRS: {len(cur_valid_prs)}")
cur_valid_all = cur_valid[cur_valid['PATNO'].isin(gen_fs) & cur_valid['PATNO'].isin(prs_fs)]
print(f"FS 1,713 with Conclusive SAA, NP3TOT, Genetics, AND PRS: {len(cur_valid_all)}")

