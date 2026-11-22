import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
cache_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

cur_df = pd.read_parquet(cache_path)
cur_bl = cur_df[cur_df['EVENT_ID'] == 'BL'].copy()

fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", low_memory=False)
fs_sa  = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_SA_18Sep2026.csv", low_memory=False)
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", low_memory=False)
fs_1713 = set(fs_cth['PATNO']).intersection(set(fs_sa['PATNO'])).intersection(set(fs_vol['PATNO']))

# Primary cohort
cohort = cur_bl[cur_bl['PATNO'].isin(fs_1713) & cur_bl['CSFSAA'].isin([0.0, 1.0, 3.0]) & cur_bl['updrs3_score'].notna()].copy()
pats = set(cohort['PATNO'])

print(f"Cohort size: {len(pats)}")

# FreeSurfer CTH stats
cth_cols = [c for c in fs_cth.columns if c not in ['PATNO', 'EVENT_ID']]
cth_sub = fs_cth[fs_cth['PATNO'].isin(pats)][cth_cols]
print("CTH regions:", len(cth_cols), "Mean range:", cth_sub.mean().min(), cth_sub.mean().max())

# FreeSurfer SA stats
sa_cols = [c for c in fs_sa.columns if c not in ['PATNO', 'EVENT_ID']]
sa_sub = fs_sa[fs_sa['PATNO'].isin(pats)][sa_cols]
print("SA regions:", len(sa_cols), "Mean range:", sa_sub.mean().min(), sa_sub.mean().max())

# FreeSurfer ASEG stats
vol_cols = [c for c in fs_vol.columns if c not in ['PATNO', 'EVENT_ID']]
vol_sub = fs_vol[fs_vol['PATNO'].isin(pats)][vol_cols]
print("ASEG regions:", len(vol_cols), "Mean range:", vol_sub.mean().min(), vol_sub.mean().max())

# Motor items stats
updrs = pd.read_csv(f"{DATA_DIR}\\Motor___MDS-UPDRS\\MDS-UPDRS_Part_III_18Sep2026.csv", low_memory=False)
updrs_bl = updrs[(updrs['EVENT_ID'] == 'BL') & (updrs['PATNO'].isin(pats))]
# Deduplicate: prefer OFF
def pick_row(g):
    if len(g) == 1: return g.iloc[0]
    off = g[g['PDSTATE'] == 'OFF']
    if len(off) > 0: return off.iloc[0]
    un = g[g['PDTRTMNT'] == 0]
    if len(un) > 0: return un.iloc[0]
    return g.iloc[0]
updrs_dedup = updrs_bl.groupby('PATNO', as_index=False).apply(pick_row, include_groups=False)

motor_items = [
    'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
    'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
    'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT',
    'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML',
    'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL',
    'NP3RTALJ', 'NP3RTCON'
]
updrs_sub = updrs_dedup[motor_items]
print("Motor items:", len(motor_items), "Mean total sum:", updrs_sub.sum(axis=1).mean())

# Genetics & PRS
gen = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", low_memory=False)
gen_sub = gen[gen['PATNO'].isin(pats)][['LRRK2', 'GBA', 'APOE', 'PATHVAR_COUNT']]
print("Genetics distributions:")
print(gen_sub['LRRK2'].value_counts(dropna=False))
print(gen_sub['GBA'].value_counts(dropna=False))

prs = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", low_memory=False)
prs_sub = prs[prs['PATNO'].isin(pats)][['GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS']]
print("PRS stats:")
print(prs_sub.describe())

