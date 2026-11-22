import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
cache_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

# Load cohort
cur_df = pd.read_parquet(cache_path)
cur_bl = cur_df[cur_df['EVENT_ID'] == 'BL'].copy()

fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", usecols=['PATNO'])
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", usecols=['PATNO'])
fs_1713 = set(fs_cth['PATNO']).intersection(set(fs_vol['PATNO']))

cohort_bl = cur_bl[cur_bl['PATNO'].isin(fs_1713) & cur_bl['CSFSAA'].isin([0.0, 1.0, 3.0]) & cur_bl['updrs3_score'].notna()].copy()
cohort_pats = set(cohort_bl['PATNO'])

# Load raw UPDRS III
updrs = pd.read_csv(f"{DATA_DIR}\\Motor___MDS-UPDRS\\MDS-UPDRS_Part_III_18Sep2026.csv", low_memory=False)
updrs_bl = updrs[(updrs['EVENT_ID'] == 'BL') & (updrs['PATNO'].isin(cohort_pats))].copy()

print(f"Cohort size: {len(cohort_pats)}")
print(f"UPDRS BL rows for cohort: {len(updrs_bl)}, unique PATNO: {updrs_bl['PATNO'].nunique()}")

# Priority for deduplication: OFF state (or unmedicated)
# In PPMI, PDSTATE == 'OFF' or PDTRTMNT == 0 is preferred.
# If multiple, take OFF, then ON, then latest
def select_motor_row(group):
    # If only 1 row
    if len(group) == 1:
        return group.iloc[0]
    # Check for OFF state
    off_rows = group[group['PDSTATE'] == 'OFF']
    if len(off_rows) > 0:
        return off_rows.iloc[0]
    # Check for untreated
    untreated = group[group['PDTRTMNT'] == 0]
    if len(untreated) > 0:
        return untreated.iloc[0]
    # Otherwise first
    return group.iloc[0]

updrs_dedup = updrs_bl.groupby('PATNO', as_index=False).apply(select_motor_row, include_groups=False)
print(f"Deduplicated UPDRS rows: {len(updrs_dedup)}")

motor_items = [
    'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL',
    'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL',
    'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT',
    'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', 'NP3PTRML',
    'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL',
    'NP3RTALJ', 'NP3RTCON'
]

print("Missingness across 33 motor items in deduplicated UPDRS:")
item_missing = updrs_dedup[motor_items].isna().sum()
print(f"Max item missingness: {item_missing.max()} ({item_missing.max()/len(updrs_dedup):.2%})")
print(f"Total missing values across {len(updrs_dedup)} x 33 items: {item_missing.sum()} ({item_missing.sum()/(len(updrs_dedup)*33):.4%})")

