import os
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"

fs_cth = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_CTH_18Sep2026.csv"), low_memory=False)
fs_sa  = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_SA_18Sep2026.csv"), low_memory=False)
fs_vol = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_ASEG_VOL_18Sep2026.csv"), low_memory=False)
saa    = pd.read_csv(os.path.join(DATA_DIR, r"Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv"), low_memory=False)
updrs3 = pd.read_csv(os.path.join(DATA_DIR, r"Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv"), low_memory=False)

print("=== CHECKING THE 3 PARTICIPANTS MISSING IN ASEG ===")
missing_in_vol = set(fs_cth['PATNO']) - set(fs_vol['PATNO'])
for pat in missing_in_vol:
    in_saa = pat in set(saa['PATNO'])
    saa_events = saa[saa['PATNO'] == pat]['CLINICAL_EVENT'].tolist() if in_saa else []
    print(f"PATNO {pat}: in_saa={in_saa}, saa_events={saa_events}")

print("\n=== SAA DUPLICATE AUDIT AT BASELINE ===")
saa_bl = saa[saa['CLINICAL_EVENT'] == 'BL']
print(f"SAA rows with CLINICAL_EVENT == 'BL': {len(saa_bl)}")
print(f"SAA unique PATNO with CLINICAL_EVENT == 'BL': {saa_bl['PATNO'].nunique()}")
saa_bl_dups = saa_bl[saa_bl.duplicated(subset=['PATNO'], keep=False)].sort_values('PATNO')
print(f"Number of duplicate BL rows in SAA: {len(saa_bl_dups)}")
if len(saa_bl_dups) > 0:
    print("Sample duplicate BL rows in SAA:")
    print(saa_bl_dups[['PATNO', 'CLINICAL_EVENT', 'SAAMethod', 'SAA_Status', 'SAA_Type', 'RUNDATE']].head(10))

print("\n=== SAA TARGET CLASS DISTRIBUTION ===")
print("Overall SAA_Status (all rows):")
print(saa['SAA_Status'].value_counts(dropna=False))

print("\nBaseline SAA_Status (all BL rows):")
print(saa_bl['SAA_Status'].value_counts(dropna=False))

# If deduplicated by PATNO:
saa_bl_unique = saa_bl.drop_duplicates(subset=['PATNO'], keep='first')
print("\nBaseline SAA_Status (deduplicated by PATNO, first):")
print(saa_bl_unique['SAA_Status'].value_counts(dropna=False))

# Check for disagreement among duplicates
if len(saa_bl_dups) > 0:
    grouped = saa_bl_dups.groupby('PATNO')['SAA_Status'].nunique()
    disagreements = grouped[grouped > 1]
    print(f"Participants with conflicting SAA_Status duplicates at BL: {len(disagreements)}")
    if len(disagreements) > 0:
        print("Conflicting participants:", disagreements.index.tolist())
        print(saa_bl[saa_bl['PATNO'].isin(disagreements.index)][['PATNO', 'CLINICAL_EVENT', 'SAAMethod', 'SAA_Status', 'RUNDATE']])

print("\n=== NON-BASELINE SAA FOR FS PARTICIPANTS ===")
fs_pts = set(fs_cth['PATNO'])
fs_in_saa = saa[saa['PATNO'].isin(fs_pts)]
fs_saa_events = fs_in_saa.groupby('PATNO')['CLINICAL_EVENT'].unique()
fs_no_bl_saa = [pat for pat, evs in fs_saa_events.items() if 'BL' not in evs]
print(f"FS participants with SAA but NO BL event: {len(fs_no_bl_saa)}")
if len(fs_no_bl_saa) > 0:
    sample_events = {p: list(fs_saa_events[p]) for p in fs_no_bl_saa[:5]}
    print(f"Sample visits for FS participants without BL SAA: {sample_events}")

