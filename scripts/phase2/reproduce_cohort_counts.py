import os
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"

# Load files
fs_cth = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_CTH_18Sep2026.csv"), low_memory=False)
fs_sa  = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_SA_18Sep2026.csv"), low_memory=False)
fs_vol = pd.read_csv(os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_ASEG_VOL_18Sep2026.csv"), low_memory=False)
saa    = pd.read_csv(os.path.join(DATA_DIR, r"Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv"), low_memory=False)
updrs3 = pd.read_csv(os.path.join(DATA_DIR, r"Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv"), low_memory=False)
gen    = pd.read_csv(os.path.join(DATA_DIR, r"_Subject_Characteristics\iu_genetic_consensus_20251025_18Sep2026.csv"), low_memory=False)
prs    = pd.read_csv(os.path.join(DATA_DIR, r"_Subject_Characteristics\Polygenic_Risk_Scores_18Sep2026.csv"), low_memory=False)

print("=== FREESURFER COHERENCE ===")
pts_cth = set(fs_cth['PATNO'])
pts_sa  = set(fs_sa['PATNO'])
pts_vol = set(fs_vol['PATNO'])

print(f"CTH pts: {len(pts_cth)}")
print(f"SA pts:  {len(pts_sa)}")
print(f"VOL pts: {len(pts_vol)}")
print(f"CTH == SA: {pts_cth == pts_sa}")
print(f"VOL - CTH (extra in VOL): {pts_vol - pts_cth}")
print(f"CTH - VOL (missing in VOL): {pts_cth - pts_vol}")
fs_all = pts_cth.intersection(pts_sa).intersection(pts_vol)
print(f"FreeSurfer complete 3-modality (CTH & SA & VOL): {len(fs_all)}")

print("\n=== SAA ANALYSIS ===")
print("SAA columns:", list(saa.columns))
print("CLINICAL_EVENT counts:\n", saa['CLINICAL_EVENT'].value_counts(dropna=False).head(15))
print("SAA_Status counts:\n", saa['SAA_Status'].value_counts(dropna=False))
if 'SAA_Type' in saa.columns:
    print("SAA_Type counts:\n", saa['SAA_Type'].value_counts(dropna=False))
if 'SAAMethod' in saa.columns:
    print("SAAMethod counts:\n", saa['SAAMethod'].value_counts(dropna=False))

# SAA unique participants
pts_saa_any = set(saa['PATNO'])
print(f"SAA unique participants (any visit): {len(pts_saa_any)}")

# Baseline SAA
saa_bl = saa[saa['CLINICAL_EVENT'].isin(['BL', 'Baseline', 'SC', 'V01'])]
pts_saa_bl = set(saa_bl['PATNO'])
print(f"SAA unique participants at baseline (BL/Baseline/SC/V01): {len(pts_saa_bl)}")

# Participant with SAA at BL vs any
print(f"Participants with SAA but NOT baseline SAA: {len(pts_saa_any - pts_saa_bl)}")

print("\n=== MDS-UPDRS PART III ANALYSIS ===")
pts_updrs_any = set(updrs3['PATNO'])
print(f"MDS-UPDRS Part III unique participants (any visit): {len(pts_updrs_any)}")
updrs_bl = updrs3[updrs3['EVENT_ID'].isin(['BL', 'SC', 'V01'])]
pts_updrs_bl = set(updrs_bl['PATNO'])
print(f"MDS-UPDRS Part III unique participants at baseline (BL/SC/V01): {len(pts_updrs_bl)}")
updrs_strict_bl = updrs3[updrs3['EVENT_ID'] == 'BL']
pts_updrs_strict_bl = set(updrs_strict_bl['PATNO'])
print(f"MDS-UPDRS Part III unique participants at strict BL: {len(pts_updrs_strict_bl)}")

print("\n=== GENETICS & PRS ANALYSIS ===")
pts_gen = set(gen['PATNO'])
pts_prs = set(prs['PATNO'])
print(f"Genetic consensus unique participants: {len(pts_gen)}")
print(f"PRS unique participants: {len(pts_prs)}")

print("\n=== INTERSECTION AUDIT ===")
# FreeSurfer vs SAA vs MDS-UPDRS
inter_fs_saa_any = fs_all.intersection(pts_saa_any)
inter_fs_saa_bl  = fs_all.intersection(pts_saa_bl)
print(f"FS (complete 1713) & SAA (any visit): {len(inter_fs_saa_any)}")
print(f"FS (complete 1713) & SAA (baseline): {len(inter_fs_saa_bl)}")

inter_fs_saa_updrs_any = fs_all.intersection(pts_saa_any).intersection(pts_updrs_any)
inter_fs_saa_updrs_bl  = fs_all.intersection(pts_saa_bl).intersection(pts_updrs_bl)
inter_fs_saa_updrs_strict_bl = fs_all.intersection(pts_saa_bl).intersection(pts_updrs_strict_bl)

print(f"FS & SAA (any) & MDS-UPDRS (any): {len(inter_fs_saa_updrs_any)}")
print(f"FS & SAA (bl)  & MDS-UPDRS (bl):  {len(inter_fs_saa_updrs_bl)}")
print(f"FS & SAA (bl)  & MDS-UPDRS (strict BL): {len(inter_fs_saa_updrs_strict_bl)}")

# Also test with all CTH (1716)
inter_cth_saa_updrs_any = pts_cth.intersection(pts_saa_any).intersection(pts_updrs_any)
inter_cth_saa_updrs_bl  = pts_cth.intersection(pts_saa_bl).intersection(pts_updrs_bl)
print(f"CTH (1716) & SAA (any) & MDS-UPDRS (any): {len(inter_cth_saa_updrs_any)}")
print(f"CTH (1716) & SAA (bl) & MDS-UPDRS (bl): {len(inter_cth_saa_updrs_bl)}")

# Now add Genetics & PRS
inter_quad_any = inter_fs_saa_updrs_any.intersection(pts_gen)
inter_quad_bl  = inter_fs_saa_updrs_bl.intersection(pts_gen)
print(f"FS & SAA (any) & MDS-UPDRS (any) & GenConsensus: {len(inter_quad_any)}")
print(f"FS & SAA (bl)  & MDS-UPDRS (bl)  & GenConsensus: {len(inter_quad_bl)}")

inter_quad_prs_any = inter_quad_any.intersection(pts_prs)
inter_quad_prs_bl  = inter_quad_bl.intersection(pts_prs)
print(f"FS & SAA (any) & MDS-UPDRS (any) & GenConsensus & PRS: {len(inter_quad_prs_any)}")
print(f"FS & SAA (bl)  & MDS-UPDRS (bl)  & GenConsensus & PRS: {len(inter_quad_prs_bl)}")

