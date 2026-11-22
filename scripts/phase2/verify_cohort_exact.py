import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
cache_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

# Load FreeSurfer IDs
fs_cth = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_CTH_18Sep2026.csv", usecols=['PATNO', 'EVENT_ID'])
fs_sa  = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_APARC_SA_18Sep2026.csv", usecols=['PATNO', 'EVENT_ID'])
fs_vol = pd.read_csv(f"{DATA_DIR}\\MR_Analysis_Results\\FS7_ASEG_VOL_18Sep2026.csv", usecols=['PATNO', 'EVENT_ID'])

pts_cth = set(fs_cth['PATNO'])
pts_sa  = set(fs_sa['PATNO'])
pts_vol = set(fs_vol['PATNO'])
fs_1713 = pts_cth.intersection(pts_sa).intersection(pts_vol)
fs_1716 = pts_cth.union(pts_sa)

print("FreeSurfer 7.1 participant counts:")
print(f"  CTH: {len(pts_cth)}, SA: {len(pts_sa)}, ASEG: {len(pts_vol)}")
print(f"  Complete 3-modality FS: {len(fs_1713)}")
print(f"  Union CTH/SA: {len(fs_1716)}")

# Load Curated Cut from cache
cur_df = pd.read_parquet(cache_path)
cur_bl = cur_df[cur_df['EVENT_ID'] == 'BL'].copy()
print(f"\nCurated Cut at BL: {len(cur_bl)} rows, {cur_bl['PATNO'].nunique()} unique participants")

# Load raw SAA table
saa_df = pd.read_csv(f"{DATA_DIR}\\Biospecimen_Analysis\\SAA_Biospecimen_Analysis_Results_18Sep2026.csv", low_memory=False)
saa_pts_any = set(saa_df['PATNO'])
saa_pts_bl = set(saa_df[saa_df['CLINICAL_EVENT'] == 'BL']['PATNO'])
print(f"Raw SAA unique PATNO: {len(saa_pts_any)} (any visit), {len(saa_pts_bl)} (BL)")

# Load raw MDS-UPDRS Part III table
updrs_df = pd.read_csv(f"{DATA_DIR}\\Motor___MDS-UPDRS\\MDS-UPDRS_Part_III_18Sep2026.csv", low_memory=False)
updrs_pts_any = set(updrs_df['PATNO'])
updrs_pts_bl = set(updrs_df[updrs_df['EVENT_ID'] == 'BL']['PATNO'])
print(f"Raw MDS-UPDRS III unique PATNO: {len(updrs_pts_any)} (any visit), {len(updrs_pts_bl)} (BL)")

# Load Genetics & PRS
gen_df = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\iu_genetic_consensus_20251025_18Sep2026.csv", low_memory=False)
prs_df = pd.read_csv(f"{DATA_DIR}\\_Subject_Characteristics\\Polygenic_Risk_Scores_18Sep2026.csv", low_memory=False)
gen_pts = set(gen_df['PATNO'])
prs_pts = set(prs_df['PATNO'])
print(f"Genetics unique PATNO: {len(gen_pts)}, PRS unique PATNO: {len(prs_pts)}")

# Map into FS cohort
print("\n" + "="*50)
print("INTERSECTION COUNTS WITH FREESURFER COHORT (N=1,713)")
print("="*50)

cur_fs = cur_bl[cur_bl['PATNO'].isin(fs_1713)].copy()
print(f"1. FS 1,713 present in Curated Cut BL: {len(cur_fs)}")
missing_cur_fs = fs_1713 - set(cur_bl['PATNO'])
print(f"   Missing from Curated Cut BL: {len(missing_cur_fs)} {missing_cur_fs}")

# Check SAA in FS cohort:
# Option A: From Raw SAA table at BL
raw_saa_fs_bl = set(fs_1713).intersection(saa_pts_bl)
print(f"2. FS 1,713 with Raw SAA at BL: {len(raw_saa_fs_bl)}")

# Option B: From Raw SAA table at ANY visit
raw_saa_fs_any = set(fs_1713).intersection(saa_pts_any)
print(f"3. FS 1,713 with Raw SAA at ANY visit: {len(raw_saa_fs_any)}")

# Option C: From Curated Cut CSFSAA at BL
cur_saa_nonnull = cur_fs[cur_fs['CSFSAA'].notna()]
print(f"4. FS 1,713 with Curated Cut CSFSAA at BL (non-null): {len(cur_saa_nonnull)}")
cur_saa_conclusive = cur_fs[cur_fs['CSFSAA'].isin([0.0, 1.0, 3.0])]
print(f"5. FS 1,713 with Curated Cut CSFSAA at BL (Conclusive: 0=Neg, 1=Pos LBD, 3=Pos MSA): {len(cur_saa_conclusive)}")
print("   CSFSAA Breakdown in FS 1,713:")
print(cur_fs['CSFSAA'].value_counts(dropna=False))

# Check MDS-UPDRS Part III in FS cohort
updrs_fs_bl = set(fs_1713).intersection(updrs_pts_bl)
print(f"\n6. FS 1,713 with Raw MDS-UPDRS III at BL: {len(updrs_fs_bl)}")
cur_updrs3_nonnull = cur_fs[cur_fs['updrs3_score'].notna()]
print(f"7. FS 1,713 with Curated updrs3_score at BL: {len(cur_updrs3_nonnull)}")

# Tri-modal intersection (FS + SAA + MDS-UPDRS III)
tri_any = fs_1713.intersection(saa_pts_any).intersection(updrs_pts_any)
tri_bl_raw = fs_1713.intersection(saa_pts_bl).intersection(updrs_pts_bl)
print(f"\n8. Tri-modal (FS 1,713 & SAA any & UPDRS any): {len(tri_any)}")
print(f"9. Tri-modal (FS 1,713 & SAA BL & UPDRS BL): {len(tri_bl_raw)}")

# With CTH 1,716
tri_1716_any = fs_1716.intersection(saa_pts_any).intersection(updrs_pts_any)
tri_1716_bl_raw = fs_1716.intersection(saa_pts_bl).intersection(updrs_pts_bl)
print(f"10. Tri-modal (CTH 1,716 & SAA any & UPDRS any): {len(tri_1716_any)}")
print(f"11. Tri-modal (CTH 1,716 & SAA BL & UPDRS BL): {len(tri_1716_bl_raw)}")

# Conclusive target cohort in Curated Cut:
# Patient has: FS 1,713 + Conclusive CSFSAA (0, 1, 3) + updrs3_score
primary_curated = cur_fs[cur_fs['CSFSAA'].isin([0.0, 1.0, 3.0]) & cur_fs['updrs3_score'].notna()]
print(f"\n12. Curated Primary Benchmark Cohort at BL (FS 1,713 & Conclusive SAA & updrs3_score): {len(primary_curated)}")
print("    Target Distribution:")
print("    - Positive (1 or 3):", primary_curated['CSFSAA'].isin([1.0, 3.0]).sum(), f"({primary_curated['CSFSAA'].isin([1.0, 3.0]).mean():.1%})")
print("    - Negative (0):", (primary_curated['CSFSAA'] == 0.0).sum(), f"({(primary_curated['CSFSAA'] == 0.0).mean():.1%})")

# Adding Genetics & PRS
primary_gen = primary_curated[primary_curated['PATNO'].isin(gen_pts)]
primary_all = primary_curated[primary_curated['PATNO'].isin(gen_pts) & primary_curated['PATNO'].isin(prs_pts)]
print(f"13. With Genetic Consensus: {len(primary_gen)}")
print(f"14. With Genetic Consensus AND PRS: {len(primary_all)}")

