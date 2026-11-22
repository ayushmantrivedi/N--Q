import os
import pandas as pd
import numpy as np

base_dir = r"F:\NADI-Q data"

# 1. Local MRI payloads on disk
ppmi_raw_ref = set(os.listdir(r"F:\NADI-Q data\PPMI\MRI_raw\PPMI")) if os.path.exists(r"F:\NADI-Q data\PPMI\MRI_raw\PPMI") else set()
ppmi_raw_down = set(os.listdir(r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI")) if os.path.exists(r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI") else set()
ppmi_proc_v2 = set([f.split("_")[0] for f in os.listdir(r"F:\NADI-Q data\PPMI\MRI_processed_v2") if f.endswith("_t1.nii.gz")])

all_local_ppmi_mri = set(int(p) for p in (ppmi_raw_ref | ppmi_raw_down) if p.isdigit())
print(f"PPMI physical MRI on disk: {len(all_local_ppmi_mri)} participants")
print(f"  Reference V2 N=15: {len(ppmi_raw_ref)}")
print(f"  Downloaded N=19: {len(ppmi_raw_down)}")
print(f"  Processed V2 NIfTI: {len(ppmi_proc_v2)}")

# 2. Tabular FreeSurfer IDPs
fs_aparc = pd.read_csv(r"F:\NADI-Q data\PPMI\MR_Analysis_Results\FS7_APARC_CTH_18Sep2026.csv", usecols=["PATNO"])
ppmi_fs_pts = set(fs_aparc["PATNO"].unique())
print(f"PPMI FreeSurfer 7 IDP participants: {len(ppmi_fs_pts)}")

# 3. Biomarkers (SAA & Biofluid)
saa_df = pd.read_csv(r"F:\NADI-Q data\PPMI\Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv", usecols=["PATNO"])
ppmi_saa_pts = set(saa_df["PATNO"].unique())
print(f"PPMI SAA participants: {len(ppmi_saa_pts)}")

# 4. Clinical Motor MDS-UPDRS
updrs_df = pd.read_csv(r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv", usecols=["PATNO"])
ppmi_updrs_pts = set(updrs_df["PATNO"].unique())
print(f"PPMI MDS-UPDRS Part III participants: {len(ppmi_updrs_pts)}")

# 5. Wearable Gait
gait1 = pd.read_csv(r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\Gait_Data___Arm_swing__Axivity__18Sep2026.csv", usecols=["PATNO"])
gait2 = pd.read_csv(r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\Gait_Data___Arm_swing__Opals__18Sep2026.csv", usecols=["PATNO"])
ppmi_gait_pts = set(gait1["PATNO"].unique()) | set(gait2["PATNO"].unique())
print(f"PPMI Wearable Gait participants: {len(ppmi_gait_pts)}")

# 6. Smartphone App
roche_df = pd.read_csv(r"F:\NADI-Q data\PPMI\Roche_Smartphone_App\Roche_PD_Monitoring_App_v2_data_18Sep2026.csv", usecols=["PATNO"])
ppmi_roche_pts = set(roche_df["PATNO"].unique())
print(f"PPMI Roche App participants: {len(ppmi_roche_pts)}")

# 7. Genetics
gen_df = pd.read_csv(r"F:\NADI-Q data\PPMI\_Subject_Characteristics\iu_genetic_consensus_20251025_18Sep2026.csv", usecols=["PATNO"])
ppmi_gen_pts = set(gen_df["PATNO"].unique())
print(f"PPMI Genetic Consensus participants: {len(ppmi_gen_pts)}")

# Intersections!
print("\n--- PPMI Multimodal Intersections ---")
print(f"Clinical + SAA Biomarker: {len(ppmi_updrs_pts & ppmi_saa_pts)}")
print(f"Clinical + SAA + FreeSurfer IDPs: {len(ppmi_updrs_pts & ppmi_saa_pts & ppmi_fs_pts)}")
print(f"Clinical + SAA + FreeSurfer IDPs + Genetics: {len(ppmi_updrs_pts & ppmi_saa_pts & ppmi_fs_pts & ppmi_gen_pts)}")
print(f"Clinical + SAA + FreeSurfer IDPs + Gait: {len(ppmi_updrs_pts & ppmi_saa_pts & ppmi_fs_pts & ppmi_gait_pts)}")
print(f"Clinical + SAA + Physical MRI on disk: {len(ppmi_updrs_pts & ppmi_saa_pts & all_local_ppmi_mri)}")
print(f"Clinical + SAA + FreeSurfer IDPs + Physical MRI: {len(ppmi_updrs_pts & ppmi_saa_pts & ppmi_fs_pts & all_local_ppmi_mri)}")

