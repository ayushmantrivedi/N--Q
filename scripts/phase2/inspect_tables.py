import os
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"

def inspect_file(name, filepath, patno_col="PATNO", visit_col="EVENT_ID"):
    print(f"\n=================== {name} ===================")
    if not os.path.exists(filepath):
        print(f"FILE NOT FOUND: {filepath}")
        return None
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"Columns (first 15): {list(df.columns[:15])}")
    
    # Check PATNO column
    actual_patno_col = None
    for c in [patno_col, "PATNO", "participant_id", "Subject"]:
        if c in df.columns:
            actual_patno_col = c
            break
    print(f"Participant col: {actual_patno_col}")
    
    # Check visit column
    actual_visit_col = None
    for c in [visit_col, "EVENT_ID", "event_id", "VISCODE", "visit"]:
        if c in df.columns:
            actual_visit_col = c
            break
    print(f"Visit col: {actual_visit_col}")
    
    if actual_patno_col:
        unique_pts = df[actual_patno_col].nunique()
        print(f"Unique participants: {unique_pts}")
        dup_rows = df.duplicated(subset=[actual_patno_col]).sum()
        print(f"Duplicate participant rows: {dup_rows}")
        if actual_visit_col:
            dup_pt_visit = df.duplicated(subset=[actual_patno_col, actual_visit_col]).sum()
            print(f"Duplicate (participant, visit) pairs: {dup_pt_visit}")
            visit_dist = df[actual_visit_col].value_counts().head(10).to_dict()
            print(f"Top 10 visits: {visit_dist}")
            bl_pts = df[df[actual_visit_col].isin(['BL', 'SC', 'V01', 'Baseline'])][actual_patno_col].nunique()
            print(f"Baseline participants (BL/SC/V01): {bl_pts}")
    return df

# Inspect each table
fs_cth = inspect_file("FreeSurfer CTH", os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_CTH_18Sep2026.csv"))
fs_sa  = inspect_file("FreeSurfer SA",  os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_APARC_SA_18Sep2026.csv"))
fs_vol = inspect_file("FreeSurfer ASEG",os.path.join(DATA_DIR, r"MR_Analysis_Results\FS7_ASEG_VOL_18Sep2026.csv"))
saa    = inspect_file("CSF SAA",        os.path.join(DATA_DIR, r"Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv"))
updrs3 = inspect_file("MDS-UPDRS Part III", os.path.join(DATA_DIR, r"Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv"))
gen    = inspect_file("Genetic Consensus",  os.path.join(DATA_DIR, r"_Subject_Characteristics\iu_genetic_consensus_20251025_18Sep2026.csv"))
prs    = inspect_file("Polygenic Risk Scores", os.path.join(DATA_DIR, r"_Subject_Characteristics\Polygenic_Risk_Scores_18Sep2026.csv"))

