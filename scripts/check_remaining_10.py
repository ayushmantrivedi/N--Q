import pandas as pd

df_corr = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_corrected_v1.csv")
df_val = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_download_validation_v1.csv")

xing_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv"
mri_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Magnetic_Resonance_Imaging__MRI__18Sep2026.csv"
df_xing = pd.read_csv(xing_path)
df_mri = pd.read_csv(mri_path)

verified_patnos = [220494, 229409, 278950, 311004, 315185, 149006, 151050, 170472, 183263, 222893]
batch_002_patnos = [235630, 214284, 183051, 156304]

remaining_10 = [p for p in df_corr["PATNO"] if p not in verified_patnos and p not in batch_002_patnos]
print("Remaining 10 participants:", remaining_10)

details = []
for p in remaining_10:
    sub_x = df_xing[(df_xing["PATNO"] == p) & (df_xing["EVENT_ID"] == "BL")]
    sub_m = df_mri[(df_mri["PATNO"] == p) & (df_mri["EVENT_ID"] == "BL")]
    
    bl_date = sub_x["MRI_SCAN_DATE"].values[0] if not sub_x.empty else (sub_m["INFODT"].values[0] if not sub_m.empty else "N/A")
    proto = sub_x["PROTOCOL"].values[0] if not sub_x.empty else "N/A"
    t1_avail = sub_x["MRI_SEQ_T1_WEIGHTED"].values[0] if not sub_x.empty else "N/A"
    mri_cmplt = sub_m["MRICMPLT"].values[0] if not sub_m.empty else "N/A"
    
    details.append({
        "PATNO": p,
        "visit": "BL",
        "EVENT_ID": "BL",
        "baseline_mri_date": bl_date,
        "protocol": proto,
        "t1_reported": t1_avail,
        "mri_completed": mri_cmplt
    })

df_rem = pd.DataFrame(details)
print(df_rem.to_string(index=False))

