import pandas as pd

df_corr = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_corrected_v1.csv")
df_val = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_download_validation_v1.csv")

print("Corrected manifest length:", len(df_corr))
print("Validation manifest length:", len(df_val))

target_patnos = [235630, 214284, 183051, 156304]
print("\nTarget PATNOs check:")
for p in target_patnos:
    row_corr = df_corr[df_corr["PATNO"] == p]
    row_val = df_val[df_val["PATNO"] == p]
    corr_st = row_corr["status"].values[0] if len(row_corr) > 0 else "MISSING"
    val_st = row_val["status"].values[0] if len(row_val) > 0 else "MISSING"
    val_vf = row_val["verification_status"].values[0] if len(row_val) > 0 else "MISSING"
    print(f"PATNO {p}: in_corr={len(row_corr)>0}, corr_status={corr_st}, val_status={val_st}, val_verif={val_vf}")

print("\nCurrent validation manifest status counts:")
print(df_val["status"].value_counts())
print("\nCurrent validation manifest verification_status counts:")
print(df_val["verification_status"].value_counts())

pending = df_val[df_val["status"] != "PREPROCESS_PENDING"]["PATNO"].tolist()
print("\nAll awaiting acquisition (not PREPROCESS_PENDING):", len(pending), pending)

# Remaining unlocated participants after these 4:
remaining_after_4 = [p for p in pending if p not in target_patnos]
print("\nRemaining unlocated after the 4 candidate downloads:", len(remaining_after_4), remaining_after_4)

