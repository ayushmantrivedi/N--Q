import pandas as pd

df_corr = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_corrected_v1.csv")
target_patnos = [235630, 214284, 183051, 156304]

sub_corr = df_corr[df_corr["PATNO"].isin(target_patnos)]
print("Target rows in ppmi_mri_batch_001_corrected_v1.csv:")
print(sub_corr[["PATNO", "visit", "EVENT_ID", "PROTOCOL", "image_eligibility", "status"]].to_string())

xing_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv"
mri_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Magnetic_Resonance_Imaging__MRI__18Sep2026.csv"

df_xing = pd.read_csv(xing_path)
df_mri = pd.read_csv(mri_path)

print("\nXing Core Lab metadata for target 4:")
sub_xing = df_xing[df_xing["PATNO"].isin(target_patnos)]
print(sub_xing[["PATNO", "EVENT_ID", "MRI_SCAN_DATE", "MRI_SEQ_T1_WEIGHTED", "PROTOCOL"]].to_string())

print("\nMagnetic Resonance Imaging metadata for target 4:")
sub_mri = df_mri[df_mri["PATNO"].isin(target_patnos)]
print(sub_mri[["PATNO", "EVENT_ID", "INFODT", "MRICMPLT"]].to_string())

