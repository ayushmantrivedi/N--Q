import pandas as pd

target_patnos = [220494, 229409, 278950, 311004, 315185, 149006, 151050, 170472, 183263, 222893]

xing_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv"
mri_path = r"F:\NADI-Q data\PPMI\MRI_Acquisition\Magnetic_Resonance_Imaging__MRI__18Sep2026.csv"

df_xing = pd.read_csv(xing_path)
df_mri = pd.read_csv(mri_path)

print("=== Xing Core Lab Metadata for 10 Participants ===")
sub_xing = df_xing[df_xing["PATNO"].isin(target_patnos)]
print(sub_xing.to_string())

print("\n=== Magnetic Resonance Imaging Metadata for 10 Participants ===")
sub_mri = df_mri[df_mri["PATNO"].isin(target_patnos)]
print(sub_mri.to_string())

