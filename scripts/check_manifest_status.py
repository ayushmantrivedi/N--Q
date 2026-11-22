import pandas as pd

df = pd.read_csv(r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_download_validation_v1.csv")
print("=== All 24 participants in Batch 001 ===")
for idx, r in df.iterrows():
    print(f"PATNO {r['PATNO']}: status={r['status']}, verif={r['verification_status']}, files={r['dicom_file_count']}, shape={r['reconstructed_shape']}, suid={r['selected_series_uid']}")

verified = df[df['status'] == 'PREPROCESS_PENDING']
print(f"\nTotal PREPROCESS_PENDING (verified): {len(verified)}")
missing = df[df['status'] != 'PREPROCESS_PENDING']
print(f"Total still missing: {len(missing)}")
print("Missing PATNOs:", missing['PATNO'].tolist())

