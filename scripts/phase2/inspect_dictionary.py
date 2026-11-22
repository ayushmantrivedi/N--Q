import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"

dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')
print("Data dictionary columns:", list(dict_df.columns))

# Search for CSFSAA and related
saa_entries = dict_df[dict_df.apply(lambda row: row.astype(str).str.contains('CSFSAA|asyn|SAA', case=False).any(), axis=1)]
print(f"Found {len(saa_entries)} dictionary rows for SAA:")
for idx, row in saa_entries.iterrows():
    print(dict(row))

# Let's also check other rows that might be interesting (e.g. PATNO, EVENT_ID, visit)
key_entries = dict_df[dict_df.apply(lambda row: row.astype(str).str.contains('PATNO|EVENT_ID|visit|COHORT', case=False).any(), axis=1)]
print(f"\nFound {len(key_entries)} dictionary rows for key IDs:")
for idx, row in key_entries.head(10).iterrows():
    print(dict(row))

