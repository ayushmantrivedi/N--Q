import pandas as pd

updrs_path = r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv"
df = pd.read_csv(updrs_path, low_memory=False)

bl_rows = df[df['EVENT_ID'] == 'BL']
print(f"Total BL rows: {len(bl_rows)}, Unique PATNO: {bl_rows['PATNO'].nunique()}")

dups = bl_rows[bl_rows.duplicated(subset=['PATNO'], keep=False)].sort_values('PATNO')
print(f"Duplicate BL rows: {len(dups)}")
print("Sample duplicate BL rows:")
print(dups[['PATNO', 'EVENT_ID', 'INFODT', 'PDTRTMNT', 'PDSTATE', 'HRPOSTMED', 'NP3TOT', 'NHY']].head(15))

print("\nPDSTATE breakdown in duplicates:")
print(dups['PDSTATE'].value_counts(dropna=False))

print("\nPDTRTMNT breakdown in duplicates:")
print(dups['PDTRTMNT'].value_counts(dropna=False))

