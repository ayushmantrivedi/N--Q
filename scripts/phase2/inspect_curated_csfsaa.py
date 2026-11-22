import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
df_cur = pd.read_excel(excel_path, sheet_name='20260810')

print("Curated cut shape:", df_cur.shape)
print("Unique PATNO:", df_cur['PATNO'].nunique())
print("Unique EVENT_ID:", df_cur['EVENT_ID'].value_counts().head(10).to_dict())

bl_df = df_cur[df_cur['EVENT_ID'] == 'BL']
print(f"Curated cut BL rows: {len(bl_df)}, unique PATNO: {bl_df['PATNO'].nunique()}")

print("\nCSFSAA value counts at BL:")
print(bl_df['CSFSAA'].value_counts(dropna=False))

print("\nCSFSAA_assay value counts at BL:")
print(bl_df['CSFSAA_assay'].value_counts(dropna=False))

print("\nCOHORT breakdown at BL:")
print(bl_df['COHORT'].value_counts(dropna=False))

print("\nCross-tab of COHORT vs CSFSAA at BL:")
print(pd.crosstab(bl_df['COHORT'], bl_df['CSFSAA'], dropna=False))

