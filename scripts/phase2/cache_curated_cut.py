import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import os

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
cache_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet"

print("Reading excel...")
df_cur = pd.read_excel(excel_path, sheet_name='20260810')
print("Saving parquet cache to:", cache_path)
df_cur.to_parquet(cache_path, index=False)
print("Saved! Parquet rows:", len(df_cur), "cols:", len(df_cur.columns))

# Find UPDRS columns
updrs_cols = [c for c in df_cur.columns if 'updrs' in c.lower() or 'np3' in c.lower() or 'np2' in c.lower() or 'np1' in c.lower()]
print("UPDRS columns in curated cut:", updrs_cols)

