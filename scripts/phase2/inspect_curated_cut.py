import os
import openpyxl
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"

wb = openpyxl.load_workbook(excel_path, read_only=True)
sheet_names = wb.sheetnames
print("Sheets in Curated Data Cut:", sheet_names)

# Inspect columns of the main sheets
for sheet in sheet_names:
    print(f"\n--- Sheet: {sheet} ---")
    df_head = pd.read_excel(excel_path, sheet_name=sheet, nrows=5)
    print("Columns count:", len(df_head.columns))
    saa_cols = [c for c in df_head.columns if 'saa' in str(c).lower() or 'syn' in str(c).lower()]
    print("SAA related columns:", saa_cols)

