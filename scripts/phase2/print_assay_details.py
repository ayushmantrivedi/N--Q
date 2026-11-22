import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')
for i in range(148, 152):
    row = dict_df.iloc[i].dropna().to_dict()
    print(f"Row {i}: {row}")

