import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')

saa_rows = dict_df[dict_df['Variable'].isin(['CSFSAA', 'CSFSAA_assay', 'asyn', 'NSD_Status', 'NSD_STAGE', 'Clinical_Stage'])]

for idx, r in saa_rows.iterrows():
    print(f"Variable: {r['Variable']}")
    print(f"  Category: {r['Category']}")
    print(f"  Description: {r['Description']}")
    print(f"  Code: {r['Code']}, Decode: {r['Decode']}")
    print(f"  Derivation: {r['Derivation Notes']}")
    print("-" * 50)
