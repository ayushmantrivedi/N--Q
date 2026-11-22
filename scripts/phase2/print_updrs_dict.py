import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')

updrs_rows = dict_df[dict_df['Variable'].isin(['updrs3_score', 'updrs3_score_on'])]
for idx, r in updrs_rows.iterrows():
    print("Variable:", r['Variable'])
    print("Description:", r['Description'])
    print("Derivation:", r['Derivation Notes'])
    print("-" * 50)

