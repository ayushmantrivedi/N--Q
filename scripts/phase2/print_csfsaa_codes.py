import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')

saa_rows = dict_df[dict_df['Variable'] == 'CSFSAA']
for idx, r in saa_rows.iterrows():
    print(r['Variable'], "|", r['Code'], "|", r['Decode'], "|", r['Description'])

