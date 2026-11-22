import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

excel_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
dict_df = pd.read_excel(excel_path, sheet_name='Data dictionary')

# Let's inspect rows around CSFSAA
idx_list = dict_df[dict_df['Variable'] == 'CSFSAA'].index
for idx in idx_list:
    print(dict_df.iloc[idx:idx+5][['Variable', 'Code', 'Decode', 'Description']])

