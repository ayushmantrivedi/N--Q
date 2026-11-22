import pandas as pd

xlsx_path = r"F:\NADI-Q data\PPMI\PPMI_Curated_Data_Cut_Public_20260810.xlsx"
df_dict = pd.read_excel(xlsx_path, sheet_name="Data dictionary", nrows=30)
print("Data Dictionary preview:")
print(df_dict.head(10).to_string())

# Load header and first 2 rows of main sheet
df_main = pd.read_excel(xlsx_path, sheet_name="20260810", nrows=2)
print("\nMain sheet columns (215):")
print(list(df_main.columns)[:40])
print("...")
print(list(df_main.columns)[40:80])

