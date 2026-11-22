import pandas as pd

updrs_path = r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv"
df = pd.read_csv(updrs_path, nrows=5)
print("MDS-UPDRS Part III columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

