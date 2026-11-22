import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

DATA_DIR = r"F:\NADI-Q data\PPMI"
saa = pd.read_csv(f"{DATA_DIR}\\Biospecimen_Analysis\\SAA_Biospecimen_Analysis_Results_18Sep2026.csv", low_memory=False)

print("SAA table shape:", saa.shape)
print("SAA methods:", saa['SAAMethod'].value_counts(dropna=False))

print("\n--- 24-hour Assay Stats by SAA_Status ---")
saa_24h = saa[saa['SAAMethod'] == 'Amprion-24h alpha-synuclein-SAA']
cols_24h = ['Fmax_24h_Rep1', 'TTT_24h_Rep1', 'AUC_24h_Rep1', 'SLOPEMax_24h_Rep1']
for c in cols_24h:
    print(f"\nMetric: {c}")
    print(saa_24h.groupby('SAA_Status')[c].describe()[['count', 'mean', 'std', 'min', '50%', 'max']])

print("\n--- 150-hour Assay Stats by SAA_Status ---")
saa_150h = saa[saa['SAAMethod'] == 'Amprion-Alpha-synuclein-SAA']
cols_150h = ['Fmax_150h_Rep1', 'TTT_150h_Rep1', 'AUC_150h_Rep1', 'T50_150h_Rep1']
for c in cols_150h:
    if c in saa_150h.columns:
        print(f"\nMetric: {c}")
        print(saa_150h.groupby('SAA_Status')[c].describe()[['count', 'mean', 'std', 'min', '50%', 'max']])

