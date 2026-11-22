import pandas as pd
import numpy as np

def check():
    df_n341 = pd.read_parquet(r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ppmi_longitudinal_5yr_multimodal_n341.parquet")
    patnos = set(df_n341['PATNO'])
    print(f"Total PATNOs in longitudinal multimodal cohort: {len(patnos)}")

    # SAA
    df_saa = pd.read_csv(r"F:\NADI-Q data\PPMI\Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv")
    saa_bl = df_saa[(df_saa['CLINICAL_EVENT'] == 'BL') & (df_saa['PATNO'].isin(patnos))]
    print(f"SAA BL unique PATNOs: {len(saa_bl['PATNO'].unique())} / {len(patnos)}")
    print("SAA Status counts:\n", saa_bl['SAA_Status'].value_counts())

    # Current biospecimen
    df_curr = pd.read_csv(r"F:\NADI-Q data\PPMI\Biospecimen_Analysis\Current_Biospecimen_Analysis_Results_18Sep2026.csv", low_memory=False)
    curr_bl = df_curr[(df_curr['CLINICAL_EVENT'] == 'BL') & (df_curr['PATNO'].isin(patnos))]
    
    test_targets = ['NFL', 'NfL', 'GFAP', 'pTau', 'pTau181', 'tTau', 'ABeta 1-42', 'ABeta42', 'CSF Alpha-synuclein']
    for t in test_targets:
        sub = curr_bl[curr_bl['TESTNAME'] == t]
        print(f"Test {t}: {len(sub['PATNO'].unique())} unique PATNOs, types: {sub['TYPE'].unique()}")

if __name__ == "__main__":
    check()

