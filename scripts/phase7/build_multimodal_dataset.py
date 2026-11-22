"""
NĀḌĪ-Q Phase 7: Multimodal Cohort Assembler.
Merges FreeSurfer MRI morphometry (204 IDPs), quantitative biospecimen assays (NfL, GFAP, pTau, tTau, ABeta, SAA kinetics),
genetics/PRS, and clinical longitudinal motor sequences.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

def assemble():
    print("=" * 80)
    print("ASSEMBLING PHASE 7 MULTIMODAL COHORT (N=341)")
    print("=" * 80)

    # 1. Base longitudinal multimodal cohort from Phase 6
    n341_path = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ppmi_longitudinal_5yr_multimodal_n341.parquet"
    df_base = pd.read_parquet(n341_path)
    print(f"Base cohort loaded: {df_base.shape}")
    patnos = set(df_base['PATNO'])

    # 2. Biospecimen Quantitative Assays (Current_Biospecimen_Analysis_Results_18Sep2026.csv)
    curr_bio_path = r"F:\NADI-Q data\PPMI\Biospecimen_Analysis\Current_Biospecimen_Analysis_Results_18Sep2026.csv"
    print("Loading biospecimen assays from:", curr_bio_path)
    df_curr = pd.read_csv(curr_bio_path, low_memory=False)
    
    # Filter for baseline events (CLINICAL_EVENT == 'BL') and our cohort
    curr_bl = df_curr[(df_curr['CLINICAL_EVENT'] == 'BL') & (df_curr['PATNO'].isin(patnos))].copy()
    
    # Clean numeric TESTVALUE
    def clean_val(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().replace('<', '').replace('>', '').replace(',', '')
        try:
            return float(s)
        except ValueError:
            return np.nan

    curr_bl['val_num'] = curr_bl['TESTVALUE'].apply(clean_val)

    # Extract target analytes:
    analytes_map = {
        'CSF_NFL': ('NFL', ['Cerebrospinal Fluid', 'CSF']),
        'PLASMA_NFL': ('NFL', ['Plasma', 'PLASMA']),
        'CSF_GFAP': ('GFAP', ['Cerebrospinal Fluid', 'CSF']),
        'PLASMA_GFAP': ('GFAP', ['Plasma', 'PLASMA']),
        'CSF_PTAU181': ('pTau181', ['Cerebrospinal Fluid', 'CSF']),
        'CSF_TTAU': ('tTau', ['Cerebrospinal Fluid', 'CSF']),
        'CSF_ABETA42': ('ABeta42', ['Cerebrospinal Fluid', 'CSF']),
        'CSF_ALPHASYN': ('CSF Alpha-synuclein', ['Cerebrospinal Fluid', 'CSF'])
    }

    bio_extracted = {p: {} for p in patnos}

    for col_name, (test_name, sample_types) in analytes_map.items():
        sub = curr_bl[(curr_bl['TESTNAME'] == test_name) & (curr_bl['TYPE'].isin(sample_types))]
        for p, grp in sub.groupby('PATNO'):
            vals = grp['val_num'].dropna()
            if len(vals) > 0:
                bio_extracted[p][col_name] = float(vals.mean())

    df_bio = pd.DataFrame.from_dict(bio_extracted, orient='index').reset_index()
    df_bio.rename(columns={'index': 'PATNO'}, inplace=True)
    print(f"Extracted quantitative CSF/Plasma analytes: {df_bio.shape}")

    # 3. SAA Kinetics (SAA_Biospecimen_Analysis_Results_18Sep2026.csv)
    saa_path = r"F:\NADI-Q data\PPMI\Biospecimen_Analysis\SAA_Biospecimen_Analysis_Results_18Sep2026.csv"
    print("Loading SAA kinetics from:", saa_path)
    df_saa = pd.read_csv(saa_path)
    saa_bl = df_saa[(df_saa['CLINICAL_EVENT'] == 'BL') & (df_saa['PATNO'].isin(patnos))].copy()

    # Aggregate SAA status (1 = Positive, 0 = Negative / Inconclusive) and 24h kinetic metrics
    saa_dict = {}
    for p, grp in saa_bl.groupby('PATNO'):
        is_pos = 1.0 if (grp['SAA_Status'] == 'Positive').any() else 0.0
        
        # Mean across replicates for kinetic features
        fmax_cols = [c for c in grp.columns if c.startswith('Fmax_24h_Rep')]
        slope_cols = [c for c in grp.columns if c.startswith('SLOPEMax_24h_Rep')]
        auc_cols = [c for c in grp.columns if c.startswith('AUC_24h_Rep')]
        ttt_cols = [c for c in grp.columns if c.startswith('TTT_24h_Rep')]

        def safe_mean(cols):
            present = [c for c in cols if c in grp]
            if not present:
                return np.nan
            vals = grp[present].apply(pd.to_numeric, errors='coerce').values.flatten()
            valid = vals[~np.isnan(vals)]
            return float(np.mean(valid)) if len(valid) > 0 else np.nan

        saa_dict[p] = {
            'SAA_positive': is_pos,
            'SAA_Fmax_24h': safe_mean(fmax_cols),
            'SAA_SlopeMax_24h': safe_mean(slope_cols),
            'SAA_AUC_24h': safe_mean(auc_cols),
            'SAA_TTT_24h': safe_mean(ttt_cols),
        }

    df_saa_agg = pd.DataFrame.from_dict(saa_dict, orient='index').reset_index()
    df_saa_agg.rename(columns={'index': 'PATNO'}, inplace=True)
    print(f"Extracted SAA kinetics: {df_saa_agg.shape}")

    # 4. Merge all together
    df_merged = df_base.merge(df_bio, on='PATNO', how='left')
    df_merged = df_merged.merge(df_saa_agg, on='PATNO', how='left')
    print(f"Final merged dataset shape: {df_merged.shape}")

    # Check missingness of new bio features
    bio_cols = list(df_bio.columns.drop('PATNO')) + list(df_saa_agg.columns.drop('PATNO'))
    print("\nBiofluid & Kinetic Missingness:")
    for c in bio_cols:
        n_pres = df_merged[c].notna().sum()
        pct = n_pres / len(df_merged) * 100
        print(f"  {c}: {n_pres} / {len(df_merged)} ({pct:.1f}%)")

    # Save to Phase 7 output
    out_parquet = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet"
    df_merged.to_parquet(out_parquet, index=False)
    print(f"\nSaved successfully to: {out_parquet}")

if __name__ == "__main__":
    assemble()
