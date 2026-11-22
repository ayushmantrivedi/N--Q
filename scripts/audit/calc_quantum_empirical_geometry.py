import os
import pandas as pd
import numpy as np

def compute_effective_rank(X):
    """
    Computes Roy & Vetterli entropy-based effective rank:
    erank(X) = exp( - sum(p_k * log(p_k)) ) where p_k = s_k / sum(s_j)
    """
    # Standardize
    X_clean = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    U, s, Vt = np.linalg.svd(X_clean, full_matrices=False)
    p = s / np.sum(s)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    erank = np.exp(entropy)
    
    # Participation ratio
    pr = (np.sum(s**2))**2 / np.sum(s**4)
    
    # Cumulative explained variance
    var_exp = (s**2) / np.sum(s**2)
    cum_var = np.cumsum(var_exp)
    
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "effective_rank": round(float(erank), 2),
        "participation_ratio": round(float(pr), 2),
        "singular_values": s[:10].tolist(),
        "cum_var_4": round(float(cum_var[min(3, len(cum_var)-1)] * 100), 1),
        "cum_var_8": round(float(cum_var[min(7, len(cum_var)-1)] * 100), 1),
        "cum_var_12": round(float(cum_var[min(11, len(cum_var)-1)] * 100), 1),
        "cum_var_16": round(float(cum_var[min(15, len(cum_var)-1)] * 100), 1),
        "cum_var_32": round(float(cum_var[min(31, len(cum_var)-1)] * 100), 1),
    }

print("Computing empirical feature geometry on real PPMI data...")

# 1. FreeSurfer 7 Morphometry (ASEG Volumes)
df_aseg = pd.read_csv(r"F:\NADI-Q data\PPMI\MR_Analysis_Results\FS7_ASEG_VOL_18Sep2026.csv")
num_cols_aseg = df_aseg.select_dtypes(include=[np.number]).columns
num_cols_aseg = [c for c in num_cols_aseg if c not in ["PATNO", "Unnamed: 0", "EVENT_ID"]]
X_aseg = df_aseg[num_cols_aseg].dropna()
res_aseg = compute_effective_rank(X_aseg.values)
print("\nFreeSurfer 7 ASEG Volumes Geometry:")
print(res_aseg)

# 2. FreeSurfer 7 Cortical Thickness (APARC CTH)
df_cth = pd.read_csv(r"F:\NADI-Q data\PPMI\MR_Analysis_Results\FS7_APARC_CTH_18Sep2026.csv")
num_cols_cth = df_cth.select_dtypes(include=[np.number]).columns
num_cols_cth = [c for c in num_cols_cth if c not in ["PATNO", "Unnamed: 0", "EVENT_ID"]]
X_cth = df_cth[num_cols_cth].dropna()
res_cth = compute_effective_rank(X_cth.values)
print("\nFreeSurfer 7 APARC Cortical Thickness Geometry:")
print(res_cth)

# 3. MDS-UPDRS Part III Motor Score Items
df_updrs = pd.read_csv(r"F:\NADI-Q data\PPMI\Motor___MDS-UPDRS\MDS-UPDRS_Part_III_18Sep2026.csv", low_memory=False)
num_cols_updrs = [c for c in df_updrs.columns if c.startswith("NP3") and c not in ["NP3TOT", "NP3SPCH", "NP3FTAP"]]
# Fill or drop na
X_updrs = df_updrs[num_cols_updrs].dropna()
res_updrs = compute_effective_rank(X_updrs.values)
print("\nMDS-UPDRS Part III Items Geometry:")
print(res_updrs)

# 4. Merged Multimodal Matrix (Intersection of FreeSurfer + MDS-UPDRS + Demographics)
merged_df = pd.merge(df_aseg[["PATNO"] + num_cols_aseg[:20]], df_updrs[["PATNO"] + num_cols_updrs[:15]], on="PATNO").dropna()
X_multi = merged_df.drop(columns=["PATNO"]).values
res_multi = compute_effective_rank(X_multi)
print("\nCombined Multimodal (Brain Morphometry + Motor Exam) Geometry:")
print(res_multi)

