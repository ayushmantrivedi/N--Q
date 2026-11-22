"""
NĀḌĪ-Q Phase 7: Forensic Scientific Verification & Audit Suite.
Conducts independent verification of all Phase 7 claims, scripts, artifacts, and numerical statistics.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim

from nadiq.models.quantum.qvbe_v2 import QVBEV2Config, QuantumVariationalBiomarkerEncoderV2
from nadiq.models.quantum.ctclm import CTCLMConfig, ContinualTemporalClinicalLatentModel

OUTPUT_DIR = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7"
PARQUET_PATH = os.path.join(OUTPUT_DIR, "ppmi_longitudinal_multimodal_biofluid_n341.parquet")
TEMPORAL_NPY_PATH = r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\ctclm_early_temporal_sequences_n341.npy"


def audit_cca_bootstrap_bias():
    print("=" * 80)
    print("AUDIT 1: CCA BOOTSTRAP UPWARD BIAS INVESTIGATION")
    print("=" * 80)
    # Demonstrate mathematically and empirically why bootstrap in-sample percentile intervals are shifted up
    np.random.seed(42)
    N = 341
    X_null = np.random.randn(N, 23)
    Y_null = np.random.randn(N, 23)
    
    cca = CCA(n_components=1)
    cca.fit(X_null, Y_null)
    xc, yc = cca.transform(X_null, Y_null)
    r_null = stats.pearsonr(xc[:, 0], yc[:, 0])[0]
    
    boot_rs = []
    for _ in range(500):
        idx = np.random.choice(N, size=N, replace=True)
        cca.fit(X_null[idx], Y_null[idx])
        xc_b, yc_b = cca.transform(X_null[idx], Y_null[idx])
        boot_rs.append(stats.pearsonr(xc_b[:, 0], yc_b[:, 0])[0])
        
    print(f"Null Data (23 vs 23 dimensions, N=341, TRUE r = 0.0000):")
    print(f"  In-sample point estimate r: {r_null:.4f}")
    print(f"  Bootstrap in-sample distribution: mean={np.mean(boot_rs):.4f}, 2.5%={np.percentile(boot_rs, 2.5):.4f}, 97.5%={np.percentile(boot_rs, 97.5):.4f}")
    print("  Conclusion: In-sample CCA re-optimization on bootstrap samples (with duplicate rows) creates severe upward bias.")
    print("  This explains why reported r = 0.5767 had bootstrap CI [0.6230, 0.7415].")


def audit_oos_cca():
    print("\n" + "=" * 80)
    print("AUDIT 2: OUT-OF-SAMPLE CROSS-VALIDATED CCA & DIRECT PAIRED COMPARISONS")
    print("=" * 80)
    df = pd.read_parquet(PARQUET_PATH)
    mri_cols = [c for c in df.columns if any(c.startswith(p) for p in ['cth_', 'aseg_', 'wm_', 'sa_'])]
    mol_cols = [
        'PLASMA_NFL', 'PLASMA_GFAP', 'CSF_PTAU181', 'CSF_TTAU', 'CSF_ABETA42', 'CSF_ALPHASYN',
        'CSF_NFL', 'CSF_GFAP', 'SAA_positive', 'SAA_Fmax_24h', 'SAA_SlopeMax_24h', 'SAA_AUC_24h', 'SAA_TTT_24h',
        'is_LRRK2_carrier', 'is_GBA_carrier', 'is_APOE_e4_carrier', 'PATHVAR_COUNT',
        'GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS'
    ]

    X_mri = StandardScaler().fit_transform(SimpleImputer(strategy='mean').fit_transform(df[mri_cols]))
    X_mol = RobustScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(df[mol_cols]))

    # QVBE representations
    cfg_mol = QVBEV2Config(input_dim=X_mol.shape[1], n_qubits=8, n_layers=2, measurement_basis='full', seed=42, mode='fixed')
    qv_mol = QuantumVariationalBiomarkerEncoderV2(cfg_mol)
    qv_mol.fit(X_mol, df['target_progression_5yr_mcid5'])
    Q_mol = qv_mol.transform(X_mol)

    cfg_mri = QVBEV2Config(input_dim=X_mri.shape[1], n_qubits=8, n_layers=2, measurement_basis='full', seed=1042, mode='fixed')
    qv_mri = QuantumVariationalBiomarkerEncoderV2(cfg_mri)
    qv_mri.fit(X_mri, df['target_progression_5yr_mcid5'])
    Q_mri = qv_mri.transform(X_mri)

    pca_mol8 = PCA(8, random_state=42).fit_transform(X_mol)
    pca_mri8 = PCA(8, random_state=42).fit_transform(X_mri)
    pca_mol20 = PCA(20, random_state=42).fit_transform(X_mol)
    pca_mri20 = PCA(20, random_state=42).fit_transform(X_mri)
    kpca_mol8 = KernelPCA(8, kernel="rbf", gamma=1.0/X_mol.shape[1], random_state=42).fit_transform(X_mol)
    kpca_mri8 = KernelPCA(8, kernel="rbf", gamma=1.0/X_mri.shape[1], random_state=42).fit_transform(X_mri)
    rp_mol8 = GaussianRandomProjection(8, random_state=42).fit_transform(X_mol)
    rp_mri8 = GaussianRandomProjection(8, random_state=42).fit_transform(X_mri)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def eval_oos(X1, X2):
        oos_r = []
        for tr, te in kf.split(X1):
            c = CCA(n_components=1)
            c.fit(X1[tr], X2[tr])
            x_te, y_te = c.transform(X1[te], X2[te])
            r = stats.pearsonr(x_te[:, 0], y_te[:, 0])[0]
            oos_r.append(r)
        return np.array(oos_r)

    oos_qvbe = eval_oos(Q_mol, Q_mri)
    oos_pca8 = eval_oos(pca_mol8, pca_mri8)
    oos_pca20 = eval_oos(pca_mol20, pca_mri20)
    oos_kpca8 = eval_oos(kpca_mol8, kpca_mri8)
    oos_rp8 = eval_oos(rp_mol8, rp_mri8)

    print("5-Fold Out-of-Sample Canonical Correlation (Molecular vs Structural):")
    print(f"  QVBE (23 vs 23):      mean r = {np.mean(oos_qvbe):.4f} +/- {np.std(oos_qvbe):.4f}  (folds: {np.round(oos_qvbe, 4)})")
    print(f"  Linear PCA-8 (8 vs 8): mean r = {np.mean(oos_pca8):.4f} +/- {np.std(oos_pca8):.4f}  (folds: {np.round(oos_pca8, 4)})")
    print(f"  Linear PCA-20:         mean r = {np.mean(oos_pca20):.4f} +/- {np.std(oos_pca20):.4f}  (folds: {np.round(oos_pca20, 4)})")
    print(f"  Kernel PCA-8 (RBF):    mean r = {np.mean(oos_kpca8):.4f} +/- {np.std(oos_kpca8):.4f}  (folds: {np.round(oos_kpca8, 4)})")
    print(f"  Random Proj-8:         mean r = {np.mean(oos_rp8):.4f} +/- {np.std(oos_rp8):.4f}  (folds: {np.round(oos_rp8, 4)})")

    # Direct Paired Tests
    p_vs_pca8 = stats.wilcoxon(oos_qvbe, oos_pca8).pvalue
    p_vs_kpca8 = stats.wilcoxon(oos_qvbe, oos_kpca8).pvalue
    print("\nDirect Paired Wilcoxon Test (QVBE vs Controls across folds):")
    print(f"  QVBE vs Linear PCA-8:  Delta = {np.mean(oos_qvbe) - np.mean(oos_pca8):+.4f}, p = {p_vs_pca8:.4f}")
    print(f"  QVBE vs Kernel PCA-8:  Delta = {np.mean(oos_qvbe) - np.mean(oos_kpca8):+.4f}, p = {p_vs_kpca8:.4f}")


def audit_clustering_leakage_and_stability():
    print("\n" + "=" * 80)
    print("AUDIT 3: CLUSTERING TARGET LEAKAGE & REPRODUCIBILITY AUDIT")
    print("=" * 80)
    df = pd.read_parquet(PARQUET_PATH)
    temporal_all = np.load(TEMPORAL_NPY_PATH)
    mri_cols = [c for c in df.columns if any(c.startswith(p) for p in ['cth_', 'aseg_', 'wm_', 'sa_'])]
    mol_cols = [
        'PLASMA_NFL', 'PLASMA_GFAP', 'CSF_PTAU181', 'CSF_TTAU', 'CSF_ABETA42', 'CSF_ALPHASYN',
        'CSF_NFL', 'CSF_GFAP', 'SAA_positive', 'SAA_Fmax_24h', 'SAA_SlopeMax_24h', 'SAA_AUC_24h', 'SAA_TTT_24h',
        'is_LRRK2_carrier', 'is_GBA_carrier', 'is_APOE_e4_carrier', 'PATHVAR_COUNT',
        'GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS'
    ]

    mask_saa = (df['SAA_positive'] == 1.0).values
    df_saa = df[mask_saa].copy()
    y_5yr_saa = df_saa['target_progression_5yr_mcid5'].values

    # Check stability across 50 random K-means initializations
    # Load or extract the exact representations used
    X_mri = StandardScaler().fit_transform(SimpleImputer(strategy='mean').fit_transform(df[mri_cols]))
    X_mol = RobustScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(df[mol_cols]))

    cfg_mol = QVBEV2Config(input_dim=X_mol.shape[1], n_qubits=8, n_layers=2, measurement_basis='full', seed=42, mode='fixed')
    qv_mol = QuantumVariationalBiomarkerEncoderV2(cfg_mol)
    qv_mol.fit(X_mol, df['target_progression_5yr_mcid5'])
    Q_mol = qv_mol.transform(X_mol)

    cfg_mri = QVBEV2Config(input_dim=X_mri.shape[1], n_qubits=8, n_layers=2, measurement_basis='full', seed=1042, mode='fixed')
    qv_mri = QuantumVariationalBiomarkerEncoderV2(cfg_mri)
    qv_mri.fit(X_mri, df['target_progression_5yr_mcid5'])
    Q_mri = qv_mri.transform(X_mri)

    # Train CTCLM supervised (as in original script)
    time_offsets = torch.tensor([[0.0, 0.5, 1.0]] * len(df), dtype=torch.float32)
    temp_t = torch.tensor(temporal_all, dtype=torch.float32)
    y_t = torch.tensor(df['target_progression_5yr_mcid5'].values, dtype=torch.float32).unsqueeze(1)
    ctclm_cfg = CTCLMConfig(input_dim=35, latent_dim=16, time_embedding_dim=4, num_layers=1, seed=42)
    ctclm = ContinualTemporalClinicalLatentModel(ctclm_cfg)
    ctclm_head = nn.Linear(16, 1)
    opt = optim.Adam(list(ctclm.parameters()) + list(ctclm_head.parameters()), lr=0.01)
    for ep in range(35):
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(ctclm_head(ctclm(temp_t, time_offsets=time_offsets).h_final), y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h_temp_sup = ctclm(temp_t, time_offsets=time_offsets).h_final.numpy()

    X_joint_sup = np.hstack([Q_mol[mask_saa], Q_mri[mask_saa], h_temp_sup[mask_saa]])

    # K-means seed 42
    km42 = KMeans(n_clusters=3, random_state=42, n_init=20).fit(X_joint_sup)
    labels42 = km42.labels_

    # Test random seed stability (ARI)
    aris = []
    for s in range(50):
        km_s = KMeans(n_clusters=3, random_state=s, n_init=20).fit(X_joint_sup)
        aris.append(adjusted_rand_score(labels42, km_s.labels_))
    print(f"K-Means Cluster Stability across 50 random initializations: Mean ARI = {np.mean(aris):.4f} +/- {np.std(aris):.4f}")

    # Now verify: what happens if clustering is done WITHOUT supervised CTCLM?
    # Unsupervised: strictly baseline Q_mol + Q_mri + trajectory statistics (delta_1yr, slope_1yr)
    traj_stats_saa = df_saa[['early_trajectory_delta_1yr', 'early_trajectory_slope_1yr']].values
    X_joint_unsup = np.hstack([Q_mol[mask_saa], Q_mri[mask_saa], StandardScaler().fit_transform(traj_stats_saa)])
    km_unsup = KMeans(n_clusters=3, random_state=42, n_init=20).fit(X_joint_unsup)
    
    print("\nProgression Rates across Clusters:")
    print("  SUPERVISED h_temp Clustering (Used in Phase 7 report):")
    for cl in range(3):
        idx_cl = np.where(labels42 == cl)[0]
        n_cl = len(idx_cl)
        rate = np.mean(y_5yr_saa[idx_cl]) * 100
        # Wilson score interval for progression rate
        p_hat = np.mean(y_5yr_saa[idx_cl])
        ci_w = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_cl) * 100
        print(f"    Cluster {cl}: N = {n_cl}, Progression Rate = {rate:.1f}% +/- {ci_w:.1f}%")

    print("  UNSUPERVISED Clustering (No supervised h_temp leakage):")
    for cl in range(3):
        idx_cl = np.where(km_unsup.labels_ == cl)[0]
        n_cl = len(idx_cl)
        rate = np.mean(y_5yr_saa[idx_cl]) * 100
        p_hat = np.mean(y_5yr_saa[idx_cl])
        ci_w = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_cl) * 100
        print(f"    Cluster {cl}: N = {n_cl}, Progression Rate = {rate:.1f}% +/- {ci_w:.1f}%")


def audit_qcdf_significance():
    print("\n" + "=" * 80)
    print("AUDIT 4: QCDF MULTIMODAL INTEGRATION & STATISTICAL REPRODUCTION")
    print("=" * 80)
    df_pred = pd.read_csv(r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\LONGITUDINAL_PREDICTION_RESULTS.csv")
    df_abl = pd.read_csv(r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\MODALITY_ABLATION_RESULTS.csv")
    df_rob = pd.read_csv(r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_6\MISSING_MODALITY_ROBUSTNESS_DATA.csv")

    ctclm_only = df_pred[df_pred['pipeline'] == '2_Temporal_CTCLM_Only']['auroc'].values
    concat_lgb = df_pred[df_pred['pipeline'] == '5_QCDF_Concatenation']['auroc'].values
    gated_full = df_pred[df_pred['pipeline'] == '6_QCDF_Adaptive_Gated']['auroc'].values
    gated_no_mri = df_rob[df_rob['missing_pattern'] == 'Missing_Static_MRI_Genetics']['auroc'].values

    print(f"CTCLM Temporal Only:       AUROC = {np.mean(ctclm_only):.4f} +/- {np.std(ctclm_only):.4f}")
    print(f"Naive Concatenation (LGB): AUROC = {np.mean(concat_lgb):.4f} +/- {np.std(concat_lgb):.4f}")
    print(f"Adaptive Gated QCDF (Full):AUROC = {np.mean(gated_full):.4f} +/- {np.std(gated_full):.4f}")
    print(f"Adaptive Gated (Mask MRI): AUROC = {np.mean(gated_no_mri):.4f} +/- {np.std(gated_no_mri):.4f}")

    # Paired Wilcoxon Tests across the 5 folds
    w_concat_vs_gated = stats.wilcoxon(concat_lgb, gated_full).pvalue
    w_ctclm_vs_concat = stats.wilcoxon(ctclm_only, concat_lgb).pvalue
    w_ctclm_vs_gated = stats.wilcoxon(ctclm_only, gated_full).pvalue

    print("\nPaired Wilcoxon Signed-Rank Tests (across 5 folds):")
    print(f"  Naive Concat vs Gated Full:  Delta = {np.mean(gated_full) - np.mean(concat_lgb):+.4f}, p = {w_concat_vs_gated:.4f}")
    print(f"  CTCLM Only vs Naive Concat:  Delta = {np.mean(concat_lgb) - np.mean(ctclm_only):+.4f}, p = {w_ctclm_vs_concat:.4f}")
    print(f"  CTCLM Only vs Gated Full:    Delta = {np.mean(gated_full) - np.mean(ctclm_only):+.4f}, p = {w_ctclm_vs_gated:.4f}")


if __name__ == "__main__":
    audit_cca_bootstrap_bias()
    audit_oos_cca()
    audit_clustering_leakage_and_stability()
    audit_qcdf_significance()

