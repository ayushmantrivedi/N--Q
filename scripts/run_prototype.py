import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import sys

sys.path.append(r'C:\Users\ayush\OneDrive\Desktop\Nadi-Q\src')

from nadiq.models.mri.bvae import BetaVAE, BetaVAEConfig
from nadiq.models.quantum.qvbe import QuantumVariationalBiomarkerEncoder, QVBEConfig
from nadiq.models.fusion.qcdf import QCDF, QCDFConfig
from nadiq.models.representation import ModalityRepresentation, ModalityBundle, RepresentationType

def run_experiment(seed, run_name):
    print(f"--- Running {run_name} (Seed {seed}) ---")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    EXP_DIR = r'F:\NADI-Q data\PPMI\experiments\prototype_v1'
    os.makedirs(EXP_DIR, exist_ok=True)
    
    # 1. Load Manifest
    df = pd.read_csv(r'F:\NADI-Q data\PPMI\manifests\ppmi_real_multimodal_preprocessed_v1.csv')
    
    # Features
    bio_cols = ['mtbr_tau243', 'abeta40', 'abeta42', 'ptau181', 'nfl', 'gfap', 'ptau217', 'csf_saa']
    gait_cols = [c for c in df.columns if '_U' in c or '_DT' in c or '_OP' in c or '_CL' in c or 'TUG' in c]
    
    # Preprocess targets
    target_map = {'HC': 0, 'PRODROMAL': 1, 'PD': 2}
    y = df['clinical_state'].map(target_map).values
    
    # Extract Bio and Gait matrices
    X_bio = df[bio_cols].values
    X_gait = df[gait_cols].values
    
    # Simple imputation for NaN (classical baseline + standard prep)
    bio_imputer = SimpleImputer(strategy='median')
    X_bio_imp = bio_imputer.fit_transform(X_bio)
    
    gait_imputer = SimpleImputer(strategy='median')
    X_gait_imp = gait_imputer.fit_transform(X_gait)
    
    # Load MRI
    X_mri = []
    for patno in df['PATNO']:
        npy_path = rf"F:\NADI-Q data\PPMI\MRI_processed_v2\{patno}\{patno}_processed.npy"
        mri_arr = np.load(npy_path)
        X_mri.append(mri_arr)
    X_mri = np.stack(X_mri, axis=0) # (15, 128, 128, 128)
    X_mri = np.expand_dims(X_mri, axis=1) # (15, 1, 128, 128, 128)
    
    # Split
    indices = np.arange(len(df))
    # Note: With N=15 and PD=1, stratify might fail. We just do random split.
    train_idx, val_idx = train_test_split(indices, test_size=5, random_state=seed)
    
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    # Tensors
    t_mri_tr = torch.tensor(X_mri[train_idx], dtype=torch.float32)
    t_mri_va = torch.tensor(X_mri[val_idx], dtype=torch.float32)
    
    t_bio_tr = torch.tensor(X_bio_imp[train_idx], dtype=torch.float32)
    t_bio_va = torch.tensor(X_bio_imp[val_idx], dtype=torch.float32)
    
    t_gait_tr = torch.tensor(X_gait_imp[train_idx], dtype=torch.float32)
    t_gait_va = torch.tensor(X_gait_imp[val_idx], dtype=torch.float32)
    
    y_tr = torch.tensor(y[train_idx], dtype=torch.long)
    y_va = torch.tensor(y[val_idx], dtype=torch.long)
    
    # A. Train Beta-VAE (MRI)
    print("Training compact Beta-VAE...")
    bvae_cfg = BetaVAEConfig(input_shape=(1, 128, 128, 128), latent_dim=16, beta=4.0, seed=seed)
    bvae = BetaVAE(bvae_cfg)
    optimizer_bvae = optim.Adam(bvae.parameters(), lr=1e-3)
    
    bvae.train()
    for ep in range(5): # Very short for N=15
        optimizer_bvae.zero_grad()
        out = bvae(t_mri_tr)
        out.total_loss.backward()
        optimizer_bvae.step()
        
    bvae.eval()
    with torch.no_grad():
        out_tr = bvae(t_mri_tr)
        out_va = bvae(t_mri_va)
        print(f"VAE Recon Loss Tr: {out_tr.recon_loss.item():.4f}, Va: {out_va.recon_loss.item():.4f}")
        latent_mri_tr = out_tr.z
        latent_mri_va = out_va.z
        
    # B. Initialize QVBEs and QCDF
    print("Initializing Quantum Encoders and Fusion...")
    qvbe_mri = QuantumVariationalBiomarkerEncoder(QVBEConfig(input_dim=16, n_qubits=4, n_layers=2, seed=seed))
    
    # We must use the actual shape after imputation because SimpleImputer drops completely empty columns
    bio_in_dim = t_bio_tr.shape[1]
    qvbe_bio = QuantumVariationalBiomarkerEncoder(QVBEConfig(input_dim=bio_in_dim, n_qubits=4, n_layers=2, seed=seed))
    
    # Gait is purely classical as requested if single-visit
    gait_encoder = nn.Linear(t_gait_tr.shape[1], 32)
    
    qcdf = QCDF(QCDFConfig(mri_dim=4, biomarker_dim=4, kinematics_dim=32, num_classes=3, seed=seed))
    
    all_params = list(qvbe_mri.parameters()) + list(qvbe_bio.parameters()) + list(gait_encoder.parameters()) + list(qcdf.parameters())
    optimizer_fusion = optim.Adam(all_params, lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    def forward_bundle(latent_m, raw_b, raw_g, mask_m=True, mask_b=True, mask_g=True):
        emb_m = qvbe_mri(latent_m).embedding if mask_m else torch.zeros(latent_m.shape[0], 4)
        emb_b = qvbe_bio(raw_b).embedding if mask_b else torch.zeros(raw_b.shape[0], 4)
        emb_g = gait_encoder(raw_g) if mask_g else torch.zeros(raw_g.shape[0], 32)
        
        bundle = ModalityBundle(
            mri=ModalityRepresentation(modality="mri", embedding=emb_m, representation_type=RepresentationType.QUANTUM, available=mask_m),
            biomarkers=ModalityRepresentation(modality="biomarkers", embedding=emb_b, representation_type=RepresentationType.QUANTUM, available=mask_b),
            kinematics=ModalityRepresentation(modality="kinematics", embedding=emb_g, representation_type=RepresentationType.CLASSICAL_STATIC, available=mask_g)
        )
        return qcdf(bundle)
        
    print("Training Fusion...")
    # Train End-to-End
    for ep in range(10):
        optimizer_fusion.zero_grad()
        out = forward_bundle(latent_mri_tr, t_bio_tr, t_gait_tr)
        loss = criterion(out.logits, y_tr)
        loss.backward(retain_graph=True)
        optimizer_fusion.step()
        
    # Eval
    with torch.no_grad():
        out_va_full = forward_bundle(latent_mri_va, t_bio_va, t_gait_va)
        val_loss = criterion(out_va_full.logits, y_va).item()
        preds = out_va_full.logits.argmax(dim=-1)
        acc = (preds == y_va).float().mean().item()
        
    print(f"Full Fusion Val Loss: {val_loss:.4f}, Acc: {acc:.4f}")
    
    # C. Ablations
    print("Running Ablations...")
    metrics = {"val_acc_full": acc}
    with torch.no_grad():
        # MRI Only
        out = forward_bundle(latent_mri_va, t_bio_va, t_gait_va, mask_b=False, mask_g=False)
        metrics["val_acc_mri_only"] = (out.logits.argmax(dim=-1) == y_va).float().mean().item()
        # Bio Only
        out = forward_bundle(latent_mri_va, t_bio_va, t_gait_va, mask_m=False, mask_g=False)
        metrics["val_acc_bio_only"] = (out.logits.argmax(dim=-1) == y_va).float().mean().item()
        # Gait Only
        out = forward_bundle(latent_mri_va, t_bio_va, t_gait_va, mask_m=False, mask_b=False)
        metrics["val_acc_gait_only"] = (out.logits.argmax(dim=-1) == y_va).float().mean().item()
        # MRI + Bio
        out = forward_bundle(latent_mri_va, t_bio_va, t_gait_va, mask_g=False)
        metrics["val_acc_mri_bio"] = (out.logits.argmax(dim=-1) == y_va).float().mean().item()
        # MRI + Gait
        out = forward_bundle(latent_mri_va, t_bio_va, t_gait_va, mask_b=False)
        metrics["val_acc_mri_gait"] = (out.logits.argmax(dim=-1) == y_va).float().mean().item()
        
    # D. Classical Baselines
    X_tr_flat = np.hstack([latent_mri_tr.numpy(), t_bio_tr.numpy(), t_gait_tr.numpy()])
    X_va_flat = np.hstack([latent_mri_va.numpy(), t_bio_va.numpy(), t_gait_va.numpy()])
    
    clf = LogisticRegression(max_iter=1000)
    # Check if more than 1 class exists in train (since N=10, maybe only 1 class randomly picked!)
    if len(np.unique(y_tr.numpy())) > 1:
        clf.fit(X_tr_flat, y_tr.numpy())
        metrics["val_acc_classical_lr"] = clf.score(X_va_flat, y_va.numpy())
    else:
        metrics["val_acc_classical_lr"] = "N/A (Single class in train)"
        
    # Artifacts
    if run_name == "Run_1":
        with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save latents to prove execution
        torch.save({
            "latent_mri": latent_mri_va,
            "q_embed_mri": qvbe_mri(latent_mri_va).embedding,
            "q_embed_bio": qvbe_bio(t_bio_va).embedding,
            "y_true": y_va,
            "patnos_val": df.iloc[val_idx]['PATNO'].values.tolist()
        }, os.path.join(EXP_DIR, "artifacts.pt"))
        
    return metrics, latent_mri_va.numpy()

if __name__ == "__main__":
    m1, l1 = run_experiment(seed=42, run_name="Run_1")
    m2, l2 = run_experiment(seed=42, run_name="Run_2")
    
    # Assert Reproducibility
    assert np.allclose(l1, l2, atol=1e-5), "Run 1 and Run 2 latents are not identical. Reproducibility failed."
    print("Reproducibility verified.")
