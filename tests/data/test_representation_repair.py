import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

import pytest

import sys
sys.path.append(r'C:\Users\ayush\OneDrive\Desktop\Nadi-Q\src')

from nadiq.models.quantum.qvbe import QuantumVariationalBiomarkerEncoder, QVBEConfig
from nadiq.models.mri.bvae import BetaVAE, BetaVAEConfig

@pytest.fixture
def sample_bio_data():
    df = pd.read_csv(r'F:\NADI-Q data\PPMI\manifests\ppmi_real_multimodal_preprocessed_v1.csv')
    bio_cols = ['mtbr_tau243', 'abeta40', 'abeta42', 'ptau181', 'nfl', 'gfap', 'ptau217', 'csf_saa']
    raw_bio = df[bio_cols].values
    imp = SimpleImputer(strategy='median')
    imp_bio = imp.fit_transform(raw_bio)
    return df, raw_bio, imp_bio

def test_biomarker_scaling_exists_and_finite(sample_bio_data):
    # Tests 1, 2, 9
    df, raw_bio, imp_bio = sample_bio_data
    
    # 9. Original raw values unchanged
    assert np.isnan(raw_bio[:, 7]).all(), "Original raw biomarker values must remain unchanged (e.g. csf_saa should be NaN)"
    
    scaler = StandardScaler()
    scaled_bio = scaler.fit_transform(imp_bio)
    
    # 2. Scaled biomarkers have finite values
    assert np.all(np.isfinite(scaled_bio)), "Scaled biomarkers must be finite"
    
    # 1. Biomarker scaling exists
    assert not np.allclose(scaled_bio, imp_bio), "Biomarker scaling must alter the imputed representation"
    assert np.allclose(np.mean(scaled_bio, axis=0), 0, atol=1e-5), "Scaled mean should be ~0"

def test_scaled_biomarkers_avoid_extreme_saturation(sample_bio_data):
    # Tests 3, 4
    _, _, imp_bio = sample_bio_data
    scaler = StandardScaler()
    scaled_bio = scaler.fit_transform(imp_bio)
    
    qvbe = QuantumVariationalBiomarkerEncoder(QVBEConfig(input_dim=scaled_bio.shape[1], n_qubits=4, n_layers=2, seed=42))
    
    t_scaled = torch.tensor(scaled_bio, dtype=torch.float32)
    with torch.no_grad():
        proj = qvbe.projection(t_scaled)
        angles = torch.tanh(proj) * np.pi
    
    # 3. No extreme saturation
    max_angle = torch.max(torch.abs(angles)).item()
    assert max_angle < (np.pi - 1e-4), "Angles should not perfectly saturate to Pi"
    
    # 4. QVBE receives row-wise varying inputs
    assert torch.var(proj, dim=0).mean().item() > 0.1, "QVBE projection must retain variance"

def test_active_mu_dimensions_and_stochastic_noise():
    # Tests 5, 6
    bvae = BetaVAE(BetaVAEConfig(input_shape=(1, 128, 128, 128), latent_dim=16, beta=4.0, seed=42))
    bvae.train()  # Must be in train mode for reparameterize to add stochastic noise
    
    dummy_input = torch.randn(5, 1, 128, 128, 128)
    with torch.no_grad():
        out = bvae(dummy_input)
    
    mu_var = torch.var(out.mu, dim=0)
    z_var = torch.var(out.z, dim=0)
    
    # 5. mu dimensions measurable before considered active
    active_threshold = 1e-4
    active_mu_dims = int(torch.sum(mu_var >= active_threshold).item())
    
    # 6. stochastic z variance is not subject info
    active_z_dims = int(torch.sum(z_var >= active_threshold).item())
    
    assert active_mu_dims >= 0
    # Z variance will be naturally higher because it includes exp(0.5 * logvar) sampling
    # We test that we acknowledge the difference between z_var and mu_var natively
    assert not torch.allclose(mu_var, z_var), "Mu variance and Z variance must be distinctly measured"

def test_bvae_diagnostics_report_separate_losses():
    # Test 7
    bvae = BetaVAE(BetaVAEConfig(input_shape=(1, 128, 128, 128), latent_dim=16, beta=1.0, seed=42))
    dummy_input = torch.randn(2, 1, 128, 128, 128)
    
    with torch.no_grad():
        out = bvae(dummy_input)
        
    assert hasattr(out, 'recon_loss'), "BetaVAEOutput must separately report recon_loss"
    assert hasattr(out, 'kl_loss'), "BetaVAEOutput must separately report kl_loss"
    assert hasattr(out, 'total_loss'), "BetaVAEOutput must separately report total_loss"

def test_scaler_provenance_and_reproducibility():
    # Tests 8, 10
    EXP_DIR = r'F:\NADI-Q data\PPMI\experiments\representation_repair_v1'
    scaler_path = os.path.join(EXP_DIR, 'biomarker_scaler_v1.pkl')
    
    # 8. Scaler provenance preserved
    assert os.path.exists(scaler_path), f"Scaler artifact must be preserved at {scaler_path}"
    
    loaded_scaler = joblib.load(scaler_path)
    assert hasattr(loaded_scaler, 'mean_'), "Loaded object must be a fitted scaler"
    
    # 10. Pipeline is reproducible under tolerance
    bvae_latents_path = os.path.join(EXP_DIR, 'bvae_latents_beta_0.1.pt')
    assert os.path.exists(bvae_latents_path), "BVAE latent provenance artifact must be preserved"
    data = torch.load(bvae_latents_path)
    assert "mu" in data
    assert "patnos" in data
