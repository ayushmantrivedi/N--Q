"""
Tests for MRI Preprocessing Interface.
"""

import pytest
import torch
from nadiq.preprocessing.mri import MRIPreprocessor, MRIPreprocessingConfig

def test_mri_preprocessing_shape():
    config = MRIPreprocessingConfig(target_shape=(16, 16, 16), normalize_intensity=True)
    preprocessor = MRIPreprocessor(config)
    
    # 5D tensor: (B, C, D, H, W)
    raw = torch.randn(2, 1, 32, 32, 32)
    processed = preprocessor(raw)
    
    assert processed.shape == (2, 1, 16, 16, 16)

def test_mri_preprocessing_intensity_normalization():
    config = MRIPreprocessingConfig(target_shape=(32, 32, 32), normalize_intensity=True)
    preprocessor = MRIPreprocessor(config)
    
    raw = torch.randn(2, 1, 32, 32, 32) * 5.0 + 10.0
    processed = preprocessor(raw)
    
    mean = processed.mean(dim=(2, 3, 4))
    std = processed.std(dim=(2, 3, 4), unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)

def test_mri_preprocessing_finite():
    config = MRIPreprocessingConfig(target_shape=(16, 16, 16))
    preprocessor = MRIPreprocessor(config)
    
    # Zeros should not cause NaNs during normalization
    raw = torch.zeros(2, 1, 32, 32, 32)
    processed = preprocessor(raw)
    
    assert torch.isfinite(processed).all()
