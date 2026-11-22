"""
Tests for ADNI MRI Adapter / Locator.
"""

import pytest
from pathlib import Path
from nadiq.data.sources.adni import ADNIMRILocator

def test_adni_locator_missing_dir(tmp_path):
    locator = ADNIMRILocator(tmp_path / "nonexistent")
    assert locator.find_mri_payload("002_S_0413", "2006-05-19") is None

def test_adni_locator_mock_dir(tmp_path):
    locator = ADNIMRILocator(tmp_path)
    
    # Even if directory exists, V1 locator currently returns None
    # since payloads are missing. It should fail gracefully.
    assert locator.find_mri_payload("002_S_0413", "2006-05-19") is None

def test_adni_locator_metadata_status():
    locator = ADNIMRILocator("F:/NADI-Q data")
    status = locator.get_metadata_status()
    
    assert not status["payloads_available"]
    assert "Awaiting manual download" in status["note"]
