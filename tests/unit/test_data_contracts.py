"""
Unit tests for nadiq.data.contracts.

Tests that the data contract structures:
  - Validate inputs correctly
  - Raise appropriate errors for invalid data
  - Maintain the interface invariants documented in data_contract.md
"""

import pytest
import numpy as np

from nadiq.data.contracts import (
    ClinicalLabel,
    ClinicalSample,
    DataSplit,
    FeatureBlock,
    FeatureDtype,
    LabelType,
    ModalityType,
    SampleMetadata,
)


# ---------------------------------------------------------------------------
# FeatureBlock tests
# ---------------------------------------------------------------------------


class TestFeatureBlock:

    def test_valid_tabular_block(self) -> None:
        """A valid tabular feature block should be constructed without error."""
        values = np.zeros((32,), dtype=np.float32)
        names = [f"feature_{i}" for i in range(32)]
        block = FeatureBlock(
            modality=ModalityType.TABULAR,
            values=values,
            feature_names=names,
        )
        assert block.modality == ModalityType.TABULAR
        assert block.values.shape == (32,)
        assert len(block.feature_names) == 32
        assert block.missing_mask is None

    def test_mismatched_feature_names_raises(self) -> None:
        """Mismatched feature_names and values dimensions must raise ValueError."""
        values = np.zeros((32,))
        names = [f"feature_{i}" for i in range(10)]  # Wrong length
        with pytest.raises(ValueError, match="feature_names length"):
            FeatureBlock(
                modality=ModalityType.TABULAR,
                values=values,
                feature_names=names,
            )

    def test_mismatched_missing_mask_raises(self) -> None:
        """Mismatched missing_mask shape must raise ValueError."""
        values = np.zeros((32,))
        names = [f"f_{i}" for i in range(32)]
        mask = np.zeros((16,), dtype=bool)  # Wrong shape
        with pytest.raises(ValueError, match="missing_mask shape"):
            FeatureBlock(
                modality=ModalityType.TABULAR,
                values=values,
                feature_names=names,
                missing_mask=mask,
            )

    def test_valid_missing_mask(self) -> None:
        """A correctly shaped missing mask should be accepted."""
        values = np.zeros((32,))
        names = [f"f_{i}" for i in range(32)]
        mask = np.zeros((32,), dtype=bool)
        mask[5] = True  # Feature 5 is missing
        block = FeatureBlock(
            modality=ModalityType.TABULAR,
            values=values,
            feature_names=names,
            missing_mask=mask,
        )
        assert block.missing_mask is not None
        assert block.missing_mask[5] == True  # noqa: E712 — np.True_ == True is correct

    def test_time_series_block_shape(self) -> None:
        """A time-series block has shape (n_timesteps, n_features)."""
        values = np.zeros((100, 5))  # 100 timesteps, 5 features
        names = [f"vital_{i}" for i in range(5)]
        block = FeatureBlock(
            modality=ModalityType.TIME_SERIES,
            values=values,
            feature_names=names,
        )
        assert block.values.shape == (100, 5)


# ---------------------------------------------------------------------------
# ClinicalSample tests
# ---------------------------------------------------------------------------


class TestClinicalSample:

    def _make_sample(self) -> ClinicalSample:
        values = np.zeros((32,))
        names = [f"f_{i}" for i in range(32)]
        block = FeatureBlock(
            modality=ModalityType.TABULAR,
            values=values,
            feature_names=names,
        )
        return ClinicalSample(
            sample_id="test_001",
            features={"tabular": block},
            label=ClinicalLabel(
                task="binary_risk",
                value=0,
                label_type=LabelType.BINARY,
            ),
            metadata=SampleMetadata(
                dataset_id="synthetic_tabular_v1",
                split=DataSplit.TRAIN,
            ),
        )

    def test_sample_construction(self) -> None:
        sample = self._make_sample()
        assert sample.sample_id == "test_001"
        assert "tabular" in sample.features
        assert sample.label is not None
        assert sample.label.task == "binary_risk"

    def test_get_existing_modality(self) -> None:
        sample = self._make_sample()
        block = sample.get_modality("tabular")
        assert block.modality == ModalityType.TABULAR

    def test_get_missing_modality_raises(self) -> None:
        sample = self._make_sample()
        with pytest.raises(KeyError, match="imaging"):
            sample.get_modality("imaging")

    def test_modalities_property(self) -> None:
        sample = self._make_sample()
        assert sample.modalities == ["tabular"]

    def test_sample_id_is_string(self) -> None:
        """sample_id must be a string — no integer IDs that could be patient IDs."""
        sample = self._make_sample()
        assert isinstance(sample.sample_id, str)


# ---------------------------------------------------------------------------
# Seeding tests (import from utils)
# ---------------------------------------------------------------------------


class TestSeeding:

    def test_seed_all_sets_numpy(self) -> None:
        """seed_all should produce reproducible numpy samples."""
        from nadiq.utils.seeding import seed_all

        seed_all(42)
        a = np.random.rand(10)

        seed_all(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)

    def test_seed_out_of_range_raises(self) -> None:
        from nadiq.utils.seeding import seed_all

        with pytest.raises(ValueError, match="Seed must be in"):
            seed_all(-1)

        with pytest.raises(ValueError, match="Seed must be in"):
            seed_all(2**32)

    def test_seed_state_context_restores(self) -> None:
        """SeedState context manager should restore global random state."""
        from nadiq.utils.seeding import SeedState, seed_all
        import random

        seed_all(0)
        before = random.random()

        seed_all(0)
        with SeedState(seed=99):
            _ = random.random()  # Consume some randomness inside context

        after = random.random()
        assert abs(before - after) < 1e-10, "Random state should be restored after context"
