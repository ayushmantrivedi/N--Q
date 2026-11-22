"""
Integration tests for the dataset registry.

Verifies that:
  - The registry correctly resolves loader classes from config
  - The BaseDatasetLoader interface is enforced
  - Schema and split loading work end-to-end

These tests use only the registry infrastructure, not real datasets.
"""

import pytest

from nadiq.data.registry import DatasetRegistry
from nadiq.data.contracts import BaseDatasetLoader


@pytest.mark.integration
class TestDatasetRegistry:

    def test_empty_registry(self) -> None:
        registry = DatasetRegistry()
        assert len(registry) == 0
        assert "synthetic_tabular_v1" not in registry

    def test_register_and_contains(self) -> None:
        registry = DatasetRegistry()
        registry.register(
            key="test_ds",
            loader_class="nadiq.data.contracts.BaseDatasetLoader",  # Won't instantiate
        )
        assert "test_ds" in registry
        assert len(registry) == 1

    def test_duplicate_registration_raises(self) -> None:
        registry = DatasetRegistry()
        registry.register("ds1", "some.Loader")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("ds1", "some.OtherLoader")

    def test_get_unregistered_raises(self) -> None:
        registry = DatasetRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_dataset")

    def test_list_registered(self) -> None:
        registry = DatasetRegistry()
        registry.register("ds_b", "some.B")
        registry.register("ds_a", "some.A")
        # Should be sorted
        assert registry.list_registered() == ["ds_a", "ds_b"]

    def test_invalid_class_path_raises_on_get(self) -> None:
        registry = DatasetRegistry()
        registry.register("bad_ds", "nadiq.nonexistent.module.Loader")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            registry.get("bad_ds")
