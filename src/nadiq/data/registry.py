"""
Dataset registry for Nāḍī-Q.

Resolves dataset IDs (strings from config) to concrete loader instances.
Models never import concrete loaders; they receive them through this registry.

Usage:
    from nadiq.data.registry import DatasetRegistry

    registry = DatasetRegistry.from_config_dir("configs/data/")
    loader = registry.get("synthetic_tabular_v1")
    schema = loader.get_schema()
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

from nadiq.data.contracts import BaseDatasetLoader

__all__ = ["DatasetRegistry"]


class DatasetRegistry:
    """
    A registry that maps dataset IDs to loader instances.

    Loaders are registered via configuration files that specify
    the loader class path and constructor arguments.

    This class does NOT instantiate loaders at registration time —
    loaders are created on demand to avoid loading data unnecessarily.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}

    def register(self, key: str, loader_class: str, **kwargs: Any) -> None:
        """
        Register a dataset loader by class path and configuration.

        Args:
            key:          Unique string key for this dataset.
            loader_class: Dotted import path to the loader class.
                          e.g., 'nadiq.data.loaders.synthetic.SyntheticTabularLoader'
            **kwargs:     Constructor arguments forwarded to the loader.
        """
        if key in self._registry:
            raise ValueError(f"Dataset '{key}' is already registered.")
        self._registry[key] = {"loader_class": loader_class, "kwargs": kwargs}

    def get(self, dataset_id: str) -> BaseDatasetLoader:
        """
        Instantiate and return the loader for a given dataset ID.

        Args:
            dataset_id: Registry key for the desired dataset.

        Returns:
            An instantiated BaseDatasetLoader subclass.

        Raises:
            KeyError: If the dataset_id is not registered and not a known stub.
            ImportError: If the loader class cannot be imported.
        """
        if dataset_id not in self._registry:
            # Fallback to stubs for known restricted datasets
            stub_map = {
                "adni": "nadiq.data.sources.stubs.ADNIAdapter",
                "mimic_iv": "nadiq.data.sources.stubs.MIMICIVAdapter",
                "uk_biobank": "nadiq.data.sources.stubs.UKBiobankAdapter",
                "daphnet": "nadiq.data.sources.stubs.DaphnetAdapter",
                "tadpole": "nadiq.data.sources.stubs.TadpoleAdapter",
                "oasis": "nadiq.data.sources.stubs.OasisAdapter",
                "ppmi": "nadiq.data.sources.stubs.PPMIAdapter",
                "eicu": "nadiq.data.sources.stubs.EICUAdapter",
                "nhanes": "nadiq.data.sources.stubs.NHANESAdapter",
                "parkinson_speech": "nadiq.data.sources.stubs.ParkinsonSpeechAdapter",
                "parkinson_telemonitoring": "nadiq.data.sources.stubs.ParkinsonTelemonitoringAdapter",
            }
            if dataset_id in stub_map:
                loader_class = self._import_class(stub_map[dataset_id])
                return loader_class()
            
            raise KeyError(
                f"Dataset '{dataset_id}' is not registered. "
                f"Available datasets: {sorted(self._registry.keys()) + list(stub_map.keys())}"
            )

        entry = self._registry[dataset_id]
        loader_class = self._import_class(entry["loader_class"])
        return loader_class(**entry["kwargs"])

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> "DatasetRegistry":
        """
        Build a registry from all YAML dataset configs in a directory.

        Each YAML file must contain:
            dataset_id: str
            loader_class: str (dotted import path)
            ... (additional loader constructor args)

        Args:
            config_dir: Path to directory containing dataset YAML configs.

        Returns:
            Populated DatasetRegistry instance.
        """
        try:
            from omegaconf import OmegaConf  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("OmegaConf required for registry loading") from exc

        registry = cls()
        config_dir = Path(config_dir)

        if not config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        for yaml_file in sorted(config_dir.glob("*.yaml")):
            cfg = OmegaConf.to_container(OmegaConf.load(yaml_file), resolve=True)
            assert isinstance(cfg, dict), f"Config must be a dict: {yaml_file}"
            dataset_id = cfg.pop("dataset_id")
            loader_class = cfg.pop("loader_class")
            registry.register(dataset_id, loader_class, **cfg)

        return registry

    @staticmethod
    def _import_class(dotted_path: str) -> type:
        """Import a class from a dotted module path."""
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(
                f"loader_class must be a dotted path like 'package.module.ClassName', "
                f"got: '{dotted_path}'"
            )
        module_path, class_name = parts
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name, None)
        if klass is None:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}'"
            )
        return klass

    def list_registered(self) -> list[str]:
        """Return all registered dataset IDs."""
        return sorted(self._registry.keys())

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._registry
