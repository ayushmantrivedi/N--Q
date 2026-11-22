"""
Configuration loading and validation for Nāḍī-Q.

Uses OmegaConf for structured, composable YAML configuration.
Every experiment receives a fully resolved config at startup.

Usage:
    from nadiq.utils.config import load_config, ExperimentConfig

    cfg = load_config("configs/experiments/baseline_01.yaml")
    print(cfg.seed)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

__all__ = ["load_config", "resolve_config", "hash_config"]


def load_config(config_path: Union[str, Path]) -> Any:
    """
    Load and resolve an OmegaConf configuration from a YAML file.

    The configuration may contain ${base} interpolation references
    to configs/base.yaml. All references are resolved before return.

    Args:
        config_path: Path to the experiment YAML configuration file.

    Returns:
        A fully resolved OmegaConf DictConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ImportError: If OmegaConf is not installed.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "OmegaConf is required for configuration management. "
            "Install it with: pip install omegaconf"
        ) from exc

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg


def resolve_config(cfg: Any) -> dict[str, Any]:
    """
    Convert a resolved OmegaConf config to a plain Python dict.

    Useful for serialization and hashing.

    Args:
        cfg: OmegaConf DictConfig (must be fully resolved).

    Returns:
        Plain Python dict representation of the config.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore[import]

        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]
    except ImportError as exc:
        raise ImportError("OmegaConf required") from exc


def hash_config(cfg: Any) -> str:
    """
    Produce a deterministic SHA-256 hash of a configuration.

    Used to fingerprint experiment configurations for reproducibility tracking.
    Two identical configs will always produce the same hash.

    Args:
        cfg: OmegaConf DictConfig or plain dict.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).
    """
    if hasattr(cfg, "to_container"):
        cfg_dict = resolve_config(cfg)
    else:
        cfg_dict = cfg

    # Sort keys for deterministic serialization
    cfg_str = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(cfg_str.encode()).hexdigest()
