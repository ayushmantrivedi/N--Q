"""
Abstract base interface for all models in Nāḍī-Q.

Classical, quantum, and fusion models all implement this interface.
This ensures they are independently testable and interchangeable
within the evaluation framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

__all__ = ["BaseModel", "ModelOutput"]


from dataclasses import dataclass, field


@dataclass
class ModelOutput:
    """
    Standardized output container for all model predictions.

    Attributes:
        logits:       Raw model output, shape (batch_size, n_classes) or (batch_size,).
        probabilities: Softmax/sigmoid probabilities. None if not computed.
        embeddings:   Intermediate representations (for analysis). None if not extracted.
        metadata:     Model-specific additional output (e.g., circuit expectation values).
    """

    logits: np.ndarray
    probabilities: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """
    Abstract interface for all Nāḍī-Q models.

    Classical, quantum, and fusion models must implement this interface
    to be usable within the evaluation and experiment framework.

    This interface is intentionally minimal. Framework-specific training
    logic (e.g., PyTorch Modules, sklearn estimators) lives in subclasses.
    """

    @abstractmethod
    def predict(self, features: np.ndarray) -> ModelOutput:
        """
        Run inference on a batch of feature vectors.

        Args:
            features: Input array of shape (batch_size, n_features).

        Returns:
            ModelOutput containing predictions and optional metadata.
        """

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.

        Args:
            features: Input array of shape (batch_size, n_features).

        Returns:
            Probability array of shape (batch_size, n_classes).
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model configuration."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Total number of trainable parameters."""

    def get_config(self) -> dict[str, Any]:
        """
        Return a serializable description of the model configuration.

        Used for experiment logging and reproducibility.
        Default implementation returns an empty dict; subclasses should override.
        """
        return {"model_id": self.model_id, "n_parameters": self.n_parameters}
