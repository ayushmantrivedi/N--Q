"""
Base classes for Nāḍī-Q preprocessing.

All transformers must implement fit, transform, fit_transform, and inverse_transform
where mathematically meaningful. Fit must ONLY be called on training data.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class BaseTransformer(ABC):
    """Abstract base class for all preprocessing transformers."""

    def __init__(self) -> None:
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseTransformer":
        """
        Fit the transformer to the training data.
        Must NEVER be called on validation or test data.
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the transformation to the data."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Revert the transformation if mathematically meaningful.
        Raise NotImplementedError if not applicable.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse_transform.")

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return parameters for fingerprinting."""
        pass

    def fingerprint(self) -> str:
        """Generate a deterministic hash of the fitted parameters."""
        if not self.is_fitted_:
            raise ValueError("Cannot fingerprint an unfitted transformer.")
        
        # Simple fingerprinting: convert params to string and hash
        # A more robust version would handle float rounding, but this serves
        # as a basic reproducibility check.
        params = self.get_params()
        
        def serialize_val(val: Any) -> str:
            if isinstance(val, np.ndarray):
                # Use a fast hash of the array or just format with precision
                return np.array2string(val, precision=6, separator=',')
            return str(val)
            
        param_str = "|".join(f"{k}:{serialize_val(v)}" for k, v in sorted(params.items()))
        return hashlib.sha256(param_str.encode("utf-8")).hexdigest()
