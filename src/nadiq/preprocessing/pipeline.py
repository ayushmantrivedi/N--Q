"""
Preprocessing Pipeline for Nāḍī-Q.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from nadiq.preprocessing.base import BaseTransformer


class PreprocessingPipeline(BaseTransformer):
    """
    Sequentially applies a list of transformers.
    """

    def __init__(self, steps: list[tuple[str, BaseTransformer]]) -> None:
        super().__init__()
        self.steps = steps
        
        # Verify unique names
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline steps must have unique names.")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PreprocessingPipeline":
        Xt = X
        for name, transformer in self.steps:
            Xt = transformer.fit_transform(Xt, y)
            
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before calling transform.")
            
        Xt = X
        for name, transformer in self.steps:
            Xt = transformer.transform(Xt)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before calling inverse_transform.")
            
        Xt = X
        # Apply inverse transforms in reverse order
        for name, transformer in reversed(self.steps):
            Xt = transformer.inverse_transform(Xt)
        return Xt

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, transformer in self.steps:
            params[name] = transformer.get_params()
        return params

    def fingerprint(self) -> str:
        """
        Hash the combination of all step fingerprints.
        """
        if not self.is_fitted_:
            raise ValueError("Cannot fingerprint an unfitted pipeline.")
            
        step_hashes = [
            f"{name}:{transformer.fingerprint()}" 
            for name, transformer in self.steps
        ]
        combined = "|".join(step_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
