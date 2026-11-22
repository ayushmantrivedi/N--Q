"""
Multi-Layer Perceptron Baseline for Nāḍī-Q.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from nadiq.models.base import BaseModel, ModelOutput


class MLPModel(BaseModel):
    """
    Standard MLP baseline using scikit-learn.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        # Use reasonable scientific defaults
        params = {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
        }
        params.update(kwargs)
        self.model = MLPClassifier(**params)
        self._training_time: float | None = None
        self._inference_time: float | None = None
        self._is_fitted = False

    @property
    def model_id(self) -> str:
        return "MLP"

    @property
    def n_parameters(self) -> int:
        n_params = 0
        if self._is_fitted:
            for coef in self.model.coefs_:
                n_params += coef.size
            for ic in self.model.intercepts_:
                n_params += ic.size
        return n_params

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter()
        self.model.fit(X, y)
        self._training_time = time.perf_counter() - start
        self._is_fitted = True
        return {"training_time": self._training_time}

    def predict(self, features: np.ndarray) -> ModelOutput:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predict.")
        start = time.perf_counter()
        
        probas = self.model.predict_proba(features)
        eps = 1e-15
        p = np.clip(probas[:, 1], eps, 1 - eps)
        logits = np.log(p / (1 - p))
        
        self._inference_time = time.perf_counter() - start
        
        return ModelOutput(
            logits=logits,
            probabilities=probas,
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(features)

    def get_config(self) -> dict[str, Any]:
        return self.model.get_params()

    def resource_report(self) -> dict[str, Any]:
        return {
            "n_parameters": self.n_parameters,
            "training_time": self._training_time,
            "inference_time": self._inference_time,
            "is_fitted": self._is_fitted,
        }
