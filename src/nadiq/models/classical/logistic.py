"""
Logistic Regression Baseline for Nāḍī-Q.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from nadiq.models.base import BaseModel, ModelOutput


class LogisticRegressionModel(BaseModel):
    """
    Standard Logistic Regression baseline.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        # Set default scientific baseline params, allow override
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)
        self.model = LogisticRegression(**params)
        self._training_time: float | None = None
        self._inference_time: float | None = None
        self._is_fitted = False

    @property
    def model_id(self) -> str:
        return "LogisticRegression"

    @property
    def n_parameters(self) -> int:
        if self._is_fitted:
            return self.model.coef_.size + self.model.intercept_.size
        return 0

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
        
        logits = self.model.decision_function(features)
        probas = self.model.predict_proba(features)
        preds = self.model.predict(features)
        
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
