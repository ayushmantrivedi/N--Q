"""
Synthetic tabular data loader for Nāḍī-Q.

This loader generates deterministic synthetic data for software validation,
pipeline debugging, and baseline testing. It must NEVER be used to make
clinical claims or evaluate clinical performance.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split

from nadiq.data.contracts import (
    BaseDatasetLoader,
    ClinicalLabel,
    ClinicalSample,
    DatasetSchema,
    DatasetStats,
    DataSplit,
    FeatureBlock,
    FeatureDtype,
    LabelType,
    ModalityType,
    SampleMetadata,
)


class SyntheticTabularLoader(BaseDatasetLoader):
    """
    Generates deterministic synthetic tabular data.

    Data is generated in memory using sklearn's make_classification.
    The exact same data is generated across runs for a given seed.
    """

    def __init__(
        self,
        dataset_id: str,
        n_samples: int = 5000,
        n_features: int = 32,
        n_informative: int = 10,
        class_balance: float = 0.3,
        noise_std: float = 0.1,
        seed: int = 42,
        task: str = "binary_risk",
        label_def: str = "Synthetic binary risk label",
        split_strategy: Optional[dict] = None,
        data_root: Optional[str] = None,
    ) -> None:
        self._dataset_id = dataset_id
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.class_balance = class_balance
        self.noise_std = noise_std
        self.seed = seed
        self.task = task
        self.label_def = label_def
        self.split_strategy = split_strategy or {
            "type": "stratified_holdout",
            "test_fraction": 0.2,
            "val_fraction": 0.1,
        }

        # Lazy loading
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._splits: dict[DataSplit, np.ndarray] = {}

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def _ensure_generated(self) -> None:
        if self._X is not None:
            return

        # Generate data
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_features - self.n_informative,
            n_classes=2,
            weights=[1.0 - self.class_balance, self.class_balance],
            random_state=self.seed,
            shuffle=True,
        )

        # Add Gaussian noise
        rng = np.random.default_rng(self.seed)
        X += rng.normal(0, self.noise_std, size=X.shape)

        self._X = X.astype(np.float32)
        self._y = y.astype(np.int64)

        # Create splits
        indices = np.arange(self.n_samples)
        if self.split_strategy["type"] == "stratified_holdout":
            test_frac = self.split_strategy.get("test_fraction", 0.2)
            val_frac = self.split_strategy.get("val_fraction", 0.1)
            
            # Train+Val vs Test
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_frac, stratify=y, random_state=self.seed
            )
            
            # Train vs Val
            # Adjust val_frac to be relative to train_val size
            rel_val_frac = val_frac / (1.0 - test_frac) if (1.0 - test_frac) > 0 else 0
            
            if rel_val_frac > 0:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=rel_val_frac,
                    stratify=y[train_val_idx],
                    random_state=self.seed,
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=int)

            self._splits[DataSplit.TRAIN] = train_idx
            self._splits[DataSplit.VAL] = val_idx
            self._splits[DataSplit.TEST] = test_idx

        elif self.split_strategy["type"] == "stratified_kfold":
            # For simplicity, we just use fold 0 as val, fold 1 as test, rest as train 
            # if kfold is specified, though a true runner might loop.
            # Here we just implement a static split for the loader contract.
            n_folds = self.split_strategy.get("n_folds", 5)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            folds = list(skf.split(X, y))
            test_idx = folds[0][1]
            val_idx = folds[1][1]
            train_idx = np.concatenate([folds[i][1] for i in range(2, n_folds)])
            
            self._splits[DataSplit.TRAIN] = train_idx
            self._splits[DataSplit.VAL] = val_idx
            self._splits[DataSplit.TEST] = test_idx
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy['type']}")

    def get_schema(self) -> DatasetSchema:
        return DatasetSchema(
            dataset_id=self.dataset_id,
            modalities={"tabular": ModalityType.TABULAR},
            feature_dims={"tabular": self.n_features},
            label_def=self.label_def,
            label_type=LabelType.BINARY,
            n_classes=2,
        )

    def get_stats(self) -> DatasetStats:
        self._ensure_generated()
        assert self._X is not None and self._y is not None
        
        train_idx = self._splits[DataSplit.TRAIN]
        X_train = self._X[train_idx]
        y_train = self._y[train_idx]
        
        n_samples = {k.value: len(v) for k, v in self._splits.items()}
        class_counts = {0: int(np.sum(y_train == 0)), 1: int(np.sum(y_train == 1))}
        
        feature_means = {"tabular": np.mean(X_train, axis=0)}
        feature_stds = {"tabular": np.std(X_train, axis=0)}
        missing_rates = {"tabular": np.zeros(self.n_features)}
        
        return DatasetStats(
            n_samples=n_samples,
            class_counts=class_counts,
            feature_means=feature_means,
            feature_stds=feature_stds,
            missing_rates=missing_rates,
        )

    def load_split(self, split: DataSplit) -> Iterable[ClinicalSample]:
        self._ensure_generated()
        assert self._X is not None and self._y is not None

        indices = self._splits[split]
        feature_names = [f"f{i:03d}" for i in range(self.n_features)]

        # We yield one by one to respect the generator contract,
        # though in-memory batch loading is also possible.
        for i, idx in enumerate(indices):
            # Deterministic patient ID
            sample_id = f"syn_{split.value}_{i:06d}"
            
            block = FeatureBlock(
                modality=ModalityType.TABULAR,
                values=self._X[idx],
                feature_names=feature_names,
                dtype=FeatureDtype.CONTINUOUS,
            )
            
            label = ClinicalLabel(
                task=self.task,
                value=int(self._y[idx]),
                label_type=LabelType.BINARY,
                confidence=1.0,
                source="synthetic",
            )
            
            metadata = SampleMetadata(
                dataset_id=self.dataset_id,
                split=split,
            )
            
            yield ClinicalSample(
                sample_id=sample_id,
                features={"tabular": block},
                label=label,
                metadata=metadata,
            )
