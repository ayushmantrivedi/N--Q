"""
Experiment runner for Nāḍī-Q.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Try to import pyarrow for parquet
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from nadiq.data.contracts import DataSplit, ClinicalSample
from nadiq.data.registry import DatasetRegistry
from nadiq.evaluation.calibration.metrics import compute_calibration_metrics
from nadiq.evaluation.classification.metrics import compute_classification_metrics
from nadiq.evaluation.statistics.bootstrap import compute_bootstrap_ci
from nadiq.preprocessing.filtering import VarianceFilter
from nadiq.preprocessing.imputation import MedianImputer
from nadiq.preprocessing.normalization import ZScoreNormalizer
from nadiq.preprocessing.pipeline import PreprocessingPipeline
from nadiq.utils.config import load_config
from nadiq.utils.seeding import seed_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_tabular_data(samples: list[ClinicalSample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extracts tabular features and binary labels from clinical samples."""
    if not samples:
        return np.array([]), np.array([]), []
        
    X_list = []
    y_list = []
    sample_ids = []
    
    for s in samples:
        sample_ids.append(s.sample_id)
        if "tabular" not in s.features:
            raise ValueError(f"Sample {s.sample_id} missing tabular modality.")
        X_list.append(s.features["tabular"].values)
        y_list.append(s.label.value)
        
    return np.array(X_list), np.array(y_list), sample_ids


def instantiate_model(model_class_str: str, kwargs: dict[str, Any]) -> Any:
    """Instantiate a model from a dotted class path."""
    parts = model_class_str.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model class path: {model_class_str}")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    klass = getattr(module, class_name)
    return klass(**kwargs)


def run_experiment(config_path: str | Path, results_dir: str = "results") -> None:
    # 1. Load config
    from nadiq.utils.config import hash_config
    cfg = load_config(config_path)
    fingerprint = hash_config(cfg)
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Config fingerprint: {fingerprint}")
    
    # 2. Seeding
    seed = cfg.get("seed", 42)
    seed_all(seed)
    logger.info(f"Set global seed to {seed}")
    
    # Generate experiment ID
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    run_dir = Path(results_dir) / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment ID: {exp_id}")
    
    # 3. Data Loader
    data_cfg = cfg.get("data", {})
    if not data_cfg:
        raise ValueError("Config must contain 'data' section")
        
    dataset_id = data_cfg.get("dataset_id")
    if not dataset_id:
        raise ValueError("Data config must contain 'dataset_id'")
        
    registry = DatasetRegistry()
    
    # Automatically register synthetic and tadpole loaders
    if "synthetic_tabular_v1" not in registry._registry:
        from nadiq.data.loaders.synthetic import SyntheticTabularLoader
        registry.register("synthetic_tabular_v1", "nadiq.data.loaders.synthetic.SyntheticTabularLoader", dataset_id="synthetic_tabular_v1", seed=seed)
        
    if "tadpole" not in registry._registry:
        from nadiq.data.loaders.tadpole import TADPOLELoader
        registry.register("tadpole", "nadiq.data.loaders.tadpole.TADPOLELoader", seed=seed)
            
    # Check if we have multiple horizons
    horizons = data_cfg.get("horizons", [None])
    
    # Track overall metrics across all horizons
    all_metrics = {}
    
    for h_config in horizons:
        logger.info(f"--- Running experiment for horizon: {h_config} ---")
        
        loader_kwargs = {}
        if h_config is not None:
            loader_kwargs["horizon_months"] = h_config["months"]
            loader_kwargs["horizon_definition"] = h_config["definition"]
            
        loader_class_path = registry._registry[dataset_id]["loader_class"]
        # Instantiate a fresh loader for this horizon
        loader = registry._import_class(loader_class_path)(**registry._registry[dataset_id]["kwargs"], **loader_kwargs)
            
        logger.info(f"Loading dataset: {dataset_id}")
        train_samples = list(loader.load_split(DataSplit.TRAIN))
        val_samples = list(loader.load_split(DataSplit.VAL))
        test_samples = list(loader.load_split(DataSplit.TEST))
        
        X_train, y_train, ids_train = extract_tabular_data(train_samples)
        X_val, y_val, ids_val = extract_tabular_data(val_samples)
        X_test, y_test, ids_test = extract_tabular_data(test_samples)
        
        # 4. Preprocessing Fit
        logger.info("Fitting preprocessing pipeline on training data ONLY...")
        pipeline = PreprocessingPipeline([
            ("impute", MedianImputer()),
            ("variance", VarianceFilter()),
            ("normalize", ZScoreNormalizer()),
        ])
        
        pipeline.fit(X_train, y_train)
        prep_fingerprint = pipeline.fingerprint()
        logger.info(f"Preprocessing fingerprint: {prep_fingerprint}")
        
        X_train_p = pipeline.transform(X_train)
        X_val_p = pipeline.transform(X_val)
        X_test_p = pipeline.transform(X_test)
        
        # 5. Train Model
        model_cfg = cfg.get("model", {})
        model_class = model_cfg.get("class")
        model_kwargs = model_cfg.get("params", {})
        
        logger.info(f"Instantiating model: {model_class}")
        model = instantiate_model(model_class, model_kwargs)
        
        logger.info("Training model...")
        train_info = model.fit(X_train_p, y_train)
        
        # 6. Validation and Test Evaluation
        logger.info("Evaluating model...")
        
        def evaluate(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
            if len(X) == 0:
                return {
                    "classification": {},
                    "calibration": {},
                    "auc_bootstrap_ci": None,
                    "predictions": np.array([]),
                    "probabilities": np.array([]),
                }
            out = model.predict(X)
            if out.probabilities is None:
                raise ValueError("Model did not return probabilities.")
            
            preds = (out.probabilities[:, 1] >= 0.5).astype(int)
            
            cls_metrics = compute_classification_metrics(y, preds, out.probabilities)
            cal_metrics = compute_calibration_metrics(y, out.probabilities)
            
            # Bootstrap AUC
            from sklearn.metrics import roc_auc_score
            def auc_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(roc_auc_score(y_true, y_pred[:, 1]))
                
            ci = compute_bootstrap_ci(y, out.probabilities, auc_fn, n_bootstrap=1000, seed=seed)
            
            return {
                "classification": cls_metrics,
                "calibration": cal_metrics,
                "auc_bootstrap_ci": ci,
                "predictions": preds,
                "probabilities": out.probabilities,
            }
            
        val_results = evaluate(X_val_p, y_val)
        test_results = evaluate(X_test_p, y_test)
        
        horizon_key = f"{h_config['months']}m" if h_config else "default"
        
        metrics = {
            "validation": {
                "classification": val_results["classification"],
                "calibration": {k: v for k, v in val_results["calibration"].items() if not k.startswith("calibration_curve")},
                "auc_bootstrap_ci": val_results["auc_bootstrap_ci"]
            },
            "test": {
                "classification": test_results["classification"],
                "calibration": {k: v for k, v in test_results["calibration"].items() if not k.startswith("calibration_curve")},
                "auc_bootstrap_ci": test_results["auc_bootstrap_ci"]
            }
        }
        all_metrics[horizon_key] = metrics
        
        # 7. Save Results (append horizon to dir if multiple)
        h_dir = run_dir / horizon_key if len(horizons) > 1 else run_dir
        if len(horizons) > 1:
            h_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Saving horizon results to {h_dir}")
            
        # Metrics
        with open(h_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Resource Report
        res_report = model.resource_report()
        res_report["preprocessing_fingerprint"] = prep_fingerprint
        with open(h_dir / "resource_report.json", "w") as f:
            json.dump(res_report, f, indent=2)
            
        # Scientific Audit Output
        if hasattr(loader, "audit_stats"):
            with open(h_dir / "scientific_audit.json", "w") as f:
                json.dump(loader.audit_stats, f, indent=2)
            
        # Training History
        with open(h_dir / "training_history.json", "w") as f:
            json.dump(train_info, f, indent=2)
            
        # Predictions (Parquet)
        if HAS_PYARROW and len(X_test_p) > 0:
            pred_records = []
            
            def add_records(ids: list[str], y: np.ndarray, preds: np.ndarray, probs: np.ndarray, split: str) -> None:
                for i in range(len(ids)):
                    pred_records.append({
                        "sample_id": ids[i],
                        "ground_truth": int(y[i]),
                        "prediction": int(preds[i]),
                        "probability": float(probs[i, 1]),
                        "split": split,
                        "model": model_class,
                        "experiment_id": exp_id,
                        "horizon": horizon_key
                    })
                    
            out_train = model.predict(X_train_p)
            train_preds = (out_train.probabilities[:, 1] >= 0.5).astype(int)
            add_records(ids_train, y_train, train_preds, out_train.probabilities, "train")
            add_records(ids_val, y_val, val_results["predictions"], val_results["probabilities"], "val")
            add_records(ids_test, y_test, test_results["predictions"], test_results["probabilities"], "test")
            
            table = pa.Table.from_pylist(pred_records)
            pq.write_table(table, h_dir / "predictions.parquet")
            
    # Save overall config and manifest at run_dir level
    from omegaconf import OmegaConf
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
        
    manifest = {
        "experiment_id": exp_id,
        "config_fingerprint": fingerprint,
        "model_class": model_class,
        "dataset_id": dataset_id,
        "seed": seed,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "label": "TADPOLE LONGITUDINAL BENCHMARK" if dataset_id == "tadpole" else "SYNTHETIC"
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info("Experiment completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nāḍī-Q Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results")
    args = parser.parse_args()
    
    run_experiment(args.config, args.results_dir)


if __name__ == "__main__":
    main()
