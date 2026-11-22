"""
TADPOLE Classical Benchmark Runner — Nāḍī-Q Phase 4.

Executes the full classical baseline benchmark on real TADPOLE D1 data.

Architecture:
    For each horizon H ∈ {12, 24, 36, 48, 60} months:
        1. Instantiate a fresh TADPOLELoader (H-specific cohort)
        2. Build timelines and resolve temporal labels
        3. Patient-level split (BEFORE preprocessing)
        4. Fit preprocessing on that horizon's train population ONLY
        5. Run LR, GBT, MLP — same partitions, same preprocessing
        6. Compute full metric set + bootstrap CIs
        7. Emit per-horizon scientific audit

Guarantees:
    - No global preprocessing shared across horizons
    - No patient overlap across train/val/test
    - Preprocessing fitted only on training data
    - D1-only population (subset="D1")
    - DEMENTIA endpoint (not AD)
    - BASELINE_MCI snapshot policy

Usage:
    python experiments/run_tadpole_benchmark.py
    python experiments/run_tadpole_benchmark.py --data-path /path/to/tadpole
    python experiments/run_tadpole_benchmark.py --config configs/experiments/adni_mci_conversion_v1.yaml
    python experiments/run_tadpole_benchmark.py --horizons 24 36
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Try pyarrow for Parquet output (optional)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from nadiq.data.contracts import ClinicalSample, DataSplit
from nadiq.data.loaders.tadpole import TADPOLELoader
from nadiq.evaluation.calibration.metrics import compute_calibration_metrics
from nadiq.evaluation.classification.metrics import compute_classification_metrics
from nadiq.evaluation.statistics.bootstrap import compute_bootstrap_ci
from nadiq.preprocessing.filtering import VarianceFilter
from nadiq.preprocessing.imputation import MedianImputer
from nadiq.preprocessing.normalization import ZScoreNormalizer
from nadiq.preprocessing.pipeline import PreprocessingPipeline
from nadiq.utils.seeding import seed_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — add new classical models here only
# ---------------------------------------------------------------------------
CLASSICAL_MODELS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "LogisticRegression",
        "nadiq.models.classical.logistic.LogisticRegressionModel",
        {"max_iter": 2000, "random_state": 42, "C": 1.0},
    ),
    (
        "HistGradientBoosting",
        "nadiq.models.classical.gradient_boosting.GradientBoostedTreeModel",
        {"random_state": 42},
    ),
    (
        "MLP",
        "nadiq.models.classical.mlp.MLPModel",
        {"hidden_layer_sizes": (64, 32), "max_iter": 500, "random_state": 42},
    ),
]

DEFAULT_HORIZONS = [12, 24, 36, 48, 60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_clinical_data(
    samples: list[ClinicalSample],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y, sample_ids) from a list of ClinicalSamples."""
    if not samples:
        return np.empty((0, 0)), np.empty(0), []

    X_list, y_list, ids = [], [], []
    for s in samples:
        if "clinical" not in s.features:
            raise ValueError(
                f"Sample {s.sample_id} missing 'clinical' feature block. "
                f"Available: {list(s.features.keys())}"
            )
        X_list.append(s.features["clinical"].values)
        y_list.append(s.label.value)  # type: ignore[union-attr]
        ids.append(s.sample_id)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), ids


def instantiate_model(class_path: str, kwargs: dict[str, Any]) -> Any:
    parts = class_path.rsplit(".", 1)
    module = importlib.import_module(parts[0])
    return getattr(module, parts[1])(**kwargs)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Run predict and compute the full metric suite."""
    if len(X) == 0 or len(np.unique(y)) < 2:
        return {
            "classification": {"n_samples": len(X), "note": "insufficient data or single class"},
            "calibration": {},
            "auc_bootstrap_ci": None,
            "predictions": np.array([]),
            "probabilities": np.empty((0, 2)),
        }

    out = model.predict(X)
    if out.probabilities is None:
        raise ValueError("Model.predict() must return probabilities.")

    probs = out.probabilities
    preds = (probs[:, 1] >= 0.5).astype(int)

    cls_m = compute_classification_metrics(y, preds, probs)
    cal_m = compute_calibration_metrics(y, probs)

    from sklearn.metrics import roc_auc_score

    def auc_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(roc_auc_score(yt, yp[:, 1]))

    ci = compute_bootstrap_ci(y, probs, auc_fn, n_bootstrap=1000, seed=seed)

    return {
        "classification": cls_m,
        "calibration": cal_m,
        "auc_bootstrap_ci": ci,
        "predictions": preds,
        "probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_tadpole_benchmark(
    data_path: str | None,
    feature_config_path: str | None,
    horizons: list[int],
    horizon_definition: str,
    subset: str,
    seed: int,
    results_dir: str,
) -> None:
    seed_all(seed)

    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    run_dir = Path(results_dir) / f"tadpole_{exp_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment ID : {exp_id}")
    logger.info(f"Results dir   : {run_dir}")
    logger.info(f"Subset        : {subset}")
    logger.info(f"Horizons      : {horizons}")
    logger.info(f"Seed          : {seed}")

    # Accumulate for terminal summary table
    # summary[horizon][model_name] = test_metrics dict
    summary: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    horizon_cohort_stats: dict[int, dict[str, Any]] = {}

    for H in horizons:
        logger.info(f"\n{'='*70}")
        logger.info(f"HORIZON: {H} months  [{horizon_definition}]")
        logger.info(f"{'='*70}")

        h_dir = run_dir / f"{H}m"
        h_dir.mkdir(exist_ok=True)

        # ---------------------------------------------------------------
        # 1. Build H-specific cohort
        # ---------------------------------------------------------------
        loader = TADPOLELoader(
            data_path=data_path,
            feature_config_path=feature_config_path,
            subset=subset,
            target_state="DEMENTIA",
            horizon_months=H,
            horizon_definition=horizon_definition,
            snapshot_policy="BASELINE_MCI",
            seed=seed,
        )

        try:
            train_samples = list(loader.load_split(DataSplit.TRAIN))
            val_samples   = list(loader.load_split(DataSplit.VAL))
            test_samples  = list(loader.load_split(DataSplit.TEST))
        except PermissionError as exc:
            logger.error(str(exc))
            logger.error(
                "TADPOLE data is not configured. "
                "Set TADPOLE_DATA_DIR or pass --data-path."
            )
            sys.exit(1)

        audit = loader.audit_stats
        horizon_cohort_stats[H] = audit

        logger.info(
            f"Cohort  → eligible MCI: {audit.get('n_eligible_mci', '?')}, "
            f"positive: {audit.get('n_positive', '?')}, "
            f"negative: {audit.get('n_negative', '?')}, "
            f"censored: {audit.get('n_censored', '?')}"
        )
        logger.info(
            f"Splits  → train: {audit.get('train_count', '?')}, "
            f"val: {audit.get('val_count', '?')}, "
            f"test: {audit.get('test_count', '?')}"
        )

        if len(train_samples) < 10:
            logger.warning(f"Only {len(train_samples)} training samples for {H}M. Skipping.")
            continue

        # ---------------------------------------------------------------
        # 2. Extract arrays
        # ---------------------------------------------------------------
        X_train, y_train, ids_train = extract_clinical_data(train_samples)
        X_val,   y_val,   ids_val   = extract_clinical_data(val_samples)
        X_test,  y_test,  ids_test  = extract_clinical_data(test_samples)

        logger.info(
            f"Arrays  → train {X_train.shape}, val {X_val.shape}, test {X_test.shape}"
        )

        # ---------------------------------------------------------------
        # 3. Preprocessing — fit on THIS horizon's train ONLY
        # ---------------------------------------------------------------
        pipeline = PreprocessingPipeline([
            ("impute",    MedianImputer()),
            ("variance",  VarianceFilter()),
            ("normalize", ZScoreNormalizer()),
        ])
        pipeline.fit(X_train, y_train)
        prep_fp = pipeline.fingerprint()
        logger.info(f"Preprocessing fingerprint: {prep_fp}")

        X_train_p = pipeline.transform(X_train)
        X_val_p   = pipeline.transform(X_val)
        X_test_p  = pipeline.transform(X_test)

        # ---------------------------------------------------------------
        # 4. Run each classical model
        # ---------------------------------------------------------------
        horizon_results: dict[str, dict[str, Any]] = {}

        for model_name, class_path, model_kwargs in CLASSICAL_MODELS:
            logger.info(f"  ▶ Training {model_name}...")

            model = instantiate_model(class_path, model_kwargs)
            train_info = model.fit(X_train_p, y_train)

            val_r  = evaluate_model(model, X_val_p,  y_val,  seed)
            test_r = evaluate_model(model, X_test_p, y_test, seed)

            cls_v = val_r["classification"]
            cls_t = test_r["classification"]

            logger.info(
                f"    Val   — AUROC: {cls_v.get('roc_auc', float('nan')):.3f}  "
                f"AUPRC: {cls_v.get('prc_auc', float('nan')):.3f}  "
                f"F1: {cls_v.get('f1', float('nan')):.3f}"
            )
            logger.info(
                f"    Test  — AUROC: {cls_t.get('roc_auc', float('nan')):.3f}  "
                f"AUPRC: {cls_t.get('prc_auc', float('nan')):.3f}  "
                f"F1: {cls_t.get('f1', float('nan')):.3f}"
            )

            horizon_results[model_name] = {
                "validation": {
                    "classification": cls_v,
                    "calibration": {
                        k: v for k, v in val_r["calibration"].items()
                        if not k.startswith("calibration_curve")
                    },
                    "auc_bootstrap_ci": val_r["auc_bootstrap_ci"],
                },
                "test": {
                    "classification": cls_t,
                    "calibration": {
                        k: v for k, v in test_r["calibration"].items()
                        if not k.startswith("calibration_curve")
                    },
                    "auc_bootstrap_ci": test_r["auc_bootstrap_ci"],
                },
                "training_info": train_info,
                "resource_report": model.resource_report(),
            }

            summary[H][model_name] = cls_t

            # Save predictions (Parquet, optional)
            if HAS_PYARROW and len(X_test_p) > 0 and len(test_r["predictions"]) > 0:
                pred_records = []
                for i, sid in enumerate(ids_test):
                    pred_records.append({
                        "sample_id": sid,
                        "ground_truth": int(y_test[i]),
                        "prediction":   int(test_r["predictions"][i]),
                        "probability":  float(test_r["probabilities"][i, 1]),
                        "split": "test",
                        "model": model_name,
                        "horizon_months": H,
                        "experiment_id": exp_id,
                    })
                tbl = pa.Table.from_pylist(pred_records)
                pq.write_table(tbl, h_dir / f"predictions_{model_name}.parquet")

        # ---------------------------------------------------------------
        # 5. Save horizon metrics + audit
        # ---------------------------------------------------------------
        with open(h_dir / "metrics.json", "w") as f:
            json.dump(horizon_results, f, indent=2)

        h_audit = {
            **audit,
            "preprocessing_fingerprint": prep_fp,
            "preprocessing_fit_partition": "train_only",
            "horizon_months": H,
            "horizon_definition": horizon_definition,
            "models_run": [m[0] for m in CLASSICAL_MODELS],
        }
        with open(h_dir / "scientific_audit.json", "w") as f:
            json.dump(h_audit, f, indent=2)

    # -----------------------------------------------------------------------
    # 6. Terminal summary table
    # -----------------------------------------------------------------------
    _print_summary_table(summary, horizon_cohort_stats, horizons)

    # -----------------------------------------------------------------------
    # 7. Overall manifest
    # -----------------------------------------------------------------------
    manifest = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "subset": subset,
        "horizons": horizons,
        "horizon_definition": horizon_definition,
        "seed": seed,
        "models": [m[0] for m in CLASSICAL_MODELS],
        "endpoint_label": "5-year future dementia-transition risk from MCI baseline snapshot",
        "endpoint_NOT": ["Alzheimer's Disease prediction", "5-year AD validated predictor"],
        "snapshot_policy": "BASELINE_MCI",
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "label": "TADPOLE D1 CLASSICAL BASELINE — NOT CLINICAL VALIDATED",
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save summary table as JSON too
    serialisable_summary: dict[str, Any] = {}
    for H, models in summary.items():
        serialisable_summary[f"{H}m"] = {
            mn: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in mets.items()}
            for mn, mets in models.items()
        }
    with open(run_dir / "summary_table.json", "w") as f:
        json.dump(serialisable_summary, f, indent=2)

    logger.info(f"\nAll results saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def _print_summary_table(
    summary: dict[int, dict[str, dict[str, Any]]],
    cohort_stats: dict[int, dict[str, Any]],
    horizons: list[int],
) -> None:
    model_names = [m[0] for m in CLASSICAL_MODELS]
    h_labels = [f"{H}M" for H in horizons]

    print("\n")
    print("=" * 90)
    print("TADPOLE REAL-DATA BENCHMARK — CLASSICAL BASELINE")
    print("Endpoint: 5-year future dementia-transition risk from MCI baseline snapshot")
    print("NOT: Alzheimer's Disease prediction. NOT: clinically validated.")
    print("=" * 90)

    # Cohort table
    print(f"\n{'COHORT':─<90}")
    print(f"{'Horizon':>10} {'Eligible':>10} {'Positive':>10} {'Negative':>10} "
          f"{'Censored':>10} {'Train':>8} {'Val':>6} {'Test':>6}")
    print("─" * 90)
    for H in horizons:
        s = cohort_stats.get(H, {})
        print(
            f"{H:>8}M  "
            f"{s.get('n_eligible_mci', '?'):>10}  "
            f"{s.get('n_positive', '?'):>10}  "
            f"{s.get('n_negative', '?'):>10}  "
            f"{s.get('n_censored', '?'):>10}  "
            f"{s.get('train_count', '?'):>8}  "
            f"{s.get('val_count', '?'):>6}  "
            f"{s.get('test_count', '?'):>6}"
        )

    # Metrics table — AUROC
    _print_metric_block("AUROC (test set)", "roc_auc", summary, horizons, model_names)

    # Metrics table — AUPRC
    _print_metric_block("AUPRC (test set)", "prc_auc", summary, horizons, model_names)

    # Metrics table — F1
    _print_metric_block("F1 (test set, threshold=0.5)", "f1", summary, horizons, model_names)

    # Metrics table — Brier (calibration, stored in calibration dict)
    print(f"\n{'BRIER SCORE (test set)':─<90}")
    col_w = 20
    header = f"{'Model':<25}" + "".join(f"{hl:>{col_w}}" for hl in h_labels)
    print(header)
    print("─" * (25 + col_w * len(horizons)))
    # We stored calibration separately — read from summary would need restructuring
    # so just note where to find it:
    print("  (See metrics.json / calibration / brier_score per horizon dir)")

    print("\n" + "=" * 90)
    print("Bootstrap 95% CI for AUROC: see metrics.json → auc_bootstrap_ci")
    print("Confusion matrices: see metrics.json → classification → tn/fp/fn/tp")
    print("Calibration curves: see metrics.json → calibration")
    print("=" * 90)


def _print_metric_block(
    title: str,
    metric_key: str,
    summary: dict[int, dict[str, dict[str, Any]]],
    horizons: list[int],
    model_names: list[str],
) -> None:
    h_labels = [f"{H}M" for H in horizons]
    col_w = 10

    print(f"\n{title:─<90}")
    header = f"{'Model':<25}" + "".join(f"{hl:>{col_w}}" for hl in h_labels)
    print(header)
    print("─" * (25 + col_w * len(horizons)))

    for mn in model_names:
        row_str = f"{mn:<25}"
        for H in horizons:
            val = summary.get(H, {}).get(mn, {}).get(metric_key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cell = "  N/A"
            else:
                cell = f"{val:>{col_w - 2}.3f}  "
            row_str += cell
        print(row_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TADPOLE Classical Benchmark Runner — Nāḍī-Q Phase 4"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Directory containing TADPOLE_D1_D2.csv. Falls back to TADPOLE_DATA_DIR."
    )
    parser.add_argument(
        "--feature-config", type=str, default=None,
        help="Path to feature config YAML. Falls back to NADIQ_FEATURE_CONFIG or default."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional experiment YAML config (overrides individual flags)."
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
        help="Horizon(s) in months. Default: 12 24 36 48 60"
    )
    parser.add_argument(
        "--horizon-definition", type=str, default="WITHIN",
        choices=["WITHIN", "EXACT", "AT_LEAST"],
    )
    parser.add_argument(
        "--subset", type=str, default="D1",
        choices=["D1", "D2", "D1D2"],
        help="TADPOLE subset to use. Default: D1"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
    )
    args = parser.parse_args()

    # Allow YAML config to override flags
    data_path          = args.data_path or os.environ.get("TADPOLE_DATA_DIR")
    feature_config     = args.feature_config
    horizons           = args.horizons
    horizon_definition = args.horizon_definition
    subset             = args.subset
    seed               = args.seed

    if args.config:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
            assert isinstance(cfg, dict)
            data_cfg = cfg.get("data", {})
            if data_cfg.get("data_path"):
                data_path = data_cfg["data_path"]
            if data_cfg.get("feature_config_path"):
                feature_config = data_cfg["feature_config_path"]
            if data_cfg.get("subset"):
                subset = data_cfg["subset"]
            if "horizons" in data_cfg:
                horizons = [h["months"] for h in data_cfg["horizons"]]
                horizon_definition = data_cfg["horizons"][0].get("definition", "WITHIN")
            seed = cfg.get("seed", seed)
        except Exception as exc:
            logger.warning(f"Could not load config {args.config}: {exc}. Using CLI flags.")

    run_tadpole_benchmark(
        data_path=data_path,
        feature_config_path=feature_config,
        horizons=sorted(horizons),
        horizon_definition=horizon_definition,
        subset=subset,
        seed=seed,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
