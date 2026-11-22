"""
QVBE Smoke Test — Real TADPOLE Pipeline.

Proves that the real TADPOLE data pipeline can feed QVBE and produce
a valid quantum embedding. This is NOT a training run.

Purpose ONLY:
  - Verify end-to-end connectivity: TADPOLE loader → preprocessing → QVBE
  - Confirm embedding shape and range
  - Report QVBE architecture statistics

Does NOT:
  - Train for predictive performance
  - Perform hyperparameter search
  - Claim quantum advantage
  - Make accuracy claims

Usage:
    python experiments/qvbe_smoke_test.py
    python experiments/qvbe_smoke_test.py --data-path D:/datasets/tadpole
    python experiments/qvbe_smoke_test.py --n-qubits 4 --n-layers 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("qvbe_smoke_test")


def run_smoke_test(
    data_path: str | None,
    n_qubits: int,
    n_layers: int,
    batch_size: int,
    seed: int,
) -> None:
    from nadiq.data.contracts import DataSplit
    from nadiq.data.loaders.tadpole import TADPOLELoader
    from nadiq.models.quantum.qvbe import (
        QVBEConfig,
        QuantumVariationalBiomarkerEncoder,
    )
    from nadiq.preprocessing.imputation import MedianImputer
    from nadiq.preprocessing.normalization import ZScoreNormalizer
    from nadiq.preprocessing.pipeline import PreprocessingPipeline
    from nadiq.utils.seeding import seed_all

    seed_all(seed)

    print("\n" + "=" * 70)
    print("QVBE SMOKE TEST — Real TADPOLE Pipeline")
    print("Purpose: verify data flows through QVBE (NOT training for accuracy)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load TADPOLE (D1 subset, 24-month horizon — any horizon will do)
    # -----------------------------------------------------------------------
    logger.info("Loading TADPOLE D1 (24M horizon)...")
    loader = TADPOLELoader(
        data_path=data_path,
        subset="D1",
        target_state="DEMENTIA",
        horizon_months=24,
        horizon_definition="WITHIN",
        snapshot_policy="BASELINE_MCI",
        seed=seed,
    )

    try:
        train_samples = list(loader.load_split(DataSplit.TRAIN))
    except (PermissionError, SystemExit) as exc:
        logger.error(f"TADPOLE data not available: {exc}")
        logger.error("Set TADPOLE_DATA_DIR or pass --data-path.")
        sys.exit(1)

    if not train_samples:
        logger.warning("No training samples found for 24M horizon. Trying 12M...")
        loader2 = TADPOLELoader(
            data_path=data_path,
            subset="D1",
            target_state="DEMENTIA",
            horizon_months=12,
            horizon_definition="WITHIN",
            snapshot_policy="BASELINE_MCI",
            seed=seed,
        )
        train_samples = list(loader2.load_split(DataSplit.TRAIN))

    if not train_samples:
        logger.error("No samples available from TADPOLE for smoke test.")
        sys.exit(1)

    logger.info(f"Loaded {len(train_samples)} training samples.")

    # -----------------------------------------------------------------------
    # 2. Extract feature arrays
    # -----------------------------------------------------------------------
    X_list = []
    for s in train_samples:
        if "clinical" in s.features:
            X_list.append(s.features["clinical"].values)

    if not X_list:
        logger.error("No 'clinical' feature blocks found in samples.")
        sys.exit(1)

    X_all = np.stack(X_list, axis=0).astype(np.float32)
    logger.info(f"Feature matrix: shape={X_all.shape}")

    # -----------------------------------------------------------------------
    # 3. Preprocessing (impute only — we need features for QVBE, not labels)
    # -----------------------------------------------------------------------
    pipeline = PreprocessingPipeline([
        ("impute", MedianImputer()),
        ("normalize", ZScoreNormalizer()),
    ])
    pipeline.fit(X_all, np.zeros(len(X_all)))  # dummy labels for fit
    X_proc = pipeline.transform(X_all)
    logger.info(f"Preprocessed features: shape={X_proc.shape}, any_nan={np.isnan(X_proc).any()}")

    # -----------------------------------------------------------------------
    # 4. Instantiate QVBE
    # -----------------------------------------------------------------------
    input_dim = X_proc.shape[1]
    config = QVBEConfig(
        input_dim=input_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
    )
    qvbe = QuantumVariationalBiomarkerEncoder(config)
    qvbe.eval()

    print("\n" + "-" * 70)
    print("QVBE Configuration")
    print("-" * 70)
    cfg_dict = qvbe.get_config()
    for k, v in cfg_dict.items():
        print(f"  {k:35s}: {v}")

    # -----------------------------------------------------------------------
    # 5. Run QVBE on a mini-batch
    # -----------------------------------------------------------------------
    n = min(batch_size, len(X_proc))
    X_batch = torch.tensor(X_proc[:n])
    logger.info(f"Running QVBE on {n} samples...")

    with torch.no_grad():
        out = qvbe(X_batch)

    emb = out.embedding.numpy()

    print("\n" + "-" * 70)
    print("QVBE Smoke Test Results")
    print("-" * 70)
    print(f"  Input shape           : {tuple(X_batch.shape)}")
    print(f"  Embedding shape       : {tuple(out.embedding.shape)}")
    print(f"  Embedding min         : {emb.min():.6f}")
    print(f"  Embedding max         : {emb.max():.6f}")
    print(f"  Embedding mean        : {emb.mean():.6f}")
    print(f"  All finite            : {np.isfinite(emb).all()}")
    print(f"  Range within [-1,1]   : {(np.abs(emb) <= 1.0 + 1e-5).all()}")
    print(f"  VQC param shape       : {tuple(out.vqc_params.shape)}")

    assert out.embedding.shape == (n, n_qubits), "Embedding shape mismatch!"
    assert np.isfinite(emb).all(), "Embedding contains non-finite values!"
    assert (np.abs(emb) <= 1.0 + 1e-5).all(), "Embedding out of [-1,1] range!"

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("TADPOLE -> loader -> preprocessing -> QVBE -> embedding: OK")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="QVBE smoke test on real TADPOLE data.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = args.data_path or os.environ.get("TADPOLE_DATA_DIR")
    if not data_path:
        logger.error("No data path. Use --data-path or set TADPOLE_DATA_DIR.")
        sys.exit(1)

    run_smoke_test(
        data_path=data_path,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
