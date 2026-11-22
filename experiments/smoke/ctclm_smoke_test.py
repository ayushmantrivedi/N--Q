"""
CTCLM Smoke Test -- Real TADPOLE Pipeline.

Demonstrates end-to-end connectivity:
    TADPOLE D1 -> existing loader -> preprocessing
    -> QVBE (per-snapshot embedding)
    -> CTCLM (temporal representation)
    -> valid CTCLMOutput

Purpose ONLY:
  - Verify TADPOLE -> QVBE -> CTCLM pipeline runs without error
  - Print tensor shapes and finite-value checks

Does NOT:
  - Train for predictive performance
  - Claim quantum advantage
  - Make accuracy claims
  - Perform hyperparameter search

Usage:
    python experiments/ctclm_smoke_test.py
    python experiments/ctclm_smoke_test.py --data-path D:/datasets/tadpole
    python experiments/ctclm_smoke_test.py --n-qubits 4 --latent-dim 16 --seq-len 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ctclm_smoke_test")


def run_smoke_test(
    data_path: str | None,
    n_qubits: int,
    latent_dim: int,
    time_embedding_dim: int,
    simulated_seq_len: int,
    batch_size: int,
    seed: int,
) -> None:
    from nadiq.data.contracts import DataSplit
    from nadiq.data.loaders.tadpole import TADPOLELoader
    from nadiq.models.quantum.ctclm import (
        CTCLMConfig,
        ContinualTemporalClinicalLatentModel,
    )
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
    print("CTCLM SMOKE TEST -- Real TADPOLE Pipeline")
    print("Purpose: verify TADPOLE -> QVBE -> CTCLM connectivity")
    print("NOT a training or performance evaluation run")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load TADPOLE
    # ------------------------------------------------------------------
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
        sys.exit(1)

    if not train_samples:
        logger.warning("No 24M training samples found; falling back to 12M.")
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
        logger.error("No training samples found.")
        sys.exit(1)

    logger.info(f"Loaded {len(train_samples)} training samples.")

    # ------------------------------------------------------------------
    # 2. Extract features
    # ------------------------------------------------------------------
    X_list = [s.features["clinical"].values for s in train_samples if "clinical" in s.features]
    X_all = np.stack(X_list, axis=0).astype(np.float32)
    logger.info(f"Feature matrix: {X_all.shape}")

    # ------------------------------------------------------------------
    # 3. Preprocessing
    # ------------------------------------------------------------------
    pipeline = PreprocessingPipeline([
        ("impute", MedianImputer()),
        ("normalize", ZScoreNormalizer()),
    ])
    pipeline.fit(X_all, np.zeros(len(X_all)))
    X_proc = pipeline.transform(X_all)

    # ------------------------------------------------------------------
    # 4. QVBE: encode each snapshot into a quantum embedding
    # ------------------------------------------------------------------
    input_dim = X_proc.shape[1]
    qvbe_config = QVBEConfig(input_dim=input_dim, n_qubits=n_qubits, n_layers=2, seed=seed)
    qvbe = QuantumVariationalBiomarkerEncoder(qvbe_config)
    qvbe.eval()

    # Process all samples through QVBE
    X_tensor = torch.tensor(X_proc)
    with torch.no_grad():
        qvbe_out = qvbe(X_tensor)

    embeddings = qvbe_out.embedding  # (N, n_qubits)
    logger.info(f"QVBE output: {tuple(embeddings.shape)}")

    # ------------------------------------------------------------------
    # 5. Construct a simulated temporal batch from the flat snapshots.
    # Each TADPOLE snapshot in BASELINE_MCI policy is a single-visit
    # observation (one baseline per patient). To exercise CTCLM's
    # temporal path, we stack multiple copies with synthetic time offsets
    # representing "visits at 0, 90, 180 days". This is a structural
    # connectivity test, NOT a real longitudinal sequence.
    # ------------------------------------------------------------------
    n = min(batch_size, len(embeddings))
    T = simulated_seq_len

    # Repeat embedding T times to form (B, T, D) sequence
    emb_batch = embeddings[:n].unsqueeze(1).expand(n, T, n_qubits)  # (B, T, D)

    # Simulated irregular time offsets (days): 0, 90, 180, ...
    time_offsets = torch.arange(T, dtype=torch.float32).unsqueeze(0) * 90.0
    time_offsets = time_offsets.expand(n, T)  # (B, T)

    # Simulate that some patients have fewer visits (last 1 step is padded)
    padding_mask = torch.ones(n, T, dtype=torch.bool)
    if n > 1 and T > 1:
        padding_mask[-1, T - 1:] = False  # last patient has 1 fewer visit

    # ------------------------------------------------------------------
    # 6. CTCLM forward pass
    # ------------------------------------------------------------------
    ctclm_config = CTCLMConfig(
        input_dim=n_qubits,
        latent_dim=latent_dim,
        time_embedding_dim=time_embedding_dim,
        num_layers=1,
        dropout=0.0,
        seed=seed,
    )
    ctclm = ContinualTemporalClinicalLatentModel(ctclm_config)
    ctclm.eval()

    with torch.no_grad():
        ctclm_out = ctclm(emb_batch.clone(), time_offsets, padding_mask)

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Pipeline Dimensions")
    print("-" * 70)
    print(f"  Input features        : {input_dim}")
    print(f"  QVBE n_qubits         : {n_qubits}")
    print(f"  QVBE embedding shape  : {tuple(embeddings.shape)}")
    print(f"  CTCLM batch size      : {n}")
    print(f"  CTCLM sequence len    : {T}")
    print(f"  CTCLM input dim       : {n_qubits}")
    print(f"  CTCLM latent dim      : {latent_dim}")
    print(f"  CTCLM time emb dim    : {time_embedding_dim}")
    print(f"  CTCLM parameters      : {ctclm.n_parameters}")

    print("\n" + "-" * 70)
    print("CTCLM Output")
    print("-" * 70)
    print(f"  sequence shape        : {tuple(ctclm_out.sequence.shape)}")
    print(f"  h_final shape         : {tuple(ctclm_out.h_final.shape)}")
    print(f"  seq_lengths           : {ctclm_out.seq_lengths.tolist()}")
    print(f"  h_final finite        : {torch.isfinite(ctclm_out.h_final).all().item()}")
    print(f"  sequence finite       : {torch.isfinite(ctclm_out.sequence).all().item()}")
    print(f"  h_final min/max       : {ctclm_out.h_final.min().item():.4f} / {ctclm_out.h_final.max().item():.4f}")

    assert ctclm_out.sequence.shape == (n, T, latent_dim), "sequence shape mismatch"
    assert ctclm_out.h_final.shape == (n, latent_dim), "h_final shape mismatch"
    assert torch.isfinite(ctclm_out.h_final).all(), "h_final contains non-finite values"
    assert torch.isfinite(ctclm_out.sequence).all(), "sequence contains non-finite values"

    print("\n" + "=" * 70)
    print("CTCLM SMOKE TEST PASSED")
    print("TADPOLE -> QVBE -> CTCLM -> temporal latent representation: OK")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTCLM smoke test on real TADPOLE data.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--time-embedding-dim", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=3)
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
        latent_dim=args.latent_dim,
        time_embedding_dim=args.time_embedding_dim,
        simulated_seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
