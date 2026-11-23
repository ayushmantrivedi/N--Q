#!/usr/bin/env python
"""
NĀḌĪ-Q Architecture A — Clinical Prediction CLI (Plane 2).

Completely decoupled from quantum code.
Evidence graph path is used for display/logging ONLY — never for inference.

Usage:
  python scripts/nadiq_predict.py --task T2 --seeds 42 123 456 --output_dir results/
  python scripts/nadiq_predict.py --task T2 --evidence_graph results/evidence_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nadiq.architecture_a.clinical_predictor import ArchitectureAClinicalPredictor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="NĀḌĪ-Q Architecture A — Clinical Prediction (Plane 2, 100% classical)"
    )
    parser.add_argument(
        "--task",
        choices=["T1", "T2"],
        default="T2",
        help="Prediction task.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multi-seed ensemble.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/architecture_a",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--cohort_parquet",
        default=r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet",
    )
    parser.add_argument(
        "--cache_parquet",
        default=r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet",
    )
    parser.add_argument(
        "--evidence_graph",
        default=None,
        help="Path to evidence_graph.json (display/logging only — not used in inference).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression regularisation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    predictor = ArchitectureAClinicalPredictor(
        task=args.task,
        seeds=args.seeds,
        n_folds=args.n_folds,
        cohort_path=args.cohort_parquet,
        cache_path=args.cache_parquet,
        evidence_graph_path=args.evidence_graph,
        C=args.C,
    )

    result = predictor.predict()

    print(f"\n=== Clinical Prediction Complete ===")
    print(f"  Task             : {result.task}")
    print(f"  Seeds            : {result.seeds}")
    print(f"  Folds            : {result.n_folds}")
    print(f"  Per-seed AUROCs  : {[round(a, 4) for a in result.seed_aurocs]}")
    print(f"  Ensemble AUROC   : {result.ensemble_auroc:.4f}")
    print(f"  Ensemble AUPRC   : {result.ensemble_auprc:.4f}")
    print(f"  Ensemble Brier   : {result.ensemble_brier:.4f}")
    print(f"  Δ_max (quantum)  : {result.max_quantum_delta:.2e}  (invariant: {result.max_quantum_delta < 1e-8})")
    print(f"  AUROC ≥ 0.635    : {'YES ✓' if result.ensemble_auroc >= 0.635 else 'NO — investigate'}")

    if args.evidence_graph:
        print(f"  Evidence graph   : {args.evidence_graph} [display only, not used in inference]")

    # Save
    out_path = os.path.join(args.output_dir, "CLINICAL_PREDICTION_RESULT.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    print(f"  Results saved    : {out_path}")


if __name__ == "__main__":
    main()
