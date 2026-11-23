#!/usr/bin/env python
"""
NĀḌĪ-Q Architecture A — Evidence Discovery CLI.

Usage:
  python scripts/nadiq_discover.py \\
      --cohort_parquet "F:\\NADI-Q_AUDIT_OUTPUT\\PHASE_7\\ppmi_longitudinal_multimodal_biofluid_n341.parquet" \\
      --cache_parquet  "F:\\NADI-Q_AUDIT_OUTPUT\\PHASE_2\\curated_cut_cache.parquet" \\
      --output_dir results/ \\
      --solver tabu \\
      --budget 4 \\
      --fdr_alpha 0.05

Outputs:
  <output_dir>/evidence_graph.json
  <output_dir>/evidence_graph.metta
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from nadiq.architecture_a.evidence_discovery import EvidenceDiscoveryPipeline, FEATURE_NAMES
from nadiq.architecture_a.evidence_exporter import EvidenceGraphExporter
from nadiq.experiments.phase14a.data_protocol import DataProtocol14A


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="NĀḌĪ-Q Architecture A — Biological Evidence Discovery"
    )
    parser.add_argument(
        "--cohort_parquet",
        default=r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_7\ppmi_longitudinal_multimodal_biofluid_n341.parquet",
        help="Path to PPMI longitudinal cohort parquet.",
    )
    parser.add_argument(
        "--cache_parquet",
        default=r"F:\NADI-Q_AUDIT_OUTPUT\PHASE_2\curated_cut_cache.parquet",
        help="Path to curated visit cache parquet.",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/architecture_a",
        help="Output directory for evidence graph files.",
    )
    parser.add_argument(
        "--solver",
        choices=["auto", "tabu", "exact", "qaoa_p1", "qaoa_p2"],
        default="auto",
        help="QUBO solver mode (default: auto → exact if N≤12 else tabu).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=4,
        help="Number of relationships to select (K).",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="BH-FDR strict threshold for 'Validated' tier.",
    )
    parser.add_argument(
        "--fdr_alpha_provisional",
        type=float,
        default=0.10,
        help="BH-FDR lenient threshold for 'Provisional' tier.",
    )
    parser.add_argument(
        "--task",
        choices=["T1", "T2"],
        default="T2",
        help="Task landmark for feature extraction (T2: Month-12 → Month-60).",
    )
    parser.add_argument(
        "--qaoa_timeout",
        type=float,
        default=15.0,
        help="QAOA background thread timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--dataset_label",
        default="PPMI_N341",
        help="Dataset label embedded in evidence graph JSON.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    dp = DataProtocol14A(
        cohort_path=args.cohort_parquet,
        cache_path=args.cache_parquet,
    )
    X_raw, y, pats = dp.get_task_data(task=args.task)
    print(f"Loaded cohort: N={len(pats)}, positives={int(y.sum())} ({100.0*y.mean():.1f}%)")

    # Run discovery
    pipeline = EvidenceDiscoveryPipeline(
        solver_mode=args.solver,
        budget_k=args.budget,
        alpha_validated=args.fdr_alpha,
        alpha_provisional=args.fdr_alpha_provisional,
        qaoa_timeout_sec=args.qaoa_timeout,
    )
    result = pipeline.run(X_raw)

    # Export
    exporter = EvidenceGraphExporter(result, dataset_label=args.dataset_label)
    json_path = exporter.to_json(os.path.join(args.output_dir, "evidence_graph.json"))
    metta_path = exporter.to_metta_file(os.path.join(args.output_dir, "evidence_graph.metta"))

    # Print summary
    summary = exporter.to_summary_dict()
    print(f"\n=== Evidence Discovery Complete ===")
    print(f"  Solver         : {result.solver_mode_used}")
    print(f"  Candidates     : {result.n_candidates_total} total  ({result.n_candidates_validated} Validated, {result.n_candidates_provisional} Provisional)")
    print(f"  FDR fallback   : {'YES' if result.fdr_fallback_triggered else 'no'}")
    print(f"  Selected edges : {len(result.selected_edges)}")
    print(f"  Runtime        : {result.discovery_runtime_ms:.1f} ms")
    print(f"  JSON output    : {json_path}")
    print(f"  MeTTa output   : {metta_path}")
    print(f"\n  Selected relationships:")
    for e in result.selected_edges:
        print(f"    [{e.status:20s}]  {e.src} → {e.dst}  r={e.effect_r:+.3f}  p={e.p_val:.4f}")
    print(f"\n  Statistical quality: PROVISIONAL/EXPLANATORY (not definitive causal biomarkers)")


if __name__ == "__main__":
    main()
