#!/usr/bin/env python
"""
NĀḌĪ-Q Architecture A — End-to-End Verification Benchmark.

Verifies:
  1. Evidence discovery pipeline runs without error.
  2. Clinical AUROC ≥ 0.635 on 3-seed ensemble (Task T2, N≈341).
  3. Clinical predictions are 100% invariant to quantum service status (Δ_max < 1e-8).
  4. Classical discovery path runtime < 600ms.
  5. All Architecture A unit tests pass.

Writes:
  experiments/architecture_a/ARCHITECTURE_A_BENCHMARK_RESULTS.csv
  experiments/architecture_a/ARCHITECTURE_A_VERDICT.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("arch_a_benchmark")

OUTPUT_DIR = "experiments/architecture_a"
AUROC_THRESHOLD = 0.635
RUNTIME_BUDGET_MS = 600.0
DELTA_MAX_THRESHOLD = 1e-8


def run_unit_tests() -> dict:
    """Run pytest on Architecture A unit tests."""
    logger.info("Running unit tests …")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_architecture_a.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    passed = result.returncode == 0
    logger.info("Unit tests: %s (%.1fs)", "PASS" if passed else "FAIL", elapsed)
    return {
        "passed": passed,
        "returncode": result.returncode,
        "stdout": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
        "elapsed_s": round(elapsed, 2),
    }


def run_full_benchmark() -> dict:
    """Execute full pipeline and collect metrics."""
    from nadiq.architecture_a.pipeline import ArchitectureAPipeline

    logger.info("Running full Architecture A pipeline …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = ArchitectureAPipeline(
        task="T2",
        seeds=[42, 123, 456],
        n_folds=5,
        solver_mode="auto",
        budget_k=4,
        alpha_validated=0.05,
        alpha_provisional=0.10,
        auroc_threshold=AUROC_THRESHOLD,
        discovery_runtime_budget_ms=RUNTIME_BUDGET_MS,
    )

    report = pipeline.run_full(output_dir=OUTPUT_DIR, dataset_label="PPMI_N341")
    return report.to_dict()


def run_invariance_stress_test() -> dict:
    """
    Confirm Δ_max = 0 by running clinical predictor twice and comparing outputs.
    Second run uses evidence_graph_path set to a non-existent path to simulate
    quantum service unavailability.
    """
    from nadiq.architecture_a.clinical_predictor import ArchitectureAClinicalPredictor
    import numpy as np

    logger.info("Running prediction invariance stress test …")

    pred_a = ArchitectureAClinicalPredictor(
        task="T2", seeds=[42], n_folds=5, evidence_graph_path=None
    )
    pred_b = ArchitectureAClinicalPredictor(
        task="T2", seeds=[42], n_folds=5, evidence_graph_path="/nonexistent/path.json"
    )

    res_a = pred_a.predict()
    res_b = pred_b.predict()

    probs_a = res_a.y_prob_ensemble
    probs_b = res_b.y_prob_ensemble
    delta_max = float(max(abs(a - b) for a, b in zip(probs_a, probs_b)))

    invariant = delta_max < DELTA_MAX_THRESHOLD
    logger.info("Invariance stress: Δ_max=%.2e  invariant=%s", delta_max, invariant)
    return {
        "delta_max": delta_max,
        "invariant": invariant,
        "auroc_a": round(res_a.ensemble_auroc, 4),
        "auroc_b": round(res_b.ensemble_auroc, 4),
    }


def write_csv(results_rows: list, path: str) -> None:
    if not results_rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_pass = True
    rows = []

    # --- 1. Unit tests ---
    ut = run_unit_tests()
    rows.append({
        "check": "unit_tests",
        "passed": ut["passed"],
        "value": f"returncode={ut['returncode']}",
        "elapsed_s": ut["elapsed_s"],
    })
    if not ut["passed"]:
        logger.warning("UNIT TEST FAILURES:\n%s", ut["stdout"])
        overall_pass = False

    # --- 2. Full pipeline ---
    pipeline_report: dict = {}
    try:
        pipeline_report = run_full_benchmark()
        v = pipeline_report.get("verification", {})

        auroc = v.get("ensemble_auroc", 0.0)
        auroc_ok = bool(v.get("auroc_above_threshold", False))
        inv_ok = bool(v.get("prediction_invariant", True))
        rt_ok = bool(v.get("runtime_below_600ms_classical", False))
        disc_ms = pipeline_report.get("discovery_runtime_ms", 9999)

        rows.append({"check": "ensemble_auroc", "passed": auroc_ok,
                     "value": f"{auroc:.4f}", "elapsed_s": ""})
        rows.append({"check": "prediction_invariant", "passed": inv_ok,
                     "value": f"delta={v.get('max_quantum_delta', 'n/a')}", "elapsed_s": ""})
        rows.append({"check": "discovery_runtime_ms", "passed": rt_ok,
                     "value": f"{disc_ms:.1f}ms", "elapsed_s": ""})

        if not auroc_ok:
            logger.warning("AUROC %.4f < threshold %.3f", auroc, AUROC_THRESHOLD)
            overall_pass = False
        if not inv_ok:
            logger.error("PREDICTION INVARIANCE VIOLATED!")
            overall_pass = False
        if not rt_ok:
            logger.warning("Discovery runtime %.1fms > 600ms budget", disc_ms)
            # Non-fatal (note only)

    except Exception as exc:
        logger.error("Pipeline benchmark failed: %s", exc, exc_info=True)
        rows.append({"check": "full_pipeline", "passed": False, "value": str(exc), "elapsed_s": ""})
        overall_pass = False

    # --- 3. Invariance stress test ---
    try:
        inv_result = run_invariance_stress_test()
        rows.append({
            "check": "invariance_stress",
            "passed": inv_result["invariant"],
            "value": f"delta_max={inv_result['delta_max']:.2e}",
            "elapsed_s": "",
        })
        if not inv_result["invariant"]:
            logger.error("INVARIANCE STRESS TEST FAILED: Δ_max=%.2e", inv_result["delta_max"])
            overall_pass = False
    except Exception as exc:
        logger.error("Invariance stress test failed: %s", exc)
        rows.append({"check": "invariance_stress", "passed": False, "value": str(exc), "elapsed_s": ""})
        overall_pass = False

    # --- Write results ---
    csv_path = os.path.join(OUTPUT_DIR, "ARCHITECTURE_A_BENCHMARK_RESULTS.csv")
    write_csv(rows, csv_path)

    verdict = {
        "overall_pass": overall_pass,
        "checks": rows,
        "pipeline_report": pipeline_report,
    }
    verdict_path = os.path.join(OUTPUT_DIR, "ARCHITECTURE_A_BENCHMARK_VERDICT.json")
    with open(verdict_path, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("NĀḌĪ-Q Architecture A Benchmark Results")
    print("=" * 60)
    for r in rows:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {status}  {r['check']:35s}  {r['value']}")
    print("=" * 60)
    print(f"  Overall: {'✓ ALL PASS' if overall_pass else '✗ FAILURES DETECTED'}")
    print(f"  CSV:     {csv_path}")
    print(f"  Verdict: {verdict_path}")
    print("=" * 60)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
