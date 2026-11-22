"""
Command Line Interface for Nāḍī-Q.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from nadiq.data.registry import DatasetRegistry


def handle_dataset_list(args: argparse.Namespace) -> None:
    registry = DatasetRegistry()
    
    # Pre-populate stubs
    # (In a real setup we might just query the registry that inherently knows about them)
    try:
        registry.get("adni") # force load stubs
    except KeyError:
        pass
        
    print("Registered datasets:")
    for ds_id in registry.list_registered():
        print(f"  - {ds_id}")
    
    # Also list stubs explicitly
    stubs = [
        "adni", "mimic_iv", "uk_biobank", "daphnet", "tadpole", 
        "oasis", "ppmi", "eicu", "nhanes", "parkinson_speech", "parkinson_telemonitoring"
    ]
    for stub in stubs:
        if stub not in registry:
            print(f"  - {stub} (restricted/stub)")


def handle_dataset_plan(args: argparse.Namespace) -> None:
    dataset_id = args.dataset
    registry = DatasetRegistry()
    
    try:
        loader = registry.get(dataset_id)
    except KeyError:
        print(f"Error: Dataset '{dataset_id}' not found.")
        sys.exit(1)
        
    # If it's a real loader with data (like synthetic)
    if hasattr(loader, "source"):
        source = getattr(loader, "source")
        if hasattr(source, "metadata"):
            meta = source.metadata()
            print(f"Dataset: {dataset_id.upper()}")
            print(f"Status: {meta.status}")
            print(f"Access: {meta.access_mode.name if hasattr(meta.access_mode, 'name') else meta.access_mode}")
            print(f"Bulk download: {'ENABLED' if meta.bulk_download_supported else 'DISABLED'}")
            print(f"Local data: {'REQUIRED' if meta.local_data_required else 'NOT FOUND'}")
            size = meta.estimated_size_gb
            print(f"Estimated local storage: {f'{size} GB' if size is not None else 'NOT AVAILABLE'}")
            print(f"Required action: {meta.required_action}")
            print("Network data downloaded: 0 bytes")
            return
            
    # For synthetic or other loaders without a source metadata
    print(f"Dataset: {dataset_id.upper()}")
    print("Status: READY")
    print("Access: LOCAL/SYNTHETIC")
    print("Bulk download: N/A")
    print("Local data: GENERATED IN-MEMORY")
    print("Estimated local storage: 0 GB")
    print("Required action: None")
    print("Network data downloaded: 0 bytes")


def handle_experiment_run(args: argparse.Namespace) -> None:
    import subprocess
    from pathlib import Path
    
    repo_root = Path(__file__).parent.parent.parent
    script_path = repo_root / "experiments" / "run_experiment.py"
    
    cmd = [sys.executable, str(script_path), "--config", args.config]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def handle_benchmark(args: argparse.Namespace) -> None:
    dataset_id = args.dataset
    if dataset_id != "synthetic_tabular_v1":
        print(f"Error: {dataset_id.upper()} access is not configured. No data was downloaded.")
        sys.exit(1)
    
    print(f"Running benchmark on {dataset_id}...")
    # This would map to a specific benchmark runner, we just print for now.


def main() -> None:
    parser = argparse.ArgumentParser(description="Nāḍī-Q CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # dataset
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    ds_subparsers = dataset_parser.add_subparsers(dest="dataset_cmd", required=True)
    
    ds_list_parser = ds_subparsers.add_parser("list", help="List available datasets")
    ds_list_parser.set_defaults(func=handle_dataset_list)
    
    ds_plan_parser = ds_subparsers.add_parser("plan", help="Dry-run a dataset query")
    ds_plan_parser.add_argument("dataset", type=str, help="Dataset ID")
    ds_plan_parser.set_defaults(func=handle_dataset_plan)
    
    # experiment
    exp_parser = subparsers.add_parser("experiment", help="Experiment operations")
    exp_subparsers = exp_parser.add_subparsers(dest="exp_cmd", required=True)
    
    exp_run_parser = exp_subparsers.add_parser("run", help="Run an experiment")
    exp_run_parser.add_argument("--config", type=str, required=True, help="Path to config")
    exp_run_parser.set_defaults(func=handle_experiment_run)
    
    # benchmark
    bm_parser = subparsers.add_parser("benchmark", help="Benchmark operations")
    bm_parser.add_argument("--dataset", type=str, required=True, help="Dataset ID")
    bm_parser.add_argument("--task", type=str, required=True, help="Task name")
    bm_parser.add_argument("--model", type=str, required=True, help="Model class")
    bm_parser.set_defaults(func=handle_benchmark)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
