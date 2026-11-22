# Experiments Directory — Nāḍī-Q

This directory contains **reproducible experiment entry points** only.

## What belongs here

- Python scripts that run a defined experiment end-to-end.
- Each script corresponds to one or more experiment configs in `configs/experiments/`.
- Results are written to `experiments/runs/<experiment_id>_<timestamp>/`.

## What does NOT belong here

- Model definitions (those go in `src/nadiq/models/`)
- Dataset loading logic (those go in `src/nadiq/data/`)
- Utility functions (those go in `src/nadiq/utils/`)
- Exploratory analysis (use `notebooks/`)

## Conventions

1. Every experiment script takes `--config` as its primary argument.
2. Scripts must call `seed_all(cfg.seed)` as the first operation.
3. Scripts must log the full resolved config hash before running.
4. Test set evaluation happens exactly once, at the end.
5. All results are serialized as JSON and logged to MLflow.

## Current Status

The experiment runner (`run_experiment.py`) is a template.  
It will be implemented after core interfaces are complete.

## Runs

Experiment run artifacts are written to `experiments/runs/` (gitignored).  
Commit only the configuration files, not the run outputs.
