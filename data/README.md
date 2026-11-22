# Data Directory — Nāḍī-Q

This directory contains **data access infrastructure only**.

## What is NOT stored here

- Raw patient data
- Clinical records
- Identifiable health information
- Downloaded datasets

## What IS stored here (eventually)

- Processed splits (anonymized, after ethical approval)
- Dataset manifests (listing expected files)
- Checksums for data integrity verification

## Data Access

All datasets are accessed through `nadiq.data.registry.DatasetRegistry`.  
Dataset paths are specified in `configs/data/*.yaml`.  
No model code should reference data paths directly.

## Adding a New Dataset

1. Obtain ethical/legal data access approval.
2. Write a loader class in `src/nadiq/data/loaders/`.
3. Add a config file in `configs/data/<dataset_id>.yaml`.
4. Register the loader in tests.
5. Document the dataset in `docs/data_contract.md`.

## Synthetic Data

Synthetic data for testing is generated on-the-fly by  
`nadiq.data.loaders.synthetic.SyntheticTabularLoader`.  
It is never written to disk in this directory.
