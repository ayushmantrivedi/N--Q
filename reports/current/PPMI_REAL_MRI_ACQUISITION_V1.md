# NADIQ - PPMI REAL MRI ACQUISITION V1
**Date**: 2026-09-24
**Status**: ACQUIRED & VERIFIED
**Inputs**: Local PPMI metadata + Manually Acquired LONI IDA DICOM payloads
**Prepared by**: Antigravity automated pipeline

---

## 1. ACQUISITION INVENTORY

A complete traverse of F:\NADI-Q data\PPMI\MRI_raw yielded:
- **Total MRI scans (series) found**: 15
- **Unique Subjects (PATNOs)**: 15
- **Valid 3D T1 Baseline Scans**: 15
- **Rejected Scans**: 0

### Rejection Summary
- No scans were rejected.

---

## 2. IMAGE METADATA SUMMARY (Valid Scans)

- **Dimensionality**:
  - 256x256x192: 11 scans
  - 256x256x193: 1 scans
  - 256x256x188: 1 scans
  - 256x256x152: 1 scans
  - 240x256x176: 1 scans
- **Voxel Spacing (X x Y x Z)**:
  - 1.00x1.00x1.00: 13 scans
  - 1.00x1.00x1.20: 1 scans
  - 1.05x1.05x1.20: 1 scans
- **Visit Distribution**: All valid scans are verified to be Baseline (BL).
- **Image Orientations**: 8 unique orient matrices found.

---

## 3. SCIENTIFIC INTERSECTION (REAL DATA CHECK)

Calculating the exact intersection of the 15 *manually acquired and verified* MRI subjects against the existing Baseline (BL) metadata cohorts.

- **MRI only**: 0 subjects
- **MRI ? Biomarker**: 15 subjects
- **MRI ? Gait (Opals/Axivity)**: 15 subjects
- **MRI ? Biomarker ? Gait (Triple Cohort)**: 15 subjects

> **Note**: The user indicated 24 subjects were acquired, but only 15 valid 3D T1 directories were present in the local filesystem. This report strictly reflects the exact files available on disk.

---

## 4. CLINICAL COHORT DISTRIBUTION

For the valid acquired MRI subjects:

**Generic Clinical State (COHORT)**:
- PRODROMAL: 11
- HC: 3
- PD: 1

**Curated Subgroup (subgroup)**:
- Hyposmia: 39
- Healthy Control: 19
- LRRK2 NMC: 8
- Sporadic PD: 5
- RBD: 4

---

## 5. EXACT USABLE PATNOs

The following PATNOs have verified, real 3D T1 MRI data available locally and are approved for downstream modeling:

`	ext
100232, 100445, 100712, 101178, 101463, 102245, 114551, 137259, 138755, 140451, 153191, 187759, 213013, 3004, 42012
`

---

## 6. MANIFEST & PROVENANCE

An exact file-by-file provenance manifest has been written to:
F:\NADI-Q data\PPMI\manifests\ppmi_mri_acquired_v1.csv

This manifest includes checksums, sequence metadata, voxel spacing, dimensionality, and explicit rejection reasons for any invalid files.
