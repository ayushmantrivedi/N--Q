# NADIQ - FINAL BIOMARKER DUPLICATE / PROVENANCE AUDIT V1
**Date**: 2026-09-24
**Status**: REQUIRES REVISION -> FIXED -> READY

## 1. INVESTIGATION OF MULTIPLE RECORDS

An investigation into the raw rows of the 15-subject cohort at Baseline revealed two distinct types of multiple records for a single biomarker:

1. **Identical Measurements (e.g., ABeta40, ABeta42, pTau181)**: The TESTVALUE, UNITS, and assay origins were identical. The only distinguishing feature was the update_stamp (e.g., 2025-03-13 vs 2025-05-27), indicating database-level metadata updates.
2. **Distinct Descriptive Statistics (e.g., NFL, GFAP)**: The raw records contained one row where UNITS = Average (pg/ml) and another row where UNITS = Stdev. 

**Critical Scientific Finding**:
The previous aggregation rule simply executed alid_numeric.mean(). For NFL and GFAP, this resulted in averaging the participant's actual plasma average with its own standard deviation (e.g., (26.02 + 0.55) / 2). This violates fundamental statistics and corrupts the feature space. **(Option C: distinct biological/specimen measurements that should NOT be averaged)**.

## 2. REVISED AGGREGATION RULE

The pipeline rule has been updated dynamically:
- Records with UNITS containing stdev (case-insensitive) are explicitly filtered out before aggregation.
- For records that are identical in value but differ in database timestamps (update_stamp), a mean() or exact duplicate drop is safely applied.

## 3. PROVENANCE TABLE (Duplicate Handlings)
|   PATNO | canonical biomarker   |   raw record count | record identifiers/distinguishing fields   | aggregation decision   | reason                                                                  |   final value |
|--------:|:----------------------|-------------------:|:-------------------------------------------|:-----------------------|:------------------------------------------------------------------------|--------------:|
|  100232 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      7521     |
|  100232 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       741     |
|  100232 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        25.2   |
|  100445 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      5777     |
|  100445 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       578     |
|  100445 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        21.4   |
|  100445 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        26.026 |
|  100445 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |       189.569 |
|  100445 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.201 |
|  101178 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      5462     |
|  101178 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       502     |
|  101178 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        16.9   |
|  101178 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |         9.105 |
|  101178 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        58.414 |
|  101178 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.11  |
|  101463 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      7937     |
|  101463 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       564     |
|  101463 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        37.3   |
|  101463 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        10.44  |
|  101463 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        93.244 |
|  101463 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.173 |
|  102245 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      7460     |
|  102245 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       612     |
|  102245 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        29.4   |
|  102245 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        13.235 |
|  102245 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        83.613 |
|  102245 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.128 |
|  114551 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |     10965     |
|  114551 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      1068     |
|  114551 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        30.2   |
|  114551 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        17.168 |
|  114551 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |       123.772 |
|  114551 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.23  |
|  137259 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      8709     |
|  137259 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       733     |
|  137259 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        27     |
|  138755 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      7941     |
|  138755 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       610     |
|  138755 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        28.7   |
|  153191 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      7151     |
|  153191 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       406     |
|  153191 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        25.9   |
|  153191 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        17.544 |
|  153191 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |       160.472 |
|  213013 | nfl                   |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |         8.88  |
|  213013 | gfap                  |                  2 | UNITS (Average vs Stdev)                   | Filtered Stdev         | Averaging an Average and a Standard Deviation is mathematically invalid |        74.812 |
|  213013 | ptau217               |                  3 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |         0.139 |
|    3004 | abeta40               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |      9844     |
|    3004 | abeta42               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |       952     |
|    3004 | ptau181               |                  2 | update_stamp                               | Mean / Identity        | Measurements are identical; differing only by timestamp                 |        22.3   |

## 4. STATUS

**READY** ? duplicate records are now scientifically safe to aggregate.
