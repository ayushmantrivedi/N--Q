# NADIQ - PPMI BIOMARKER AUDIT V1
**Date**: 2026-09-24
**Status**: AUDITED (Failure Identified)
**Conclusion**: **B. a mapping problem**

---

## 1. AUDIT FINDINGS

The previous preprocessing pipeline exhibited 100% missingness across all priority biomarkers for the 15-subject MRI cohort. This audit explicitly confirms that **the missingness was an artificial mapping failure, not genuine missingness**.

### Root Cause
The previous pipeline searched the Current_Biospecimen_Analysis_Results_18Sep2026.csv file using outdated/canonical variable aliases (e.g., IU_ABeta42_CSF, IU_NFL_plasma_mean). However, the actual dataset explicitly stores the TESTNAME under simpler, direct strings like ABeta42, NFL, and pTau181.

Because the strings did not match verbatim, the Pandas filtering step yielded empty DataFrames for every single marker, causing the pipeline to fall back to NaN.

---

## 2. RAW DATA VERIFICATION

A raw string search of the TESTNAME column for the 15 verified PATNOs revealed robust data availability for the following actual test names:
- ABeta40
- ABeta42
- pTau181
- NFL
- GFAP
- Ptau217p
- eMTBR-TAU243

---

## 3. FAILED MAPPING TABLE

Below is a detailed trace of the mapping failure, showing exactly how the canonical names failed to capture the raw data that was actually present on disk:

|   PATNO | canonical biomarker   | raw TESTNAME   |   raw value count |   mapped value count | units           |   final value | reason missing                                                        |
|--------:|:----------------------|:---------------|------------------:|---------------------:|:----------------|--------------:|:----------------------------------------------------------------------|
|  100232 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  100232 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       7521    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  100232 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        741    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  100232 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         25.2  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  100232 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  100232 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  100232 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  100232 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  100445 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |         18.06 | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  100445 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       5777    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  100445 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        578    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  100445 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         21.4  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  100445 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |         13.29 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  100445 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |        108.06 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  100445 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.2  | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  100445 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  100712 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  100712 | IU_ABeta40_CSF        | ABeta40        |                 1 |                    0 | pg/mL           |       5515    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  100712 | IU_ABeta42_CSF        | ABeta42        |                 1 |                    0 | pg/mL           |        496    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  100712 | IU_pTau181_CSF        | pTau181        |                 1 |                    0 | pg/mL           |         19    | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  100712 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  100712 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  100712 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  100712 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  101178 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  101178 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       5462    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  101178 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        502    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  101178 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         16.9  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  101178 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          4.66 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  101178 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         31.71 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  101178 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.11 | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  101178 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  101463 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  101463 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       7937    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  101463 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        564    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  101463 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         37.3  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  101463 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          5.25 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  101463 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         50.31 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  101463 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.17 | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  101463 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  102245 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |         37.21 | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  102245 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       7460    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  102245 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        612    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  102245 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         29.4  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  102245 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          6.65 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  102245 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         42.23 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  102245 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.13 | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  102245 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  114551 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |         34.6  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  114551 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |      10965    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  114551 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |       1068    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  114551 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         30.2  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  114551 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          8.62 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  114551 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         63.26 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  114551 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.23 | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  114551 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  137259 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  137259 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       8709    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  137259 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        733    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  137259 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         27    | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  137259 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  137259 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  137259 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  137259 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  138755 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |         21.93 | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  138755 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       7941    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  138755 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        610    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  138755 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         28.7  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  138755 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  138755 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  138755 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  138755 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  140451 | MTBR_TAU243           | eMTBR-TAU243   |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for eMTBR-TAU243                            |
|  140451 | IU_ABeta40_CSF        | ABeta40        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta40                                 |
|  140451 | IU_ABeta42_CSF        | ABeta42        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta42                                 |
|  140451 | IU_pTau181_CSF        | pTau181        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for pTau181                                 |
|  140451 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  140451 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  140451 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  140451 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  153191 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  153191 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       7151    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  153191 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        406    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  153191 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         25.9  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  153191 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          8.83 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  153191 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         85.06 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  153191 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  153191 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  187759 | MTBR_TAU243           | eMTBR-TAU243   |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for eMTBR-TAU243                            |
|  187759 | IU_ABeta40_CSF        | ABeta40        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta40                                 |
|  187759 | IU_ABeta42_CSF        | ABeta42        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta42                                 |
|  187759 | IU_pTau181_CSF        | pTau181        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for pTau181                                 |
|  187759 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|  187759 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|  187759 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|  187759 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|  213013 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |         27.02 | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|  213013 | IU_ABeta40_CSF        | ABeta40        |                 1 |                    0 | pg/mL           |      10543    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|  213013 | IU_ABeta42_CSF        | ABeta42        |                 1 |                    0 | pg/mL           |        982    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|  213013 | IU_pTau181_CSF        | pTau181        |                 1 |                    0 | pg/mL           |         37.1  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|  213013 | IU_NFL_plasma_mean    | NFL            |                 2 |                    0 | Average (pg/ml) |          4.81 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|  213013 | IU_GFAP_plasma_mean   | GFAP           |                 2 |                    0 | Average (pg/ml) |         37.62 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|  213013 | Lilly_ptau217p        | Ptau217p       |                 3 |                    0 | pg/mL           |          0.14 | Pipeline searched for 'Lilly_ptau217p' but actual name is 'Ptau217p'  |
|  213013 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|    3004 | MTBR_TAU243           | eMTBR-TAU243   |                 1 |                    0 | pg/mL           |          7.5  | Pipeline searched for 'MTBR_TAU243' but actual name is 'eMTBR-TAU243' |
|    3004 | IU_ABeta40_CSF        | ABeta40        |                 2 |                    0 | pg/mL           |       9844    | Pipeline searched for 'IU_ABeta40_CSF' but actual name is 'ABeta40'   |
|    3004 | IU_ABeta42_CSF        | ABeta42        |                 2 |                    0 | pg/mL           |        952    | Pipeline searched for 'IU_ABeta42_CSF' but actual name is 'ABeta42'   |
|    3004 | IU_pTau181_CSF        | pTau181        |                 2 |                    0 | pg/mL           |         22.3  | Pipeline searched for 'IU_pTau181_CSF' but actual name is 'pTau181'   |
|    3004 | IU_NFL_plasma_mean    | NFL            |                 1 |                    0 | PG/ML           |         56.61 | Pipeline searched for 'IU_NFL_plasma_mean' but actual name is 'NFL'   |
|    3004 | IU_GFAP_plasma_mean   | GFAP           |                 1 |                    0 | NG/ML           |          3.91 | Pipeline searched for 'IU_GFAP_plasma_mean' but actual name is 'GFAP' |
|    3004 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|    3004 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |
|   42012 | MTBR_TAU243           | eMTBR-TAU243   |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for eMTBR-TAU243                            |
|   42012 | IU_ABeta40_CSF        | ABeta40        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta40                                 |
|   42012 | IU_ABeta42_CSF        | ABeta42        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for ABeta42                                 |
|   42012 | IU_pTau181_CSF        | pTau181        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for pTau181                                 |
|   42012 | IU_NFL_plasma_mean    | NFL            |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for NFL                                     |
|   42012 | IU_GFAP_plasma_mean   | GFAP           |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for GFAP                                    |
|   42012 | Lilly_ptau217p        | Ptau217p       |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for Ptau217p                                |
|   42012 | CSFSAA                | CSF SAA        |                 0 |                    0 | N/A             |        nan    | Genuine missingness at BL for CSF SAA                                 |

---

## 4. RESOLUTION PLAN

To resolve this issue, the preprocessing pipeline must be updated to use the exact TESTNAME strings found in the raw data, and 	ests/data/test_ppmi_preprocessing.py has been updated to assert that missingness does not exceed expected genuine thresholds.
