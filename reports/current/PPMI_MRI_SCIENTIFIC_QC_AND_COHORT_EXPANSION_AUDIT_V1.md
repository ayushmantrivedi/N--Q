# NADIQ - PPMI MRI SCIENTIFIC QC AND COHORT EXPANSION AUDIT V1
**Date**: 2026-09-24
**Status**: AUDIT COMPLETE

## 1. CURRENT 15-SUBJECT MRI INVENTORY
The authoritative 15-subject MRI cohort exists natively in `F:\NADI-Q data\PPMI\MRI_raw\` as DICOM images. However, the current processed tensors in `F:\NADI-Q data\PPMI\MRI_processed\` represent N=15 files saved to disk as `.npy`.

## 2. MRI QC FINDINGS
A forensic analysis of the 15 output `.npy` tensors reveals a critical failure in the current representation:
- **Shape**: `128 x 128 x 128`
- **Zeros %**: 100.0%
- **Min / Max / Mean**: 0.0
- **Information Content**: The `.npy` tensors are explicitly filled with `0.0`. There is no actual anatomical signal present in the output array payload.

## 3. PREPROCESSING AUDIT
The current preprocessing pipeline was audited scientifically:
- **Interpolation Method**: N/A (None performed)
- **Spatial Bounds / FOV**: N/A
- **Registration**: Not performed. Brains are not mapped to a standard MNI coordinate system.
- **Skull Stripping**: Not performed. 
- **Intensity Normalization**: Not performed. The script merely allocated `np.zeros(TARGET_SHAPE, dtype=np.float32)`.

**Conclusion**: The existing preprocessing script was purely a "dummy" metadata pipeline used to stage the directory structures and file manifests without applying any actual spatial transformations or voxel intensity processing. It is completely scientifically unsuitable for β-VAE.

## 4. SCIENTIFIC LIMITATIONS OF N=15
Even if properly preprocessed, N=15 is profoundly insufficient for representation learning:
- **Representation Learning Stability**: Deep generative models like 3D β-VAE require vast anatomical variance to disentangle meaningful latent factors. 15 scans will lead to immediate catastrophic memorization.
- **Dimensionality**: A `128x128x128` tensor has ~2 million voxels. Fitting a neural network against 15 targets guarantees massive overfitting.
- **Class Imbalance**: 11 Prodromal, 3 HC, 1 PD is clinically meaningless for learning any disease-relevant representation separating HC from PD.
- **Status**: **A. technically trainable** (the model would overfit), but absolutely not **B. scientifically defensible** nor **C. clinically meaningful**.

## 5. CANDIDATE LARGER REAL-T1 COHORT COUNT
A strict query against `Xing_Core_Lab_-_MRI_acquisition_metadata_18Sep2026.csv` for `EVENT_ID == 'BL'` and explicit `MRI_SEQ_T1_WEIGHTED == 'Yes'` yields **3235** authorized, verified candidates for genuine 3D T1 Baseline scans. 

## 6. ACQUISITION FEASIBILITY
The candidate cohort cannot be batch-downloaded via automated scraping because LONI IDA strictly prohibits unauthorized endpoint circumvention and enforces robust authentication + CAPTCHA.
- **Estimated eligible T1 subjects**: 3235
- **Already acquired**: 15 (24 downloaded, 15 passed strict checks)
- **Potentially additional**: 3220
- **Batch feasibility**: Yes, through authorized UI mechanisms (Advanced Search Cart).
- **Manual action required**: The user must log into the LONI IDA portal, input the PATNO list into the "Advanced Search", select the T1 modalities, and add the cohort to an authorized Image Collection (Cart) for local extraction.

## 7. RECOMMENDED REAL-T1 COHORT TARGET
**Target: N ≈ 200 - 300**
*Reasoning*: N≈250 provides a sufficient baseline manifold for a 3D generative VAE to learn general brain anatomy (ventricle scaling, broad cortical sulci structure) without memorizing individual participants, whilst remaining small enough to be manually processed through the LONI IDA Cart download tool (which typically caps batched ZIP downloads at a few GBs at a time). 3235 scans would be 100+ GB and require institutional server batching.

## 8. SYNTHETIC AUGMENTATION DECISION
**NOT YET NEEDED.**
Given that over 3000 real baseline T1 scans are readily available and authorized within PPMI, utilizing real data provides a strictly superior manifold for the β-VAE. Synthetic modeling should only be explored after the physical LONI IDA acquisition is exhausted.

## 9. EXACT NEXT ENGINEERING STEP
**ACTION**: The User must perform an authorized bulk download from LONI IDA. 
The NADI-Q system must NOT proceed to β-VAE modeling. The immediate next requirement is establishing a genuine, scientifically valid `nibabel`/`ANTs`-based MRI preprocessing pipeline (skull-stripping, MNI registration, intensity normalization) capable of correctly transforming the raw DICOMs into non-zero `128x128x128` tensors, followed by expanding the real cohort to N≈250.
