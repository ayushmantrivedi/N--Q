# NADIQ - PPMI REAL MRI COHORT EXPANSION PLAN V1
**Date**: 2026-09-24
**Status**: READY_FOR_LONI_ACQUISITION

## 1. TOTAL CANDIDATE POOL
The existing authorized `Xing_Core_Lab` MRI candidate pool contained exactly 3,235 subjects possessing a Baseline `BL` visit marked securely as `MRI_SEQ_T1_WEIGHTED == 'Yes'`. Excluding the already acquired, fully-validated original N=15 reference cohort left a pool of **3,220** unique candidates. 

## 2. ELIGIBILITY RULES
A candidate record is eligible for selection if it meets these constraints:
- Must represent a strict `BL` visit.
- Must not be part of the `REFERENCE_V2_15` cohort to avoid cross-contamination.
- Must explicitly feature a genuine structural 3D T1 MRI sequence flag in the upstream PPMI core lab metadata.

## 3. SELECTED TARGET N
**Target Expansion N = 250**
This subset safely satisfies the minimum data distribution needed to bootstrap robust 3D latent topological mapping within the β-VAE, circumventing the critical anatomical memorization flaw diagnosed in the N=15 cohort.

## 4. SELECTION METHODOLOGY
A strictly deterministic, non-biased uniform random sampling (Random Seed: `42`) was utilized on the subset to organically capture physical scanner heterogeneity (hardware variations, institutions, multi-phase study years) inherently present within the PPMI matrix, rather than artificially selecting sequentially. The system strictly avoids forced/fabricating label balancing.

## 5. ACQUISITION METADATA COVERAGE
Where available via the `Xing_Core_Lab` and generic `Magnetic_Resonance_Imaging` matrices, protocol/scanner indicators have been physically joined to the tracking manifest. The target cohort metadata tracks `PATNO`, `EVENT_ID`, random selection seed, and sequence type expectations.

## 6. BATCH STRUCTURE
The N=250 acquisition queue is statically bucketed into exact limits (e.g., `BATCH_001`, `BATCH_002`, `BATCH_003`... capping at 50 participants per batch). This facilitates safe download throughput given strict IDA/LONI portal bandwidth constraints and limits payload unpacking faults.

## 7. LONI ACQUISITION PROCEDURE REQUIRED FROM USER
**Action Required**: The automated pipeline has purposely frozen here to observe LONI data use agreements. You must manually execute the acquisition:
1. Extract the `PATNO` lists from the batches in `F:\NADI-Q data\PPMI\manifests\ppmi_mri_cohort_expansion_v1.csv`.
2. Navigate to the **PPMI LONI IDA Portal -> Search -> Advanced Search**.
3. Input the batch of PATNOs, select MRI T1-weighted filtering for the `Baseline` visit.
4. Add to the Image Collection and download the ZIP file.
5. Extract the ZIP into `F:\NADI-Q data\PPMI\MRI_raw\PPMI\`.

## 8. POST-DOWNLOAD VERIFICATION PROCEDURE
Once a batch is manually extracted to the raw directory, execute the local pipeline tool:
`python scripts/verify_mri_batch.py`
This tooling automatically iterates through newly detected DICOMs and validates them before integration.

## 9. V2 PREPROCESSING LIMITATIONS
The `PPMI_MRI_PREPROCESSING_V2_FROZEN` pipeline intentionally maintains fixed restrictions:
- **No Native N4 Bias Correction**: Missing `antspyx`.
- **No Native HD-BET Skull Stripping**: Uses basic structural nonzero limits.
- The pipeline solely focuses on native anatomical `pydicom` reading, deterministic RAS+ `nibabel` matrix mapping, isotropic spacing, and intensity stabilization `[0,1]`.

## 10. EXACT CRITERIA FOR PROMOTING TO `VERIFIED`
To achieve `VERIFIED` status in the manifest, the script checks:
- The target nested `PATNO` folder physically exists.
- At least 30 native `.dcm` files are present.
- `pydicom` metadata matches the `PatientID` to the manifest `PATNO`.
- Volumetric variance strictly `> 0`.
- Series metadata is not flagged as `DTI`, `rsfMRI`, or `T2/FLAIR`.

## 11. EXACT CRITERIA FOR PROMOTING TO `PREPROCESS_PENDING`
All subjects that successfully pass physical DICOM verification (`VERIFIED`) and present as intact eligible arrays (`ELIGIBLE`) immediately inherit the `PREPROCESS_PENDING` label within the CSV. At this phase, they are scientifically clear to safely pass into the Frozen V2 Python pipeline block. 

## 12. BATCH VERIFICATION HARDENING
The physical post-download verification logic (erify_mri_batch.py) has been upgraded from primitive file counting to strict series-level consistency:
- **DICOM Count is Descriptive**: A hard "30 file minimum" is completely removed. A sequence is validated by its ability to physically reconstruct into a volume.
- **Series-level Consistency**: Identifies multiple nested sequence protocols (e.g., separating localizers from volumetric acquisitions) and exclusively reconstructs the T1_WEIGHTED / MPRAGE candidate.
- **Volumetric Reconstruction**: Directly asserts physiological 3D spatial properties (dimensions, positive variances, absent NaN/Inf contamination) by physically arraying the DICOM pixel stacks in real-time.
- **Identity Enforcement**: Explicitly matches physical PatientID and SeriesInstanceUID inside DICOM headers against the expansion manifest. PATNO mismatches definitively trigger FAILED_IDENTITY.
- **Multiple Series Handling**: Should multiple T1 series exist, all are classified, and the primary valid series UID is explicitly designated in the tracking manifest.
