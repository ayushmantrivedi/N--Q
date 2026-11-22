# DATA ACQUISITION V2 FINAL REPORT

## 1. MRI - OpenNeuro ds001907
- **Dataset**: ds001907
- **Version**: Verified via OpenNeuro GitHub tree
- **File Count**: 18 T1w files total in dataset
- **Valid T1 Volumes**: 0 (Sandbox network execution halted / Data queued for direct host acquisition)
- **Subjects**: e.g., sub-RC4101
- **Total Bytes**: 0 downloaded locally. Expected: ~2-5 GB.
- **Status**: **FAIL**
- **Action Required**: Must execute local download runner outside the sandbox constraint.
- **Continuation Command**: `python scripts/download_real_pd_data.py`

## 2. KINEMATICS - PhysioNet gaitpdb
- **Dataset**: gait-in-parkinsons-disease-1.0.0
- **Recordings**: 0 (Sandbox zip extraction failed due to `IncompleteRead` network drop)
- **Subjects**: 166 total documented
- **Channels**: 16 force sensors
- **Total Bytes**: 0 valid bytes downloaded locally. Expected: ~302 MB.
- **Status**: **FAIL**
- **Action Required**: Must execute resumable downloader locally to bypass network chunking limits.
- **Continuation Command**: `python scripts/download_real_pd_data.py`

## 3. BIOMARKERS - UCI Parkinson's Acoustic Biomarkers
- **Dataset**: UCI ML Parkinson's Disease Data Set
- **Rows**: 195 (Expected)
- **Subjects**: 31 (Expected)
- **Biomarker Columns**: 22 numeric continuous features (Expected)
- **Total Bytes**: 0 downloaded locally. Expected: ~40 KB.
- **Status**: **FAIL**
- **Action Required**: Run the local runner to fetch and parse the CSV.
- **Continuation Command**: `python scripts/download_real_pd_data.py`

## SUMMARY
The discovery scripts successfully routed and validated the authoritative sources for the three modalities, locating the precise NIfTI paths, valid Zip headers, and an authentic open-access Biomarker dataset. 

However, because I am strictly forbidden from mocking data, using placeholders, or faking validation results, the acquisition phase terminates here with a `FAIL` status because the complete physical payloads could not be fully downloaded to the disk inside this environment.

No training logic has been executed. No synthetic data has been generated. The data acquisition gates have successfully caught the missing payload.

**Local Windows Execution Command:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\local_prepare_and_train.ps1
```
