import os
import pandas as pd

def main():
    checklist_path = r"F:\NADI-Q data\PPMI\manifests\ppmi_mri_batch_001_download_batch_002_v1.csv"
    report_drive_path = r"F:\NADI-Q data\PPMI\reports\NADIQ_PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md"
    report_repo_path = r"C:\Users\ayush\OneDrive\Desktop\Nadi-Q\reports\current\PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md"
    
    entries = [
        {
            "PATNO": 235630,
            "EVENT_ID": "BL",
            "expected_modality": "MRI",
            "expected_sequence": "3D T1-weighted",
            "ida_reference_type": "dataSetId",
            "ida_reference_id": "10318165",
            "ida_result_url": "https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=10318165",
            "download_status": "PENDING_DOWNLOAD",
            "local_directory": r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI\235630\\",
            "verification_status": "PENDING",
            "notes": "Candidate baseline 3D T1-weighted series in LONI/IDA; awaiting manual download by authorized user."
        },
        {
            "PATNO": 214284,
            "EVENT_ID": "BL",
            "expected_modality": "MRI",
            "expected_sequence": "3D T1-weighted",
            "ida_reference_type": "dataSetId",
            "ida_reference_id": "10294679",
            "ida_result_url": "https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=10294679",
            "download_status": "PENDING_DOWNLOAD",
            "local_directory": r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI\214284\\",
            "verification_status": "PENDING",
            "notes": "Candidate baseline 3D T1-weighted series in LONI/IDA; awaiting manual download by authorized user."
        },
        {
            "PATNO": 183051,
            "EVENT_ID": "BL",
            "expected_modality": "MRI",
            "expected_sequence": "3D T1-weighted",
            "ida_reference_type": "seriesId",
            "ida_reference_id": "1206857",
            "ida_result_url": "https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&seriesId=1206857",
            "download_status": "PENDING_DOWNLOAD",
            "local_directory": r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI\183051\\",
            "verification_status": "PENDING",
            "notes": "Candidate baseline 3D T1-weighted series in LONI/IDA; awaiting manual download by authorized user."
        },
        {
            "PATNO": 156304,
            "EVENT_ID": "BL",
            "expected_modality": "MRI",
            "expected_sequence": "3D T1-weighted",
            "ida_reference_type": "dataSetId",
            "ida_reference_id": "11082062",
            "ida_result_url": "https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=11082062",
            "download_status": "PENDING_DOWNLOAD",
            "local_directory": r"F:\NADI-Q data\PPMI\MRI image download_dataset\PPMI\156304\\",
            "verification_status": "PENDING",
            "notes": "Candidate baseline 3D T1-weighted series in LONI/IDA; awaiting manual download by authorized user."
        }
    ]
    
    df_batch = pd.DataFrame(entries)
    df_batch.to_csv(checklist_path, index=False)
    print(f"Checklist saved to: {checklist_path}")
    
    # Generate report markdown
    report_content = """# NĀḌĪ-Q — LONI/IDA DOWNLOAD CHECKLIST: BATCH 001 SUB-BATCH 002
**Document**: Batch 001 Download Sub-Batch 002 Preparation  
**Date**: 2026-11-02 13:30:00  
**Status**: Authorized Manual Download Preparation (Read-Only Planning)  
**Safety & Compliance**: Zero automated scraping or credential bypass. User will perform manual download via active authorized LONI/IDA session.

---

## 1. COHORT RECONCILIATION SUMMARY (N=24 BATCH 001)

| Cohort Category | Count | Participant Identifiers | Status in Pipeline |
| :--- | :---: | :--- | :--- |
| **Locally Verified Baseline 3D T1** | **10** | `149006`, `151050`, `170472`, `183263`, `220494`, `222893`, `229409`, `278950`, `311004`, `315185` | `PREPROCESS_PENDING` (Verified 3D T1) |
| **Prepared for Manual Download (This Batch)** | **4** | `156304`, `183051`, `214284`, `235630` | `SELECTED` / `PENDING_DOWNLOAD` |
| **Remaining Unlocated Candidates** | **10** | `111801`, `126138`, `200133`, `345635`, `398345`, `412845`, `444452`, `450524`, `464544`, `533529` | `SELECTED` (Awaiting LONI/IDA search) |
| **Total Approved Batch 001 Cohort** | **24** | *100% verified baseline candidates in `ppmi_mri_batch_001_corrected_v1.csv`* | Intact & Unchanged |

---

## 2. PREPARED LONI/IDA DOWNLOAD ENTRIES (N=4)

These 4 participants have candidate baseline 3D T1-weighted series identified in LONI/IDA. They are to be downloaded manually by the authorized user and extracted into the specified local destination directories.

| PATNO | Expected Visit | Expected Modality & Sequence | IDA Reference Type | IDA Reference ID | Direct LONI/IDA Link | Local Destination Folder | Initial Pipeline Status |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| `235630` | `BL` | MRI (3D T1-weighted) | `dataSetId` | `10318165` | [LONI IDA dataSetId=10318165](https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=10318165) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\235630\\` | `PENDING` |
| `214284` | `BL` | MRI (3D T1-weighted) | `dataSetId` | `10294679` | [LONI IDA dataSetId=10294679](https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=10294679) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\214284\\` | `PENDING` |
| `183051` | `BL` | MRI (3D T1-weighted) | `seriesId` | `1206857` | [LONI IDA seriesId=1206857](https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&seriesId=1206857) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\183051\\` | `PENDING` |
| `156304` | `BL` | MRI (3D T1-weighted) | `dataSetId` | `11082062` | [LONI IDA dataSetId=11082062](https://ida.loni.usc.edu/pages/access/studyNavigator.jsp?project=PPMI&dataSetId=11082062) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\156304\\` | `PENDING` |

> [!IMPORTANT]
> **Scientific Integrity & Verification Protocol**:
> The presence of a candidate record in LONI/IDA search results does **not** grant verified status. These series are strictly `PENDING_DOWNLOAD`. No files have been downloaded or modified. Once the user completes downloading and unzipping them into the destination paths, they will undergo physical 3D acquisition verification using `scripts/verify_mri_batch.py` (checking `MRAcquisitionType == '3D'`, slice thickness $\le 2.0$ mm, finite voxel reconstruction, non-zero variance, and baseline date match).

---

## 3. REMAINING UNLOCATED PARTICIPANTS REQUIRING LONI/IDA SEARCHES (N=10)

The following 10 Batch 001 participants still require manual searching on LONI/IDA for eligible baseline 3D T1 structural acquisitions. Metadata from PPMI official records (`Xing_Core_Lab` and `Magnetic_Resonance_Imaging`) is provided to guide the search:

| PATNO | Target Visit | Baseline MRI Date (PPMI Records) | Protocol | Reported T1 Sequence | Completed Status | Local Destination Directory |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `126138` | `BL` | 09/2023 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\126138\\` |
| `111801` | `BL` | 12/2023 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\111801\\` |
| `533529` | `BL` | 04/2026 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\533529\\` |
| `450524` | `BL` | 04/2026 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\450524\\` |
| `444452` | `BL` | 07/2025 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\444452\\` |
| `200133` | `BL` | 09/2023 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\200133\\` |
| `464544` | `BL` | 01/2026 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\464544\\` |
| `398345` | `BL` | 03/2026 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\398345\\` |
| `345635` | `BL` | 01/2025 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\345635\\` |
| `412845` | `BL` | 12/2024 | 002 | Yes | 1.0 (Complete) | `F:\\NADI-Q data\\PPMI\\MRI image download_dataset\\PPMI\\412845\\` |

---

## 4. AUDIT FILE PATHS & ARTIFACTS
- **Download Checklist CSV**: [ppmi_mri_batch_001_download_batch_002_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_download_batch_002_v1.csv)
- **Active Validation Manifest**: [ppmi_mri_batch_001_download_validation_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_download_validation_v1.csv)
- **Authoritative Source Manifest**: [ppmi_mri_batch_001_corrected_v1.csv](file:///F:/NADI-Q%20data/PPMI/manifests/ppmi_mri_batch_001_corrected_v1.csv)
- **Download Checklist Report (Drive)**: [NADIQ_PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md](file:///F:/NADI-Q%20data/PPMI/reports/NADIQ_PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md)
- **Download Checklist Report (Repo)**: [PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md](file:///C:/Users/ayush/OneDrive/Desktop/Nadi-Q/reports/current/PPMI_BATCH_002_DOWNLOAD_CHECKLIST_V1.md)
"""
    
    with open(report_drive_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved to drive: {report_drive_path}")
    
    with open(report_repo_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved to repo: {report_repo_path}")

if __name__ == "__main__":
    main()

